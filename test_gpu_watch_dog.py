import contextlib
from datetime import datetime, timedelta
import importlib.util
import os
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import gpu_watch_dog as watch


class ScheduleTests(unittest.TestCase):
    def test_relative_interval_and_none(self):
        finished = datetime(2026, 10, 24, 10, 17, 23)
        self.assertEqual(watch.next_requeue_time('@every 30m', finished),
                         datetime(2026, 10, 24, 10, 47, 23))
        self.assertIsNone(watch.next_requeue_time(None, finished))

    @unittest.skipUnless(importlib.util.find_spec('croniter'), 'croniter 未安装')
    def test_wall_clock_cron(self):
        self.assertEqual(
            watch.next_requeue_time('*/30 * * * *', datetime(2026, 10, 24, 10, 17)),
            datetime(2026, 10, 24, 10, 30))
        self.assertEqual(
            watch.next_requeue_time('0 9 * * mon-fri', datetime(2026, 10, 23, 10)),
            datetime(2026, 10, 26, 9))

    def test_invalid_configuration_rejected_before_launch(self):
        for expression in ('', '@every 0m', '@every -1h', 'bad cron', 30):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                watch.prepare_jobs([{'cmd': 'echo ok', 'mem_free': 1,
                                     'requeue_cron': expression}])
        with self.assertRaisesRegex(ValueError, '移除 f 前缀'):
            watch.prepare_jobs([{'cmd': 'CUDA_VISIBLE_DEVICES=None python train.py',
                                 'mem_free': 1}])

    def test_launch_sets_global_and_child_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            # 真正启动无害子进程，检查两次启动是否各自获得正确的 GPU 编号。
            for gpu in (2, 5):
                snippet = 'import os; print(os.environ["CUDA_VISIBLE_DEVICES"])'
                command = f'{shlex.quote(sys.executable)} -c {shlex.quote(snippet)}'
                if os.name != 'nt':
                    command = 'CUDA_VISIBLE_DEVICES={FREE_GPU_DEVICES} ' + command
                proc, log = watch.launch_training(command, gpu, directory)
                self.assertEqual(watch.FREE_GPU_DEVICES, gpu)
                self.assertEqual(proc.wait(timeout=10), 0)
                with open(log) as output:
                    self.assertEqual(output.read().strip(), str(gpu))


class WatchLoopTests(unittest.TestCase):
    def simulate(self, expression, returncode=0, remove=True, check_times=2,
                 duration=120, gpu_count=2, limit=2400):
        clock = {'seconds': 0, 'interrupted': False}
        launches = []
        epoch = datetime(2026, 10, 24, 10, 17)

        class FakeDatetime(datetime):
            @classmethod
            def now(cls):
                return epoch + timedelta(seconds=clock['seconds'])

        class Process:
            pid = 12345

            def __init__(self, started):
                self.started = started

            def poll(self):
                return returncode if clock['seconds'] >= self.started + duration else None

        def launch(cmd, gpu, directory):
            launches.append((clock['seconds'], gpu))
            return Process(clock['seconds']), 'unused.log'

        def sleep(seconds):
            clock['seconds'] += seconds
            if clock['seconds'] >= limit:
                clock['interrupted'] = True
                raise KeyboardInterrupt

        job = {'cmd': 'test-only', 'mem_free': 10,
               'remove_on_launch': remove, 'requeue_cron': expression}
        gpus = [{'index': i, 'name': 'fake', 'mem_used': 0, 'mem_total': 100}
                for i in range(gpu_count)]
        with contextlib.ExitStack() as stack:
            for name, value in [('JOBS', [job]), ('CHECK_INTERVAL', 60),
                                ('CHECK_TIMES', check_times), ('GPU_FILTER', None),
                                ('datetime', FakeDatetime)]:
                stack.enter_context(patch.object(watch, name, value))
            stack.enter_context(patch.object(watch, 'setup_logging'))
            stack.enter_context(patch.object(watch, 'find_nvidia_smi', return_value='fake'))
            stack.enter_context(patch.object(watch, 'query_gpus', return_value=gpus))
            stack.enter_context(patch.object(watch, 'launch_training', side_effect=launch))
            stack.enter_context(patch.object(watch.time, 'sleep', side_effect=sleep))
            watch.main()
        return launches, clock

    def test_success_requeues_after_completion_then_waits_for_gpu_streak(self):
        launches, clock = self.simulate('@every 30m')
        self.assertEqual(launches, [(60, 0), (2040, 0)])
        self.assertTrue(clock['interrupted'])

    def test_schedule_overrides_remove_false_to_prevent_concurrent_duplicates(self):
        launches, _ = self.simulate('@every 30m', remove=False)
        self.assertEqual(launches, [(60, 0), (2040, 0)])

    def test_failure_does_not_requeue_but_watchdog_stays_alive(self):
        launches, clock = self.simulate('@every 30m', returncode=1)
        self.assertEqual(launches, [(60, 0)])
        self.assertTrue(clock['interrupted'])

    def test_none_finishes_and_exits(self):
        for code in (0, 1):
            with self.subTest(returncode=code):
                launches, clock = self.simulate(None, returncode=code)
                self.assertEqual(launches, [(60, 0)])
                self.assertFalse(clock['interrupted'])

    def test_still_running_task_is_not_requeued(self):
        launches, clock = self.simulate('@every 30m', duration=3600)
        self.assertEqual(launches, [(60, 0)])
        self.assertTrue(clock['interrupted'])

    def test_legacy_resident_task_still_repeats(self):
        launches, _ = self.simulate(None, remove=False, gpu_count=1, limit=600)
        self.assertGreater(len(launches), 1)


if __name__ == '__main__':
    unittest.main()
