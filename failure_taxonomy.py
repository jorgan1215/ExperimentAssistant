#!/usr/bin/env python3
"""
RL rollout 失败模式分类统计
============================
对每条 rollout，按三个环节的 对(1.0)/错(<1.0) 做完整 2x2x2 列联表：
  rc = reaction_center_content   （反应中心）
  rt = reaction_type_content     （反应类型）
  ex = reactants_exact           （反应物精确匹配）

输出：
  1) 全部 rollout 的 2x2x2 列联表（含占比）
  2) 组级统计：每组 8 条中反应物对几条；全错组/全对组占比
  3) 关键失败类别按 USPTO-50K 反应大类分层
  4) 早期 vs 晚期 step 的趋势对比
  5) "rc对+rt对+反应物错" 的近失分析（与 GT 的 Tanimoto 相似度）

用法:  python3 failure_taxonomy.py [log_file]
"""
import json, re, sys, statistics
from collections import Counter, defaultdict

try:
    from rdkit import RDLogger, Chem
    from rdkit.Chem import AllChem, DataStructs
    RDLogger.DisableLog('rdApp.*')
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

LOG = sys.argv[1] if len(sys.argv) > 1 else 'rl_rollout_2026-10-09.txt'
OK = 0.999  # content reward 判定"对"的阈值（部分正确会给 0.5/0.667 等分数）

USPTO_CLASS = {
    1: '杂原子烷基化/芳基化', 2: '酰化及相关', 3: 'C-C键形成', 4: '杂环形成',
    5: '保护', 6: '脱保护', 7: '还原', 8: '氧化', 9: '官能团互转(FGI)', 10: '官能团加成(FGA)',
}

recs = []
with open(LOG) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass
print(f'共解析 {len(recs)} 组 rollout 记录')
steps = sorted(r['global_step'] for r in recs)
print(f'step 范围: {steps[0]} .. {steps[-1]}\n')

RE_REACT = re.compile(r'<REACTANTS>(.*?)</REACTANTS>', re.S)

# ---------- 1) 2x2x2 列联表 ----------
combo = Counter()          # (rc_ok, rt_ok, ex_ok) -> count
combo_by_cls = defaultdict(Counter)
group_k_correct = Counter()  # 每组反应物正确的条数 0..8
n_rollouts = 0
near_sims = []               # rc对rt对但反应物错 的相似度
fmt_missing_react = 0

for r in recs:
    cls = r['ground_truth'].get('class')
    gt = r['ground_truth'].get('reactants_canonical', '')
    gm = Chem.MolFromSmiles(gt) if HAS_RDKIT else None
    gfp = AllChem.GetMorganFingerprintAsBitVect(gm, 2, 2048) if gm else None

    ex_list = r['raw_rewards']['reactants_exact']
    rc_list = r['raw_rewards']['reaction_center_content']
    rt_list = r['raw_rewards']['reaction_type_content']

    k = 0
    for i in range(len(ex_list)):
        ex_ok = ex_list[i] is not None and ex_list[i] >= OK
        rc_ok = rc_list[i] is not None and rc_list[i] >= OK
        rt_ok = rt_list[i] is not None and rt_list[i] >= OK
        combo[(rc_ok, rt_ok, ex_ok)] += 1
        combo_by_cls[cls][(rc_ok, rt_ok, ex_ok)] += 1
        n_rollouts += 1
        k += ex_ok

        if rc_ok and rt_ok and not ex_ok and HAS_RDKIT and gfp:
            m = RE_REACT.search(r['answers'][i])
            if not m:
                fmt_missing_react += 1
            else:
                parts = [p.strip() for p in m.group(1).strip().split('.') if p.strip()]
                mols = [Chem.MolFromSmiles(p) for p in parts]
                if mols and all(mols):
                    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, 2048) for x in mols]
                    near_sims.append(max(DataStructs.BulkTanimotoSimilarity(gfp, fps)))
    group_k_correct[k] += 1

def mark(b): return '✓' if b else '✗'

print('=' * 64)
print('1) 全部 rollout 的 rc × rt × 反应物 列联表 (n=%d)' % n_rollouts)
print('=' * 64)
print(f'{"rc":^4}{"rt":^4}{"反应物":^6}{"数量":>8}{"占比":>9}   含义')
MEANING = {
    (1,1,1): '全对 ✔',
    (1,1,0): '【核心短板】机理判断对、反应物写错',
    (1,0,0): '反应中心对，类型判断错',
    (0,1,0): '类型对，反应中心找错',
    (1,0,1): '类型错但反应物蒙对（罕见）',
    (0,1,1): '中心错但反应物蒙对（罕见）',
    (0,0,1): '推理全错但反应物对（罕见）',
    (0,0,0): '全错（机理层面就没搞懂）',
}
for key in [(1,1,1),(1,1,0),(1,0,0),(0,1,0),(0,0,0),(1,0,1),(0,1,1),(0,0,1)]:
    n = combo.get(key, 0)
    print(f'{mark(key[0]):^4}{mark(key[1]):^4}{mark(key[2]):^6}{n:>8}{100*n/n_rollouts:>8.1f}%   {MEANING[key]}')

ex_total = sum(v for (rc,rt,ex),v in combo.items() if ex)
print(f'\n反应物精确匹配总准确率: {ex_total}/{n_rollouts} = {100*ex_total/n_rollouts:.1f}%')

print()
print('=' * 64)
print('2) 组级统计（每组 8 条 rollout 中反应物正确的条数分布）')
print('=' * 64)
n_groups = sum(group_k_correct.values())
for k in range(9):
    n = group_k_correct.get(k, 0)
    bar = '█' * int(40 * n / max(1, n_groups))
    print(f'  对{k}条: {n:>4} 组 ({100*n/n_groups:>5.1f}%) {bar}')
print(f'\n  ★ 全错组（8条反应物全错，GRPO 无正样本信号）: {group_k_correct[0]} ({100*group_k_correct[0]/n_groups:.1f}%)')
print(f'  ★ 全对组（学不到东西的简单样本）:            {group_k_correct[8]} ({100*group_k_correct[8]/n_groups:.1f}%)')

print()
print('=' * 64)
print('3) 关键失败类别 × 反应大类分层（只看样本量 ≥40 的类别）')
print('=' * 64)
hdr = f'{"类别":<22}{"总数":>6}{"ex错|rc✓rt✓":>12}{"全错":>8}{"ex准确率":>9}'
print(hdr)
rows = []
for cls, c in combo_by_cls.items():
    tot = sum(c.values())
    if tot < 40: continue
    core = c.get((True,True,False), 0)
    allbad = c.get((False,False,False), 0)
    acc = sum(v for (rc,rt,ex),v in c.items() if ex) / tot
    name = f'{cls}-{USPTO_CLASS.get(cls,"?")}'
    rows.append((acc, name, tot, core, allbad))
rows.sort()
for acc, name, tot, core, allbad in rows:
    print(f'{name:<22}{tot:>6}{core:>7}({100*core/tot:>4.1f}%){allbad:>5}({100*allbad/tot:>4.1f}%){100*acc:>7.1f}%')

print()
print('=' * 64)
print('4) 早期 vs 晚期趋势（按 step 三等分）')
print('=' * 64)
sorted_steps = sorted(set(steps))
third = max(1, len(sorted_steps)//3)
phases = {'早期': set(sorted_steps[:third]), '晚期': set(sorted_steps[-third:])}
combo_ph = {p: Counter() for p in phases}
for r in recs:
    for p, ss in phases.items():
        if r['global_step'] in ss:
            for i in range(len(r['raw_rewards']['reactants_exact'])):
                ex = (r['raw_rewards']['reactants_exact'][i] or 0) >= OK
                rc = (r['raw_rewards']['reaction_center_content'][i] or 0) >= OK
                rt = (r['raw_rewards']['reaction_type_content'][i] or 0) >= OK
                combo_ph[p][(rc, rt, ex)] += 1
print(f'{"":<26}{"早期":>10}{"晚期":>10}{"变化":>10}')
for key, label in [((1,1,1),'全对'), ((1,1,0),'rc✓rt✓但反应物错'), ((0,0,0),'全错')]:
    e = combo_ph['早期'].get(key,0); l = combo_ph['晚期'].get(key,0)
    te = sum(combo_ph['早期'].values()); tl = sum(combo_ph['晚期'].values())
    pe, pl = 100*e/te, 100*l/tl
    print(f'{label:<26}{pe:>9.1f}%{pl:>9.1f}%{pl-pe:>+9.1f}pp')

print()
print('=' * 64)
print('5) "rc✓ rt✓ 但反应物错" 的近失分析（最可挽救的部分）')
print('=' * 64)
n_core = combo.get((True,True,False), 0)
print(f'该类别共 {n_core} 条 rollout，其中因缺少 </REACTANTS>（截断）无法评估的: {fmt_missing_react}')
if near_sims:
    near_sims.sort()
    n = len(near_sims)
    print(f'有效相似度评估 {n} 条（与 GT 最大片段 Tanimoto）:')
    print(f'  mean={statistics.mean(near_sims):.3f}  median={statistics.median(near_sims):.3f}  '
          f'p25={near_sims[n//4]:.3f}  p75={near_sims[3*n//4]:.3f}')
    for th in (0.9, 0.8, 0.7, 0.5):
        c = sum(1 for s in near_sims if s >= th)
        print(f'  相似度 ≥ {th}: {c} 条 ({100*c/n:.1f}%)  <- {"几乎就是对的，相似度奖励可挽救" if th==0.7 else ""}')
elif not HAS_RDKIT:
    print('（未安装 rdkit，跳过相似度分析）')
