#!/usr/bin/env python3
"""提交文件自检（任务二：人脸表情识别）—— 只用标准库，不需要 pandas/numpy。

用法:
    python check_submission.py <你的提交.csv> [--sample <官方样例.csv>]

检查项（任一项不过就 exit 1）:
    1. 文件存在且能被解析为 CSV
    2. 表头恰好是 ID,Emotion（大小写、顺序、有无多余列）
    3. 行数与官方样例一致（去掉表头）
    4. ID 集合与官方样例完全一致（多一个少一个都不行）
    5. ID 无重复、无空值
    6. Emotion 无空值，且全部是 0..5 的整数
    7. 类别分布是否退化（某个类别一张都没有 / 全填同一个类别）

最后会打印类别分布和与样例的行顺序差异提示。
"""

import argparse
import csv
import sys
from collections import Counter

# 编号由任务说明规定：0=愤怒 1=恐惧 2=快乐 3=悲伤 4=惊讶 5=中性
# 注意：这与公开 FER2013 的编号不同，以任务说明为准。
CLASSES = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
CN = {'Angry': '愤怒', 'Fear': '恐惧', 'Happy': '快乐',
      'Sad': '悲伤', 'Surprise': '惊讶', 'Neutral': '中性'}


def read_csv(path):
    # utf-8-sig 吃掉可能的 BOM（Excel 另存常带 BOM，会让第一列列名变成 '\ufeffID'）
    with open(path, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError('文件是空的')
    return rows[0], rows[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('submission')
    ap.add_argument('--sample', default=None, help='官方样例提交 CSV（用于比对 ID 集合与行数）')
    args = ap.parse_args()

    errors, warns = [], []

    try:
        header, rows = read_csv(args.submission)
    except Exception as e:
        print(f'[FATAL] 读不了 {args.submission}: {type(e).__name__}: {e}')
        return 1

    print(f'文件     : {args.submission}')
    print(f'表头     : {header}')
    print(f'数据行数 : {len(rows)}')

    # --- 2. 表头 ---
    if [h.strip() for h in header] != ['ID', 'Emotion']:
        errors.append(f'表头应为 ["ID", "Emotion"]，实际是 {header}')

    ncol = len(header)
    ids = [r[0].strip() for r in rows if len(r) >= 1]
    vals = [r[1].strip() for r in rows if len(r) >= 2]
    bad_rows = [i for i, r in enumerate(rows, 2) if len(r) != ncol]
    if bad_rows:
        errors.append(f'有 {len(bad_rows)} 行的列数与表头不符，例如第 {bad_rows[:5]} 行')

    # --- 5/6. 空值 ---
    empty_ids = [i for i, v in enumerate(ids, 2) if v == '']
    if empty_ids:
        errors.append(f'ID 有 {len(empty_ids)} 个空值，例如第 {empty_ids[:5]} 行')
    if len(vals) != len(rows):
        errors.append(f'有 {len(rows) - len(vals)} 行缺少 Emotion 列')
    else:
        empty_vals = [i for i, v in enumerate(vals, 2) if v == '']
        if empty_vals:
            errors.append(f'Emotion 有 {len(empty_vals)} 个空值，例如第 {empty_vals[:5]} 行')

    # --- 5. 重复 ---
    dup = [k for k, c in Counter(ids).items() if c > 1]
    if dup:
        errors.append(f'ID 有 {len(dup)} 个重复值，例如 {dup[:5]}')

    # --- 6. 合法标签：必须是 0..5 的整数 ---
    bad_vals = sorted({v for v in vals if v not in {str(i) for i in range(len(CLASSES))}})
    if bad_vals:
        errors.append(f'出现非法 Emotion 值: {bad_vals[:10]}'
                      f'（合法值只有 {list(range(len(CLASSES)))} 的整数字面量）')

    # --- 扩展名 ---
    no_ext = [v for v in ids if '.' not in v]
    if no_ext:
        errors.append(f'有 {len(no_ext)} 个 ID 没有扩展名（必须带 .jpg），例如 {no_ext[:5]}')

    # --- 3/4. 与官方样例比对 ---
    if args.sample:
        try:
            shead, srows = read_csv(args.sample)
            sidx = next(i for i, h in enumerate(shead) if h.strip().lower() == 'id')
            sample_ids = [r[sidx].strip() for r in srows]
            print(f'\n样例     : {args.sample}   表头={shead}  行数={len(sample_ids)}')
            if len(rows) != len(sample_ids):
                errors.append(f'行数应为 {len(sample_ids)}，实际 {len(rows)}')
            miss = set(sample_ids) - set(ids)
            extra = set(ids) - set(sample_ids)
            if miss:
                errors.append(f'缺少 {len(miss)} 个样例中的 ID，例如 {sorted(miss)[:5]}')
            if extra:
                errors.append(f'多出 {len(extra)} 个样例中不存在的 ID，例如 {sorted(extra)[:5]}')
            if not miss and not extra and ids != sample_ids:
                warns.append('ID 集合一致但行顺序与样例不同（Kaggle 一般按 ID 匹配，不影响；'
                             '按样例顺序重排更保险）')
        except Exception as e:
            errors.append(f'读样例 {args.sample} 失败: {type(e).__name__}: {e}')

    # --- 7. 类别分布 ---
    cnt = Counter(vals)
    print('\n类别分布:')
    for i, c in enumerate(CLASSES):
        n = cnt.get(str(i), 0)
        bar = '#' * int(60 * n / max(1, len(vals)))
        print(f'  {i} {c:<9s} {CN[c]}  {n:5d}  {bar}')
    if len(cnt) == 1:
        warns.append(f'所有行都是同一个类别 ({vals[0]})，模型大概率没生效')
    zero = [c for i, c in enumerate(CLASSES) if cnt.get(str(i), 0) == 0]
    if zero and len(cnt) > 1:
        warns.append(f'这些类别一张都没预测到: {zero}')

    # --- 汇总 ---
    print()
    for w in warns:
        print(f'[WARN]  {w}')
    if errors:
        for e in errors:
            print(f'[FAIL]  {e}')
        print(f'\n结论：不通过（{len(errors)} 个错误，{len(warns)} 个警告）')
        return 1
    print(f'结论：通过（0 个错误，{len(warns)} 个警告）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
