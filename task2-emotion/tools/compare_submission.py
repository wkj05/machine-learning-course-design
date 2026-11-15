#!/usr/bin/env python3
"""逐项比对提交文件与官方样例（任务二）—— 只用标准库。

用法:
    python compare_submission.py <我的提交.csv> <官方样例.csv>

比对的是"格式"，不是"内容对不对"（样例的 Emotion 全是 0，没有参考价值）。
"""

import csv
import sys
from pathlib import Path

OK, NG = '[OK  ]', '[FAIL]'


def raw(path):
    b = Path(path).read_bytes()
    if b.startswith(b'\xef\xbb\xbf'):
        bom, b2 = True, b[3:]
    else:
        bom, b2 = False, b
    return b2, bom


def read_csv(path):
    with open(path, newline='', encoding='utf-8-sig') as f:
        return list(csv.reader(f))


def main():
    mine = Path(sys.argv[1])
    ref = Path(sys.argv[2])
    print(f'我的提交 : {mine}')
    print(f'DATA 样例: {ref}\n')

    fails = []

    def check(cond, title, a, b):
        print(f'  {OK if cond else NG}  {title}')
        print(f'         我的   = {a}')
        print(f'         DATA   = {b}')
        if not cond:
            fails.append(title)

    mrows, rrows = read_csv(mine), read_csv(ref)
    mh, rh = mrows[0], rrows[0]
    md, rd = mrows[1:], rrows[1:]

    check(mh == rh, '表头完全一致（含大小写、顺序）', mh, rh)
    check(len(mh) == len(rh), '列数一致', [len(mh)], [len(rh)])
    check(len(md) == len(rd), '数据行数一致', len(md), len(rd))

    mi = [r[0] for r in md]
    ri = [r[0] for r in rd]
    check(set(mi) == set(ri), 'ID 集合完全一致',
          f'{len(set(mi))} 个唯一 ID', f'{len(set(ri))} 个唯一 ID')
    # 注意：行数不等时 zip() 会在前缀全同的情况下直接耗尽，next() 会抛 StopIteration。
    # 所以这里必须先取默认值，再判断是"第几行起不同"还是"前缀相同、只是长短不同"。
    _first = next((i for i, (a, b) in enumerate(zip(mi, ri)) if a != b), None)
    if mi == ri:
        _detail = '一致'
    elif _first is None:
        _detail = f'前 {min(len(mi), len(ri))} 行相同，但行数不同'
    else:
        _detail = f'第 {_first + 2} 行起不同'
    check(mi == ri, 'ID 顺序与样例逐一相同', _detail, '基准')

    mb, mbom = raw(mine)
    rb, rbom = raw(ref)
    check(mb.count(b'\r\n') == rb.count(b'\r\n'), '换行符风格一致（CRLF 计数）',
          mb.count(b'\r\n'), rb.count(b'\r\n'))
    check(mbom == rbom, '是否带 UTF-8 BOM', mbom, rbom)
    check(mb.endswith(b'\n'), '结尾是否有换行', mb.endswith(b'\n'), rb.endswith(b'\n'))
    check(not mb.endswith(b'\n\n'), '结尾是否有空行',
          mb.endswith(b'\n\n'), rb.endswith(b'\n\n'))

    print(f'\n  [info] Emotion 取值')
    print(f'         我的 = {sorted({r[1] for r in md if len(r) > 1})}')
    print(f'         DATA = {sorted({r[1] for r in rd if len(r) > 1})}')

    print(f'\n  文件大小: 我的 {mine.stat().st_size} 字节 / DATA {ref.stat().st_size} 字节')

    if fails:
        print(f'\n结论：有 {len(fails)} 项不一致 -> {fails}')
        return 1
    print('\n结论：格式与 DATA 中的样例 CSV 一致，可以直接提交。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
