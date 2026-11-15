# -*- coding: utf-8 -*-
"""提交文件与 DATA 中样例 CSV 的逐项结构比对（纯标准库）。

用法:
    python compare_submission.py <我的提交.csv> <DATA中的样例.csv>

比对的是"格式"而不是"内容"：表头字节、行数、ID 序列是否逐一相同、
编码/BOM、换行符、结尾是否有空行、有没有多余索引列。
"""
import sys
from pathlib import Path

import csv


def probe(path):
    raw = Path(path).read_bytes()
    info = {}
    info['bytes'] = len(raw)
    info['bom'] = raw[:3] == b'\xef\xbb\xbf'
    body = raw[3:] if info['bom'] else raw
    info['crlf'] = body.count(b'\r\n')
    info['lf'] = body.count(b'\n')
    info['final_newline'] = body.endswith(b'\n') or body.endswith(b'\r\n')
    info['blank_tail'] = body.endswith(b'\n\n') or body.endswith(b'\r\n\r\n')
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    info['header'] = rows[0] if rows else []
    info['n_rows'] = len(rows) - 1
    info['ids'] = [r[0] for r in rows[1:] if r]
    info['cats'] = [r[1] for r in rows[1:] if r]
    info['n_cols'] = sorted({len(r) for r in rows})
    return info


def main():
    mine = Path(sys.argv[1])
    ref = Path(sys.argv[2])
    a, b = probe(mine), probe(ref)

    print(f'我的提交 : {mine}')
    print(f'DATA 样例: {ref}')
    print()

    ok = True

    def line(name, va, vb, good):
        nonlocal ok
        flag = 'OK  ' if good else 'FAIL'
        if not good:
            ok = False
        print(f'  [{flag}] {name}')
        print(f'         我的   = {va!r}')
        print(f'         DATA   = {vb!r}')

    line('表头完全一致（含大小写、顺序）', a['header'], b['header'], a['header'] == b['header'])
    line('列数一致', a['n_cols'], b['n_cols'], a['n_cols'] == b['n_cols'])
    line('数据行数一致', a['n_rows'], b['n_rows'], a['n_rows'] == b['n_rows'])

    same_set = set(a['ids']) == set(b['ids'])
    same_order = a['ids'] == b['ids']
    line('ID 集合完全一致', f'{len(set(a["ids"]))} 个唯一 ID', f'{len(set(b["ids"]))} 个唯一 ID', same_set)
    line('ID 顺序与样例逐一相同', '一致' if same_order else '不同', '基准', same_order)

    line('换行符风格一致（CRLF 计数）', a['crlf'], b['crlf'],
         (a['crlf'] > 0) == (b['crlf'] > 0))
    line('是否带 UTF-8 BOM', a['bom'], b['bom'], a['bom'] == b['bom'])
    line('结尾是否有换行', a['final_newline'], b['final_newline'],
         a['final_newline'] == b['final_newline'])
    line('结尾是否有空行', a['blank_tail'], b['blank_tail'], a['blank_tail'] == b['blank_tail'])

    # 样例 CSV 是常数填充的（全填 Sugar beet），所以"我的类别是否是样例类别的子集"
    # 这个检查天然会失败，这里只作信息展示，不计入结论。
    print('\n  [info] Category 取值（样例是常数填充，只含 Sugar beet，故不做子集判断）')
    print(f'         我的 = {sorted(set(a["cats"]))}')
    print(f'         DATA = {sorted(set(b["cats"]))}')

    print(f'\n  文件大小: 我的 {a["bytes"]} 字节 / DATA {b["bytes"]} 字节')

    print('\n' + ('结论：格式与 DATA 中的样例 CSV 一致，可以直接提交。' if ok
                  else '结论：存在格式差异，见上面 FAIL 项。'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
