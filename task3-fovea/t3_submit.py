# -*- coding: utf-8 -*-
"""任务三 提交物：按 sample_submission.csv 的**行序与行数**原样写值，并做格式自检。

用法：
  python t3_submit.py --inspect
  python t3_submit.py --pred pred.json --out submission.csv
  python t3_submit.py --check  submission.csv

pred.json 形如 {"81": [x, y], "82": [x, y], ...}，键是图号（int 或 str 均可）。
"""
import argparse, json
from pathlib import Path

ROOT = Path(r'C:\Users\24625\Documents\deepseek-harness\default-workspace\_inbox\task3\detection')
SAMPLE = ROOT / 'sample_submission.csv'
ID_COL, VAL_COL = 'ImageID', 'value'


def read_bytes(p):
    return Path(p).read_bytes()


def parse_ids():
    """返回 [(image_no:int, axis:'X'|'Y', raw_id:str)]，严格按文件行序。"""
    txt = SAMPLE.read_text(encoding='utf-8-sig')
    lines = txt.splitlines()
    assert lines[0].strip() == f'{ID_COL},{VAL_COL}', f'表头异常: {lines[0]!r}'
    out = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        sid = ln.split(',')[0].strip()
        head, _, axis = sid.rpartition('_Fovea_')
        out.append((int(head), axis, sid))
    return out


def inspect():
    b = read_bytes(SAMPLE)
    print('=== sample_submission.csv 格式事实 ===')
    print('bytes      =', len(b))
    print('BOM        =', b[:3] == b'\xef\xbb\xbf')
    print('CRLF 行数  =', b.count(b'\r\n'))
    print('裸 LF 行数 =', b.count(b'\n') - b.count(b'\r\n'))
    print('结尾换行   =', b.endswith(b'\r\n'))
    txt = b.decode('utf-8-sig')
    lines = txt.split('\r\n')
    print('总行数(含表头) =', len([l for l in lines if l.strip()]))
    ids = parse_ids()
    print('数据行数   =', len(ids))
    print('图号       =', sorted({i for i, _, _ in ids}))
    print('每图轴序   =', [a for _, a, _ in ids[:4]], '...')
    assert len(ids) == 2 * len({i for i, _, _ in ids}), '图号不是成对出现'
    for k in range(0, len(ids), 2):
        assert ids[k][0] == ids[k + 1][0] and ids[k][1] == 'X' and ids[k + 1][1] == 'Y'
    print('成对校验   = OK（每图 X 在前、Y 在后）')


def build(pred, out):
    ids = parse_ids()
    missing = sorted({i for i, _, _ in ids} - set(pred))
    if missing:
        raise SystemExit(f'缺预测: {missing}')
    buf = [f'{ID_COL},{VAL_COL}']
    for i, axis, sid in ids:
        x, y = pred[i]
        buf.append(f'{sid},{float(x if axis == "X" else y):.6f}')
    data = ('\r\n'.join(buf) + '\r\n').encode('utf-8')
    Path(out).write_bytes(data)
    print(f'[written] {out}  {len(data)} bytes')


def check(p):
    b = read_bytes(p)
    s = read_bytes(SAMPLE)
    errs, warns = [], []
    if b[:3] == b'\xef\xbb\xbf':
        errs.append('有 UTF-8 BOM')
    if not b.endswith(b'\r\n'):
        errs.append('结尾没有换行')
    lf = b.count(b'\n') - b.count(b'\r\n')
    if lf:
        errs.append(f'有 {lf} 个裸 LF')
    txt = b.decode('utf-8-sig')
    lines = [l for l in txt.split('\r\n') if l.strip()]
    if lines[0].strip() != f'{ID_COL},{VAL_COL}':
        errs.append(f'表头错: {lines[0]!r}')
    if len(lines) - 1 != len(parse_ids()):
        errs.append(f'数据行数 {len(lines)-1} != 样例 {len(parse_ids())}')
    ref = [i for i, _, _ in parse_ids()]
    got, vals = [], []
    for ln in lines[1:]:
        parts = ln.split(',')
        if len(parts) != 2:
            errs.append(f'列数不是 2: {ln!r}')
            continue
        head, _, axis = parts[0].strip().rpartition('_Fovea_')
        got.append(int(head))
        try:
            vals.append(float(parts[1]))
        except ValueError:
            errs.append(f'不是数字: {ln!r}')
    if got != ref:
        errs.append('ImageID 的行序与样例不一致（按行序评分会被判错）')
    if len(vals) != len(got):
        errs.append('有非数字值')
    if vals:
        import statistics as st
        print(f'\n值域: X {min(vals[0::2]):.1f}~{max(vals[0::2]):.1f}  '
              f'Y {min(vals[1::2]):.1f}~{max(vals[1::2]):.1f}')
        print(f'均值: X {st.mean(vals[0::2]):.1f}  Y {st.mean(vals[1::2]):.1f}')
    print(f'bytes: 本文件 {len(b)} / 样例 {len(s)}  {"一致" if len(b)==len(s) else "不同（仅长度不同不一定是错，值位数可能不同）"}')
    print(f'CRLF: 本文件 {b.count(b"\r\n")} / 样例 {s.count(b"\r\n")}')
    if errs:
        print('\n错误:')
        for e in errs:
            print('  ✗', e)
    for w in warns:
        print('  ! ', w)
    print(f'\n结果: {len(errs)} 错误, {len(warns)} 警告')
    return not errs


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--inspect', action='store_true')
    ap.add_argument('--pred')
    ap.add_argument('--out')
    ap.add_argument('--check')
    a = ap.parse_args()
    if a.inspect:
        inspect()
    if a.pred:
        build({int(k): v for k, v in json.loads(Path(a.pred).read_text(encoding='utf-8')).items()},
              a.out or 'submission.csv')
    if a.check:
        ok = check(a.check)
        raise SystemExit(0 if ok else 1)
