// 把 src/*.nb.txt 转成 Jupyter notebook (.ipynb)
// 约定：以 "#@cell markdown" 或 "#@cell code" 开头的行是单元格分隔符。
const fs = require('fs');
const path = require('path');

const srcPath = process.argv[2];
const outPath = process.argv[3];
if (!srcPath || !outPath) {
  console.error('usage: node build_nb.js <src.nb.txt> <out.ipynb>');
  process.exit(2);
}

const text = fs.readFileSync(srcPath, 'utf8').replace(/\r\n/g, '\n');
const lines = text.split('\n');

const cells = [];
let cur = null;
for (const line of lines) {
  const m = /^#@cell\s+(markdown|code)\s*$/.exec(line);
  if (m) {
    if (cur) cells.push(cur);
    cur = { type: m[1], lines: [] };
    continue;
  }
  if (cur) cur.lines.push(line);
}
if (cur) cells.push(cur);

function toSource(arr) {
  // 去掉首尾空行，其余每行补回 \n
  let a = arr.slice();
  while (a.length && a[0].trim() === '') a.shift();
  while (a.length && a[a.length - 1].trim() === '') a.pop();
  return a.map((l, i) => (i === a.length - 1 ? l : l + '\n'));
}

const nb = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: { language: 'python', display_name: 'Python 3', name: 'python3' },
    language_info: {
      name: 'python',
      version: '3.12.13',
      mimetype: 'text/x-python',
      codemirror_mode: { name: 'ipython', version: 3 },
      pygments_lexer: 'ipython3',
      nbconvert_exporter: 'python',
      file_extension: '.py',
    },
  },
  // 注意：nbformat >= 4.5 规定每个 cell 必须有 id，否则 Jupyter / Kaggle 打开时
  // 会做 schema 校验并报 "Notebook validation failed"，界面上满屏红字。
  // id 只允许 [a-zA-Z0-9-_]+，长度 1..64，且不能重复。
  cells: cells.map((c, i) => ({
    id: `cell-${String(i).padStart(2, '0')}`,
    cell_type: c.type,
    metadata: {},
    source: toSource(c.lines),
    ...(c.type === 'code' ? { outputs: [], execution_count: null } : {}),
  })),
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(nb, null, 1), 'utf8');

const nCode = nb.cells.filter((c) => c.cell_type === 'code').length;
const nMd = nb.cells.length - nCode;
console.log(`wrote ${outPath}`);
console.log(`cells: ${nb.cells.length} (code=${nCode}, markdown=${nMd})`);
console.log(`bytes: ${fs.statSync(outPath).size}`);
