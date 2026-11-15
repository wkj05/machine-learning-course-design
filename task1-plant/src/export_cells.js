// 把 notebook 的每个 code cell 导出成单独 .py，便于用 CPython 做语法检查
const fs = require('fs');
const path = require('path');
const nb = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const outDir = process.argv[3];
fs.mkdirSync(outDir, { recursive: true });
let n = 0;
nb.cells.forEach((c, i) => {
  if (c.cell_type !== 'code') return;
  const src = Array.isArray(c.source) ? c.source.join('') : c.source;
  const f = path.join(outDir, `cell_${String(i).padStart(2, '0')}.py`);
  fs.writeFileSync(f, src, 'utf8');
  console.log(`cell ${i}: ${src.split('\n').length} lines -> ${path.basename(f)}`);
  n++;
});
console.log(`exported ${n} code cells`);
