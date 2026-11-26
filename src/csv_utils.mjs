import fs from 'node:fs';

export function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++; }
      else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ',') { row.push(field); field = ''; }
    else if (c === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
    else field += c;
  }
  if (field.length || row.length) { row.push(field.replace(/\r$/, '')); rows.push(row); }
  const headers = rows.shift() ?? [];
  return rows.filter(r => !(r.length === 1 && r[0] === '')).map(values => Object.fromEntries(headers.map((h, i) => [h, values[i] ?? ''])));
}

export function readCsv(file) { return parseCsv(fs.readFileSync(file, 'utf8')); }

export function writeCsv(file, rows, columns = null) {
  const headers = columns ?? [...new Set(rows.flatMap(r => Object.keys(r)))];
  const esc = value => {
    const s = value === null || value === undefined ? '' : String(value);
    return /[",\r\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
  };
  fs.mkdirSync(file.substring(0, Math.max(file.lastIndexOf('/'), file.lastIndexOf('\\'))), { recursive: true });
  fs.writeFileSync(file, [headers.join(','), ...rows.map(r => headers.map(h => esc(r[h])).join(','))].join('\n') + '\n', 'utf8');
}

