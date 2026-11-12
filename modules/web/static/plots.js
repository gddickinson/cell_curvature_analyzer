// Small SVG plots (no libraries): edge profiles (line) and intensity vs curvature (scatter with fit line).
import { esc } from './core.js';

const W = 640, H = 330, M = { l: 62, r: 16, t: 14, b: 46 };

export function niceTicks(lo, hi, n = 6) {
  if (!(hi > lo)) hi = lo + 1;
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * p).find(s => s >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}
export const fmt = v => (v === null || v === undefined || !Number.isFinite(v) ? '–'
  : Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : String(+v.toPrecision(4)));

function axes(sx, sy, xt, yt, xlabel, ylabel) {
  let g = '';
  for (const v of yt) g += `<line x1="${M.l}" x2="${W - M.r}" y1="${sy(v)}" y2="${sy(v)}" class="grid-l"/><text x="${M.l - 8}" y="${sy(v) + 4}" text-anchor="end">${fmt(v)}</text>`;
  for (const v of xt) g += `<line y1="${M.t}" y2="${H - M.b}" x1="${sx(v)}" x2="${sx(v)}" class="grid-l"/><text x="${sx(v)}" y="${H - M.b + 18}" text-anchor="middle">${fmt(v)}</text>`;
  g += `<rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}" class="axis-box"/>`;
  g += `<text x="${(M.l + W - M.r) / 2}" y="${H - 8}" text-anchor="middle" class="axis-label">${esc(xlabel)}</text>`;
  g += `<text transform="translate(14 ${(M.t + H - M.b) / 2}) rotate(-90)" text-anchor="middle" class="axis-label">${esc(ylabel)}</text>`;
  return g;
}

function scales(xs, ys, pad = 0.05) {
  let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  if (!(x1 > x0)) { x0 -= 1; x1 += 1; }
  if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  const dx = (x1 - x0) * pad, dy = (y1 - y0) * pad;
  x0 -= dx; x1 += dx; y0 -= dy; y1 += dy;
  const sx = v => M.l + (v - x0) / (x1 - x0) * (W - M.l - M.r), sy = v => H - M.b - (v - y0) / (y1 - y0) * (H - M.t - M.b);
  return { sx, sy, xt: niceTicks(x0, x1), yt: niceTicks(y0, y1), x0, x1, y0, y1 };
}

const legend = items => `<div class="legend">${items.map(s => `<span><i class="${s.dot ? 'dot' : ''}" style="background:${s.color}"></i>${esc(s.label)}</span>`).join('')}</div>`;

// series: [{x, y, color, label, dots?}]; null y values break the line.
export function linePlot(el, series, { xlabel = '', ylabel = '', title = 'plot' } = {}) {
  const pts = series.flatMap(s => s.x.map((x, i) => [x, s.y[i]])).filter(p => p[1] !== null && Number.isFinite(p[1]));
  if (!pts.length) { el.innerHTML = '<p class="mute small">Nothing to plot.</p>'; return; }
  const { sx, sy, xt, yt, y0, y1 } = scales(pts.map(p => p[0]), pts.map(p => p[1]), 0.04);
  let body = y0 < 0 && y1 > 0 ? `<line x1="${M.l}" x2="${W - M.r}" y1="${sy(0)}" y2="${sy(0)}" class="zero-l"/>` : '';
  for (const s of series) {
    let d = '', pen = false;
    s.x.forEach((x, i) => {
      const y = s.y[i];
      if (y === null || !Number.isFinite(y)) { pen = false; return; }
      d += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`; pen = true;
    });
    body += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="1.8"/>`;
    if (s.dots) s.x.forEach((x, i) => { const y = s.y[i]; if (y !== null && Number.isFinite(y)) body += `<circle cx="${sx(x).toFixed(1)}" cy="${sy(y).toFixed(1)}" r="2.4" fill="${typeof s.dots === 'function' ? s.dots(i) : s.color}"/>`; });
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="plot" role="img" aria-label="${esc(title)}">${axes(sx, sy, xt, yt, xlabel, ylabel)}${body}</svg>${legend(series.filter(s => s.label))}`;
}

// points: [{x, y, color}]; fit: {slope, intercept} or null.
export function scatterPlot(el, points, fit, { xlabel = '', ylabel = '', title = 'scatter', legendItems = [] } = {}) {
  if (!points.length) { el.innerHTML = '<p class="mute small">No valid points to plot.</p>'; return; }
  const { sx, sy, xt, yt, x0, x1 } = scales(points.map(p => p.x), points.map(p => p.y));
  let body = '';
  for (const p of points) body += `<circle cx="${sx(p.x).toFixed(1)}" cy="${sy(p.y).toFixed(1)}" r="2.6" fill="${p.color}" fill-opacity=".7"/>`;
  if (fit && Number.isFinite(fit.slope)) {
    body += `<line x1="${sx(x0)}" x2="${sx(x1)}" y1="${sy(fit.intercept + fit.slope * x0)}" y2="${sy(fit.intercept + fit.slope * x1)}" stroke="var(--accent)" stroke-width="2"/>`;
  }
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="plot" role="img" aria-label="${esc(title)}"><defs><clipPath id="clip-sc"><rect x="${M.l}" y="${M.t}" width="${W - M.l - M.r}" height="${H - M.t - M.b}"/></clipPath></defs>${axes(sx, sy, xt, yt, xlabel, ylabel)}<g clip-path="url(#clip-sc)">${body}</g></svg>${legend(legendItems)}`;
}
