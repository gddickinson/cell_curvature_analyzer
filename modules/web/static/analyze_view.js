// Drawing for the Analyze page: the frame with its edge, coloured sample points, sampling rectangles, and the
// per-frame profiles. Coordinates in results are [row, col] pixel centres; canvas x = (col + 0.5) * scale.
import { esc } from './core.js';
import { linePlot, fmt } from './plots.js';

const CONCAVE = [79, 157, 255], MID = [154, 163, 181], CONVEX = [255, 138, 61];
const mix = (a, b, t) => a.map((v, i) => Math.round(v + (b[i] - v) * t));

// Diverging colour for a value in [-1, 1]: concave (-) blue, 0 grey, convex (+) orange.
export function curvColor(v) {
  const t = Math.max(-1, Math.min(1, v || 0));
  const c = t < 0 ? mix(MID, CONCAVE, -t) : mix(MID, CONVEX, t);
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export function colourValues(frame, by) {
  if (by === 'sign') return frame.sign;
  if (by === 'magnitude') { const mx = Math.max(...frame.magnitude) || 1; return frame.magnitude.map(m => m / mx); }
  return frame.normalized;
}

function percentile(arr, q) {
  const step = Math.max(1, Math.floor(arr.length / 20000)), s = [];
  for (let i = 0; i < arr.length; i += step) s.push(arr[i]);
  s.sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))];
}

// Grey image with 0.5-99.5 percentile contrast, scaled by an integer factor for crisp overlays.
export function drawFrame(canvas, pair, result, t, { colorBy = 'normalized', regions = true, maskTint = true } = {}) {
  const { width: w, height: h } = pair, img = pair.images[t], mask = pair.masks[t];
  const s = Math.max(1, Math.min(8, Math.floor(900 / Math.max(w, h)) || 1));
  canvas.width = w * s; canvas.height = h * s;
  const lo = percentile(img, 0.005), hi = percentile(img, 0.995), span = hi - lo || 1;
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const octx = off.getContext('2d'), id = octx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const g = Math.max(0, Math.min(255, Math.round((img[i] - lo) / span * 255)));
    const tint = maskTint && mask[i] ? 18 : 0;
    id.data[4 * i] = g; id.data[4 * i + 1] = Math.min(255, g + tint); id.data[4 * i + 2] = g; id.data[4 * i + 3] = 255;
  }
  octx.putImageData(id, 0, 0);
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off, 0, 0, w * s, h * s);
  const f = result && result.frames[t];
  if (!f || f.error) return s;
  const X = p => (p[1] + 0.5) * s, Y = p => (p[0] + 0.5) * s;
  ctx.lineWidth = Math.max(1, s / 2);
  ctx.strokeStyle = 'rgba(92,242,196,.85)';
  ctx.beginPath();
  f.contour.forEach((p, i) => (i ? ctx.lineTo(X(p), Y(p)) : ctx.moveTo(X(p), Y(p))));
  ctx.stroke();
  if (regions) {
    f.regions.forEach((v, i) => {
      if (result.params.exclude_endpoints && (i === 0 || i === f.regions.length - 1)) return;
      ctx.strokeStyle = f.recovered[i] ? 'rgba(255,210,63,.8)' : f.valid[i] ? 'rgba(255,255,255,.45)' : 'rgba(255,92,122,.75)';
      ctx.setLineDash(f.valid[i] ? [] : [3, 3]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      // vertices share the pixel-centre coordinates of the points (draw.polygon tests pixel centres)
      v.forEach((p, k) => (k ? ctx.lineTo((p[1] + 0.5) * s, (p[0] + 0.5) * s) : ctx.moveTo((p[1] + 0.5) * s, (p[0] + 0.5) * s)));
      ctx.closePath(); ctx.stroke();
    });
    ctx.setLineDash([]);
  }
  const vals = colourValues(f, colorBy), r = Math.max(2.5, s * 1.1);
  f.points.forEach((p, i) => {
    ctx.beginPath(); ctx.arc(X(p), Y(p), r, 0, 2 * Math.PI);
    ctx.fillStyle = curvColor(vals[i]); ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = f.valid[i] ? '#000' : '#ff5c7a'; ctx.stroke();
  });
  // mark the first point (start of the walk) so the direction along the profile plots is visible
  const p0 = f.points[0];
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(X(p0), Y(p0), r + 3, 0, 2 * Math.PI); ctx.stroke();
  return s;
}

export function frameProfiles(curvEl, intEl, f, colorBy) {
  if (!f || f.error) { curvEl.innerHTML = intEl.innerHTML = `<p class="mute small">${esc(f ? f.error : 'No frame')}.</p>`; return; }
  const idx = f.points.map((_, i) => i), vals = colourValues(f, colorBy);
  const label = { normalized: 'normalized curvature (−1..1)', sign: 'curvature sign', magnitude: 'curvature magnitude' }[colorBy];
  linePlot(curvEl, [{ x: idx, y: colorBy === 'magnitude' ? f.magnitude : vals, color: '#9aa3b5', dots: i => curvColor(vals[i]) }],
    { xlabel: 'point along the edge (0 = ringed point)', ylabel: label, title: 'Curvature along the edge' });
  linePlot(intEl, [{ x: idx, y: f.intensities, color: '#5cf2c4', dots: true, label: 'mean intensity in the sampling rectangle (gaps: rejected)' }],
    { xlabel: 'point along the edge', ylabel: 'intensity', title: 'Intensity along the edge' });
}

export function statsTable(stats) {
  const row = (name, s) => `<tr><td>${name}</td><td class="num">${s.sample_size}</td><td class="num">${fmt(s.r_squared)}</td>
    <td class="num">${fmt(s.p_value)}</td><td class="num">${fmt(s.slope)}</td><td class="num">${fmt(s.intercept)}</td><td class="num">${fmt(s.standard_error)}</td></tr>`;
  return `<div class="scroll"><table><thead><tr><th>intensity ~ curvature</th><th class="num">n</th><th class="num">r²</th><th class="num">p</th>
    <th class="num">slope</th><th class="num">intercept</th><th class="num">SE slope</th></tr></thead><tbody>
    ${row('sign curvature', stats.sign)}${row('normalized curvature', stats.normalized)}</tbody></table></div>`;
}

// Max |a - b| over every number in two analysis results (for the JS-vs-Python agreement note).
export function maxDiff(a, b) {
  let d = 0;
  const walk = (x, y) => {
    if (typeof x === 'number' && typeof y === 'number') { d = Math.max(d, Math.abs(x - y) / Math.max(1, Math.abs(y))); return; }
    if (Array.isArray(x) && Array.isArray(y)) { if (x.length !== y.length) d = Infinity; else x.forEach((v, i) => walk(v, y[i])); return; }
    if (x && y && typeof x === 'object') for (const k of Object.keys(y)) if (k in x) walk(x[k], y[k]);
  };
  walk(a.frames, b.frames); walk(a.stats, b.stats); walk(a.movement, b.movement);
  return d;
}
