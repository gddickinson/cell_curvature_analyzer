// Browser port of modules/edge_curvature.py: open/closed edges, point spacing ("equal" / "legacy") and curvature
// (Gaussian smoothing along the edge, or the legacy three-point rule; dead-band for straight runs).
// Points are [row, col]. Sign: +1 where the edge bulges out of the cell (convex), -1 where it bends into the cell
// (concave), 0 straight (|curvature| <= straight_threshold). Tested against Python in edge_engine.test.mjs.

export const RESAMPLE_PX = 0.5;

export const isClosed = c => c.length > 2 && c[0][0] === c[c.length - 1][0] && c[0][1] === c[c.length - 1][1];

export function arcLength(c) {
  const cum = new Float64Array(c.length);
  for (let i = 1; i < c.length; i++) {
    const dr = c[i][0] - c[i - 1][0], dc = c[i][1] - c[i - 1][1];
    cum[i] = cum[i - 1] + Math.sqrt(dr * dr + dc * dc);
  }
  return cum;
}

// numpy.interp for increasing xp
function interp(x, xp, fp) {
  const n = xp.length;
  if (x > xp[n - 1]) return fp[n - 1];
  if (x < xp[0]) return fp[0];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
  let j = lo;
  while (j + 1 < n && xp[j + 1] <= x) j++;
  if (j === n - 1 || xp[j] === x) return fp[j];
  const slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
  let r = slope * (x - xp[j]) + fp[j];
  if (Number.isNaN(r)) { r = slope * (x - xp[j + 1]) + fp[j + 1]; if (Number.isNaN(r) && fp[j] === fp[j + 1]) r = fp[j]; }
  return r;
}

function atArc(c, cum, s) {
  const rows = c.map(p => p[0]), cols = c.map(p => p[1]);
  return Array.from(s, v => [interp(v, cum, rows), interp(v, cum, cols)]);
}

const dist = (a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1]; return Math.sqrt(dx * dx + dy * dy); };

// The pre-2026-11 sampler (arc-length count restarted at the segment start; see edge_curvature.py).
export function sampleLegacy(contour, n) {
  const cum = arcLength(contour), step = cum[cum.length - 1] / n;
  const pts = [contour[0]], s = [0];
  let cur = 0, i = 0;
  while (pts.length < n && i < contour.length - 1) {
    const d = dist(contour[i], contour[i + 1]);
    if (cur + d >= step) {
      const t = (step - cur) / d, a = contour[i], b = contour[i + 1];
      pts.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
      s.push(cum[i] + t * d);
      cur = 0;
    } else { cur += d; i++; }
  }
  return { points: pts, s };
}

export function sampleEdgePoints(contour, n, spacing = 'equal') {
  if (spacing === 'legacy') return sampleLegacy(contour, n);
  const cum = arcLength(contour), L = cum[cum.length - 1], s = new Float64Array(n);
  if (isClosed(contour)) for (let k = 0; k < n; k++) s[k] = k * (L / n);
  else if (n > 1) { for (let k = 0; k < n; k++) s[k] = k * (L / (n - 1)); s[n - 1] = L; }
  return { points: atArc(contour, cum, s), s: Array.from(s) };
}

// Gaussian filter (sigma in samples, radius int(4 sigma + 0.5)); wrap (closed) or repeat the ends (open).
export function gaussianSmooth(v, sigma, closed) {
  const n = v.length, radius = Math.trunc(4 * sigma + 0.5), w = [];
  let tot = 0;
  for (let k = -radius; k <= radius; k++) { const x = Math.exp(-0.5 * (k / sigma) ** 2); w.push(x); tot += x; }
  for (let k = 0; k < w.length; k++) w[k] /= tot;
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    for (let k = -radius; k <= radius; k++) {
      let j = i + k;
      j = closed ? ((j % n) + n) % n : Math.min(Math.max(j, 0), n - 1);
      acc += v[j] * w[k + radius];
    }
    out[i] = acc;
  }
  return out;
}

export function smoothedCurvature(contour, s, smoothingPx) {
  const closed = isClosed(contour), cum = arcLength(contour), L = cum[cum.length - 1];
  if (!(L > 0)) return s.map(() => 0);
  let n, ds;
  const grid = [];
  if (closed) { n = Math.max(roundHalfEven(L / RESAMPLE_PX), 8); ds = L / n; for (let j = 0; j < n; j++) grid.push(j * ds); }
  else { n = Math.max(roundHalfEven(L / RESAMPLE_PX), 2) + 1; ds = L / (n - 1); for (let j = 0; j < n; j++) grid.push(j * ds); grid[n - 1] = L; }
  const xy = atArc(contour, cum, grid);
  if (closed) smoothingPx = Math.min(smoothingPx, L / (2 * Math.PI));  // tiny loops must not shrink to a point
  const r = gaussianSmooth(xy.map(p => p[0]), smoothingPx / ds, closed);
  const c = gaussianSmooth(xy.map(p => p[1]), smoothingPx / ds, closed);
  const kappa = new Float64Array(n);
  for (let j = 0; j < n; j++) {
    const a = (j + 1) % n, b = (j - 1 + n) % n;
    const r1 = (r[a] - r[b]) / (2 * ds), c1 = (c[a] - c[b]) / (2 * ds);
    const r2 = (r[a] - 2 * r[j] + r[b]) / ds ** 2, c2 = (c[a] - 2 * c[j] + c[b]) / ds ** 2;
    const speed = (r1 * r1 + c1 * c1) ** 1.5;
    kappa[j] = speed > 0 ? (c1 * r2 - r1 * c2) / speed : 0;
  }
  if (closed) {
    const g = [...grid, L], k = [...kappa, kappa[0]];
    return s.map(v => interp(v % L, g, k));
  }
  kappa[0] = kappa[1]; kappa[n - 1] = kappa[n - 2];
  return s.map(v => interp(v, grid, kappa));
}

// Python round(): halves to even
export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

const signOf = (k, thr) => (k > thr ? 1 : k < -thr ? -1 : 0);

function normalize(sign, mag) {
  const mx = mag.length ? Math.max(...mag) : 0;
  return mag.map((m, i) => (mx > 0 ? sign[i] * (m / mx) : sign[i] * 0));
}

export function threePointCurvature(points, closed = true, thr = 0) {
  const n = points.length, sign = new Array(n).fill(0), mag = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    if (!closed && (i === 0 || i === n - 1)) continue;
    const p = points[i], pl = points[(i - 1 + n) % n], pr = points[(i + 1) % n];
    const v0 = pr[0] - pl[0], v1 = pr[1] - pl[1], vl = Math.sqrt(v0 * v0 + v1 * v1);
    const w0 = p[0] - pl[0], w1 = p[1] - pl[1];
    const cross = v0 * w1 - v1 * w0;
    mag[i] = vl > 0 ? Math.abs(cross) / (vl * vl) : 0;
    if (thr > 0) {
      const den = Math.sqrt(w0 * w0 + w1 * w1) * dist(p, pr) * vl;
      sign[i] = signOf(den > 0 ? 2 * cross / den : 0, thr);
    } else sign[i] = cross > 0 ? 1 : -1;
  }
  return { sign, magnitude: mag, normalized: normalize(sign, mag) };
}

export function edgeCurvature(contour, points, s, smoothingPx, thr) {
  if (!(smoothingPx > 0)) return threePointCurvature(points, isClosed(contour), thr);
  const kappa = smoothedCurvature(contour, s, smoothingPx);
  const sign = kappa.map(k => signOf(k, thr)), mag = kappa.map(Math.abs);
  return { sign, magnitude: mag, normalized: normalize(sign, mag) };
}
