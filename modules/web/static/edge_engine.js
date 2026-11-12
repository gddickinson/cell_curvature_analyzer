// Browser port of modules/edge_analysis.py (curvature, inward normals, intensity sampling, edge movement, pooled
// regression) and of modules/web/analysis.py::analyze_stack. Tested against Python in edge_engine.test.mjs.
// Points are [row, col]. Curvature sign: +1 where the edge bulges out of the cell (convex), -1 where it bends
// into the cell or is straight (see the Convention page and edge_analysis.py).
import { detectCellEdge, sampleEquidistant } from './contour_engine.js';
import { linregress } from './stats_engine.js';

const norm2 = (x, y) => Math.sqrt(x * x + y * y);

// Python's round(): halves go to the even integer.
export function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

export function measureCurvature(points) {
  const n = points.length, sign = new Array(n), mag = new Array(n);
  for (let i = 0; i < n; i++) {
    const p = points[i], pl = points[(i - 1 + n) % n], pr = points[(i + 1) % n];
    const v0 = pr[0] - pl[0], v1 = pr[1] - pl[1], vl = norm2(v0, v1);
    const w0 = p[0] - pl[0], w1 = p[1] - pl[1];
    const cross = v0 * w1 - v1 * w0;
    mag[i] = vl > 0 ? Math.abs(cross) / (vl * vl) : 0;
    sign[i] = cross > 0 ? 1 : -1;
  }
  const mx = n ? Math.max(...mag) : 0;
  const normalized = mag.map((m, i) => (mx > 0 ? sign[i] * (m / mx) : sign[i] * 0));
  return { sign, magnitude: mag, normalized };
}

export function inwardNormals(points, mask, h, w) {
  const n = points.length, out = new Array(n);
  const clip = (v, hi) => Math.min(Math.max(v, 0), hi);
  for (let i = 0; i < n; i++) {
    const p = points[i];
    let t0 = points[(i + 1) % n][0] - points[(i - 1 + n) % n][0], t1 = points[(i + 1) % n][1] - points[(i - 1 + n) % n][1];
    let tn = norm2(t0, t1);
    if (tn > 1e-10) { t0 /= tn; t1 /= tn; } else {
      t0 = points[(i + 2) % n][0] - points[(i - 2 + 2 * n) % n][0]; t1 = points[(i + 2) % n][1] - points[(i - 2 + 2 * n) % n][1];
      tn = norm2(t0, t1);
      if (tn > 1e-10) { t0 /= tn; t1 /= tn; } else { t0 = 1; t1 = 0; }
    }
    const n1 = [-t1, t0], n2 = [-n1[0], -n1[1]];
    const r1 = roundHalfEven(clip(p[0] + 3 * n1[0], h - 1)), c1 = roundHalfEven(clip(p[1] + 3 * n1[1], w - 1));
    const r2 = roundHalfEven(clip(p[0] + 3 * n2[0], h - 1)), c2 = roundHalfEven(clip(p[1] + 3 * n2[1], w - 1));
    if (Number.isNaN(r1) || Number.isNaN(c1) || Number.isNaN(r2) || Number.isNaN(c2)) { out[i] = n1; continue; }
    out[i] = mask[r1 * w + c1] === 1 ? n1 : n2;  // val2 == 1 or neither inside: normal2 (as Python)
  }
  return out;
}

// skimage.geometry point_in_polygon (O'Rourke): 0 outside, 1 inside, 2 vertex, 3 edge.
function pointInPolygon(xp, yp, x, y) {
  const nv = xp.length, eps = 1e-12;
  let x1 = xp[nv - 1] - x, y1 = yp[nv - 1] - y, l = 0, r = 0;
  for (let i = 0; i < nv; i++) {
    const x0 = xp[i] - x, y0 = yp[i] - y;
    if (-eps < x0 && x0 < eps && -eps < y0 && y0 < eps) return 2;
    if ((y0 > 0) !== (y1 > 0) && (x0 * y1 - x1 * y0) / (y1 - y0) > 0) r++;
    if ((y0 < 0) !== (y1 < 0) && (x0 * y1 - x1 * y0) / (y1 - y0) < 0) l++;
    x1 = x0; y1 = y0;
  }
  if ((r & 1) !== (l & 1)) return 3;
  return r & 1 ? 1 : 0;
}

// skimage.draw.polygon(rows, cols, shape): pixels [r, c] inside or on the polygon, clipped to the image.
export function polygonPixels(rows, cols, h, w) {
  const minr = Math.trunc(Math.max(0, Math.min(...rows))), minc = Math.trunc(Math.max(0, Math.min(...cols)));
  const maxr = Math.min(h - 1, Math.ceil(Math.max(...rows))), maxc = Math.min(w - 1, Math.ceil(Math.max(...cols)));
  const rr = [], cc = [];
  for (let r = minr; r <= maxr; r++) for (let c = minc; c <= maxc; c++) {
    if (pointInPolygon(cols, rows, c, r)) { rr.push(r); cc.push(c); }
  }
  return [rr, cc];
}

function rectangle(p, nrm, perp, depth, width) {
  const e = [p[0] + depth * nrm[0], p[1] + depth * nrm[1]], hw = width / 2;
  return [[p[0] + hw * perp[0], p[1] + hw * perp[1]], [e[0] + hw * perp[0], e[1] + hw * perp[1]],
    [e[0] - hw * perp[0], e[1] - hw * perp[1]], [p[0] - hw * perp[0], p[1] - hw * perp[1]]];
}

function coverage(verts, mask, h, w) {
  const [rr, cc] = polygonPixels(verts.map(v => v[0]), verts.map(v => v[1]), h, w);
  if (!rr.length) return null;
  let inside = 0;
  for (let k = 0; k < rr.length; k++) inside += mask[rr[k] * w + cc[k]];
  return { rr, cc, cov: inside / rr.length };
}

const meanAt = (image, w, { rr, cc }) => { let s = 0; for (let k = 0; k < rr.length; k++) s += image[rr[k] * w + cc[k]]; return s / rr.length; };

export function measureIntensity(image, mask, h, w, points, normals, p) {
  const n = points.length, intensities = new Array(n).fill(null), valid = new Array(n).fill(false);
  const recovered = new Array(n).fill(false), regions = [];
  for (let i = 0; i < n; i++) {
    if (p.exclude_endpoints && (i === 0 || i === n - 1)) { regions.push([[0, 0], [0, 0], [0, 0], [0, 0]]); continue; }
    const pt = points[i], nrm = normals[i], perp = [-nrm[1], nrm[0]];
    let verts = rectangle(pt, nrm, perp, p.depth, p.width);
    const c = coverage(verts, mask, h, w);
    if (c) {
      if (c.cov >= p.min_cell_coverage) {
        intensities[i] = meanAt(image, w, c); valid[i] = true;
      } else if (p.try_rotation) {
        const rv = rectangle(pt, [-nrm[0], -nrm[1]], perp, p.depth, p.width);
        const rc = coverage(rv, mask, h, w);
        if (rc && rc.cov >= p.min_cell_coverage) {
          intensities[i] = meanAt(image, w, rc); valid[i] = recovered[i] = true; verts = rv;
        }
      }
    }
    regions.push(verts);
  }
  return { intensities, regions, valid, recovered };
}

export function edgeMovement(cur, prev, threshold) {
  let ext = 0, ret = 0;
  for (let k = 0; k < cur.length; k++) { const d = cur[k] - prev[k]; if (d === 1) ext++; else if (d === -1) ret++; }
  const score = ext + ret > 0 ? (ext - ret) / (ext + ret) : 0;
  const type = score > threshold ? 'extending' : score < -threshold ? 'retracting' : 'stable';
  return { score, type, extended_px: ext, retracted_px: ret };
}

// edge_analysis.summary_correlation_stats over [{x: curvature[], y: intensity[], ok: valid[]}]
export function summaryStats(frames, which) {
  const x = [], y = [];
  for (const f of frames) if (!f.error) f.valid.forEach((ok, i) => { if (ok) { x.push(f[which][i]); y.push(f.intensities[i]); } });
  const out = { r_squared: null, p_value: null, sample_size: x.length };
  if (x.length < 2 || x.every(v => v === x[0])) return out;
  const r = linregress(x, y);
  return { ...out, r_squared: r.r * r.r, p_value: r.p, slope: r.slope, intercept: r.intercept, standard_error: r.stderr };
}

// One frame: image (Float64Array) and mask (Uint8Array of 0/1), both h*w row-major.
export function analyzeFrame(image, mask, h, w, p) {
  const contour = detectCellEdge(mask, h, w);
  if (!contour.length) return { error: 'No contour found' };
  const points = sampleEquidistant(contour, p.n_points);
  const curv = measureCurvature(points);
  const normals = inwardNormals(points, mask, h, w);
  const s = measureIntensity(image, mask, h, w, points, normals, p);
  return { contour, points, ...curv, normals, ...s };
}

// Stack: images/masks are arrays of per-frame typed arrays. Mirrors modules/web/analysis.py::analyze_stack.
export function analyzeStack(images, masks, h, w, p) {
  const bin = masks.map(m => Uint8Array.from(m, v => (v > 0 ? 1 : 0)));
  const frames = [], movement = [];
  bin.forEach((mask, t) => {
    const f = { frame: t, ...analyzeFrame(images[t], mask, h, w, p) };
    frames.push(f);
    if (!f.error && t > 0) movement.push({ frame: t, ...edgeMovement(mask, bin[t - 1], p.movement_threshold) });
  });
  return {
    shape: [h, w], n_frames: images.length, params: p, frames, movement,
    stats: { sign: summaryStats(frames, 'sign'), normalized: summaryStats(frames, 'normalized') },
  };
}
