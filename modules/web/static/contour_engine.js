// Cell edge from a binary mask: a port of skimage.measure.find_contours (marching squares + segment assembly,
// default fully_connected='low', positive_orientation='low') and of edge_analysis.detect_cell_edge /
// sample_equidistant_points. Points are [row, col] pixel centres. Tested against Python in edge_engine.test.mjs.

const frac = (a, b, level) => (b === a ? 0 : (level - a) / (b - a));

// Marching-squares segments, in skimage's traversal order. arr: Float64Array/Uint8Array of length h*w (row-major).
export function contourSegments(arr, h, w, level = 0.5) {
  const segs = [];
  for (let r0 = 0; r0 < h - 1; r0++) {
    for (let c0 = 0; c0 < w - 1; c0++) {
      const r1 = r0 + 1, c1 = c0 + 1;
      const ul = arr[r0 * w + c0], ur = arr[r0 * w + c1], ll = arr[r1 * w + c0], lr = arr[r1 * w + c1];
      if (Number.isNaN(ul) || Number.isNaN(ur) || Number.isNaN(ll) || Number.isNaN(lr)) continue;
      const sq = (ul > level ? 1 : 0) + (ur > level ? 2 : 0) + (ll > level ? 4 : 0) + (lr > level ? 8 : 0);
      if (sq === 0 || sq === 15) continue;
      const top = [r0, c0 + frac(ul, ur, level)], bottom = [r1, c0 + frac(ll, lr, level)];
      const left = [r0 + frac(ul, ll, level), c0], right = [r0 + frac(ur, lr, level), c1];
      switch (sq) {
        case 1: segs.push([top, left]); break;
        case 2: segs.push([right, top]); break;
        case 3: segs.push([right, left]); break;
        case 4: segs.push([left, bottom]); break;
        case 5: segs.push([top, bottom]); break;
        case 6: segs.push([right, top], [left, bottom]); break;   // vertex_connect_high = False
        case 7: segs.push([right, bottom]); break;
        case 8: segs.push([bottom, right]); break;
        case 9: segs.push([top, left], [bottom, right]); break;
        case 10: segs.push([bottom, top]); break;
        case 11: segs.push([bottom, left]); break;
        case 12: segs.push([left, right]); break;
        case 13: segs.push([top, right]); break;
        case 14: segs.push([left, top]); break;
      }
    }
  }
  return segs;
}

const key = p => p[0] + ',' + p[1];

// skimage _assemble_contours: joins segments into polylines (deques), ordered by creation index.
export function assembleContours(segs) {
  let idx = 0;
  const contours = new Map(), starts = new Map(), ends = new Map();
  for (const [from, to] of segs) {
    if (from[0] === to[0] && from[1] === to[1]) continue;
    const kt = key(to), kf = key(from);
    const tailE = starts.get(kt); starts.delete(kt);
    const headE = ends.get(kf); ends.delete(kf);
    if (tailE && headE) {
      const [tail, tailNum] = tailE, [head, headNum] = headE;
      if (tail === head) {
        head.push(to);
      } else if (tailNum > headNum) {
        for (const p of tail) head.push(p);
        contours.delete(tailNum);
        starts.set(key(head[0]), [head, headNum]);
        ends.set(key(head[head.length - 1]), [head, headNum]);
      } else {
        starts.delete(key(head[0]));
        tail.unshift(...head);
        contours.delete(headNum);
        starts.set(key(tail[0]), [tail, tailNum]);
        ends.set(key(tail[tail.length - 1]), [tail, tailNum]);
      }
    } else if (!tailE && !headE) {
      const c = [from, to];
      contours.set(idx, c);
      starts.set(kf, [c, idx]);
      ends.set(kt, [c, idx]);
      idx++;
    } else if (!headE) {
      const [tail, tailNum] = tailE;
      tail.unshift(from);
      starts.set(kf, [tail, tailNum]);
    } else {
      const [head, headNum] = headE;
      head.push(to);
      ends.set(kt, [head, headNum]);
    }
  }
  return [...contours.entries()].sort((a, b) => a[0] - b[0]).map(e => e[1]);
}

export function findContours(arr, h, w, level = 0.5) {
  return assembleContours(contourSegments(arr, h, w, level));
}

// Longest contour of the mask (first one on ties, like Python's max); [] if none.
export function detectCellEdge(mask, h, w) {
  let best = [];
  for (const c of findContours(mask, h, w, 0.5)) if (c.length > best.length) best = c;
  return best;
}

// sqrt(dx*dx + dy*dy) as numpy.linalg.norm computes it (Math.hypot can differ in the last bit)
const dist = (a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1]; return Math.sqrt(dx * dx + dy * dy); };

// Points every (contour length / n) along the contour, starting at contour[0] (edge_analysis.sample_equidistant_points).
export function sampleEquidistant(contour, n) {
  let length = 0;
  for (let i = 0; i < contour.length - 1; i++) length += dist(contour[i], contour[i + 1]);
  const step = length / n;
  const out = [contour[0]];
  let cur = 0, i = 0;
  while (out.length < n && i < contour.length - 1) {
    const d = dist(contour[i], contour[i + 1]);
    if (cur + d >= step) {
      const t = (step - cur) / d, a = contour[i], b = contour[i + 1];
      out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
      cur = 0;
    } else {
      cur += d;
      i++;
    }
  }
  return out;
}
