// node modules/web/static/edge_engine.test.mjs
// Asserts the JS ports (contour_engine.js, edge_engine.js, stats_engine.js) reproduce the repo's Python analysis
// (modules/edge_analysis.py via modules/web/analysis.py::analyze_stack, skimage find_contours / draw.polygon,
// scipy linregress) on tests/fixtures/edge_fixture.json (regenerate: python scripts/make_edge_fixture.py),
// and pins the curvature sign convention on a circle.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { findContours, detectCellEdge, sampleEquidistant } from './contour_engine.js';
import { analyzeStack, measureCurvature, polygonPixels, roundHalfEven } from './edge_engine.js';
import { linregress } from './stats_engine.js';
import { readTiff } from './tiff_reader.js';

const TOL = 1e-9;
const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/edge_fixture.json', import.meta.url)));
const buf = b64 => Buffer.from(b64, 'base64');
const typed = (b64, kind) => {
  const b = buf(b64), ab = b.buffer.slice(b.byteOffset, b.byteOffset + b.length);
  return kind === 'u1' ? new Uint8Array(ab) : kind === 'u2' ? new Uint16Array(ab) : new Float64Array(ab);
};
let worst = 0, nvals = 0;
function close(a, b, what) {
  if (a === null || b === null) { assert.equal(a, b, what); return; }
  if (Array.isArray(a)) { assert.equal(a.length, b.length, what + ' length'); a.forEach((v, i) => close(v, b[i], what)); return; }
  if (typeof a === 'boolean' || typeof a === 'string') { assert.equal(a, b, what); return; }
  const d = Math.abs(a - b) / Math.max(1, Math.abs(b));
  worst = Math.max(worst, d); nvals++;
  assert.ok(d <= TOL, `${what}: ${a} vs ${b}`);
}

// 1. find_contours on random binary arrays: same contours, same order, same points.
for (const c of fx.contours) {
  const [h, w] = c.shape;
  close(findContours(typed(c.mask_u8, 'u1'), h, w, 0.5), c.contours, 'find_contours');
}
// 2. draw.polygon: identical pixel lists (including lattice-aligned rectangles, edges and vertices on pixels).
for (const p of fx.polygons) {
  const [rr, cc] = polygonPixels(p.r, p.c, 20, 20);
  assert.deepEqual([rr, cc], [p.rr, p.cc], `polygon ${JSON.stringify(p.r)} ${JSON.stringify(p.c)}`);
}
// 3. linregress.
for (const r of fx.linregress) {
  const g = linregress(r.x, r.y);
  for (const k of ['slope', 'intercept', 'r', 'p', 'stderr']) close(g[k], r[k], `linregress n=${r.x.length} ${k}`);
}
// 4. Whole-stack analysis.
for (const c of fx.cases) {
  const [T, h, w] = c.shape, n = h * w;
  const im = typed(c.image, c.image_dtype), mk = typed(c.mask_u8, 'u1');
  const images = [], masks = [];
  for (let t = 0; t < T; t++) { images.push(Float64Array.from(im.subarray(t * n, (t + 1) * n))); masks.push(mk.subarray(t * n, (t + 1) * n)); }
  const exp = c.expected;
  const got = analyzeStack(images, masks, h, w, exp.params);
  assert.equal(got.frames.length, exp.frames.length, c.name);
  got.frames.forEach((f, i) => {
    const e = exp.frames[i];
    if (e.error) { assert.equal(f.error, e.error, `${c.name} frame ${i}`); return; }
    for (const k of ['contour', 'points', 'sign', 'magnitude', 'normalized', 'normals', 'intensities', 'regions', 'valid', 'recovered']) {
      close(f[k], e[k], `${c.name} frame ${i} ${k}`);
    }
  });
  assert.equal(got.movement.length, exp.movement.length, `${c.name} movement`);
  got.movement.forEach((m, i) => { for (const k of ['frame', 'score', 'type', 'extended_px', 'retracted_px']) close(m[k], exp.movement[i][k], `${c.name} movement ${k}`); });
  for (const which of ['sign', 'normalized']) {
    const g = got.stats[which], e = exp.stats[which];
    for (const k of Object.keys(e)) close(g[k], e[k], `${c.name} stats ${which} ${k}`);
  }
  console.log(`${c.name}: ${T} frame(s) ok`);
}

// 5. TIFF reader on files written by tifffile.
for (const t of fx.tiffs) {
  const b = readFileSync(new URL('../../../tests/fixtures/tiff/' + t.file, import.meta.url));
  const got = await readTiff(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
  const exp = Array.isArray(t.data[0][0]) ? t.data : [t.data];
  assert.equal(got.frames.length, exp.length, t.file + ' frames');
  assert.equal(got.height, exp[0].length, t.file); assert.equal(got.width, exp[0][0].length, t.file);
  exp.forEach((fr, k) => assert.deepEqual(Array.from(got.frames[k]), fr.flat(), `${t.file} frame ${k}`));
  if (t.um_per_px === null) assert.equal(got.umPerPx, null, t.file);
  else assert.ok(Math.abs(got.umPerPx - t.um_per_px) < 1e-6, `${t.file} um/px ${got.umPerPx}`);
}
console.log(`tiff_reader: ${fx.tiffs.length} files ok`);

// 6. Sign convention on a circle: +1 everywhere (convex), and a reversed walk flips it.
const H = 120, W = 120, disc = new Uint8Array(H * W);
for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) disc[r * W + c] = (r - 60) ** 2 + (c - 60) ** 2 < 40 ** 2 ? 1 : 0;
const pts = sampleEquidistant(detectCellEdge(disc, H, W), 24);
const cv = measureCurvature(pts);
assert.ok(cv.sign.every(s => s === 1), 'circle: every point convex (+1)');
const expected = Math.tan(Math.PI / 24) / 2, med = [...cv.magnitude].sort((a, b) => a - b)[12];
assert.ok(Math.abs(med - expected) / expected < 0.1, 'circle magnitude ~ tan(pi/n)/2');
assert.ok(measureCurvature([...pts].reverse()).sign.every(s => s === -1), 'reversed traversal flips the sign');
assert.deepEqual([0.5, 1.5, 2.5, -0.5, 2.4, 2.6].map(roundHalfEven), [0, 2, 2, 0, 2, 3]);

console.log(`edge_engine: ${fx.cases.length} stack cases, ${fx.contours.length} contour, ${fx.polygons.length} polygon, ` +
  `${fx.linregress.length} regression cases; ${nvals} values, worst relative |JS - Python| = ${worst.toExponential(2)} (tol ${TOL}) ok`);
