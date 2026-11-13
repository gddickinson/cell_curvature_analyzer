// Curvature sign page: the convention and options of modules/edge_curvature.py, shown on test shapes computed live
// with the JS port (the same shapes and assertions are in tests/test_edge_curvature.py and edge_engine.test.mjs).
import { route, app, api } from './core.js';
import { detectCellEdge } from './contour_engine.js';
import { edgeCurvature, sampleEdgePoints } from './curvature_engine.js';
import { curvColor } from './analyze_view.js';

const LEGACY = { point_spacing: 'legacy', smoothing_px: 0, straight_threshold: 0 };
let OPTS = null;  // defaults from /api/info

function signs(edge, n, o) {
  const { points, s } = sampleEdgePoints(edge, n, o.point_spacing);
  return { points, sign: edgeCurvature(edge, points, s, o.smoothing_px, o.straight_threshold).sign };
}

function maskOf(h, w, inside) {
  const m = new Uint8Array(h * w);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) m[r * w + c] = inside(r, c) ? 1 : 0;
  return m;
}

const SHAPES = [
  { name: 'Circle', n: 100, h: 120, w: 120, f: (r, c) => (r - 60) ** 2 + (c - 60) ** 2 < 40 ** 2,
    cap: 'Convex everywhere, so every point should be +1.' },
  { name: 'Circle with a bite', n: 100, h: 120, w: 120,
    f: (r, c) => (r - 60) ** 2 + (c - 60) ** 2 < 45 ** 2 && !((r - 60) ** 2 + (c - 105) ** 2 < 18 ** 2),
    cap: 'The bite bends into the cell (−1); its two corners and the rest of the circle are convex (+1).' },
  { name: 'Wound edge (open contour)', n: 50, h: 100, w: 160,
    f: (r, c) => r > 50 - 14 * Math.exp(-(((c - 80) / 12) ** 2)) + 10 * Math.exp(-(((c - 30) / 10) ** 2)),
    cap: 'Cell below, wound above: the protrusion is +1, the dip −1, flat runs straight (grey, 0) with the dead-band.' },
];

function drawShape(canvas, s, o) {
  const mask = maskOf(s.h, s.w, s.f), { points: pts, sign } = signs(detectCellEdge(mask, s.h, s.w), s.n, o), k = 2;
  canvas.width = s.w * k; canvas.height = s.h * k;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b1020'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#1f3b40';
  for (let r = 0; r < s.h; r++) for (let c = 0; c < s.w; c++) if (mask[r * s.w + c]) ctx.fillRect(c * k, r * k, k, k);
  pts.forEach((p, i) => { ctx.beginPath(); ctx.arc((p[1] + 0.5) * k, (p[0] + 0.5) * k, 4, 0, 2 * Math.PI); ctx.fillStyle = curvColor(sign[i]); ctx.fill(); });
  return { plus: sign.filter(v => v === 1).length, zero: sign.filter(v => v === 0).length,
    minus: sign.filter(v => v === -1).length, n: pts.length };
}

function spacingTable() {
  const h = 120, w = 120, mask = maskOf(h, w, (r, c) => (r - 60) ** 2 + (c - 60) ** 2 < 40 ** 2), edge = detectCellEdge(mask, h, w);
  let L = 0;
  for (let i = 0; i < edge.length - 1; i++) L += Math.hypot(edge[i + 1][0] - edge[i][0], edge[i + 1][1] - edge[i][1]);
  const plus = (n, o) => signs(edge, n, o).sign.filter(v => v === 1).length;
  return [12, 24, 36, 48, 60, 80, 100].map(n => `<tr><td class="num">${n}</td><td class="num">${(L / n).toFixed(1)}</td>
    <td class="num">${plus(n, LEGACY)} / ${n}</td><td class="num">${plus(n, OPTS)} / ${n}</td></tr>`).join('');
}

const DIAGRAM = `<svg viewBox="0 0 420 210" class="plot" role="img" aria-label="Three neighbouring edge points, the chord between the outer two and the cross product sign">
  <path d="M10 200 Q 210 -40 410 200 Z" fill="#1f3b40"/><text x="210" y="170" text-anchor="middle" class="axis-label">cell</text>
  <text x="210" y="22" text-anchor="middle">outside (wound)</text>
  <line x1="70" y1="120" x2="350" y2="120" stroke="#9aa3b5" stroke-dasharray="4 4"/>
  <line x1="210" y1="80" x2="210" y2="120" stroke="#ffd23f"/><text x="216" y="104" fill="#ffd23f">h</text>
  <circle cx="70" cy="120" r="6" fill="#5cf2c4"/><text x="58" y="142">p_left</text>
  <circle cx="350" cy="120" r="6" fill="#5cf2c4"/><text x="330" y="142">p_right</text>
  <circle cx="210" cy="80" r="7" fill="#ff8a3d"/><text x="222" y="72">p (+1, convex)</text>
  <text x="210" y="138" text-anchor="middle">chord v = p_right − p_left</text></svg>`;

const opt = o => `smoothing ${o.smoothing_px > 0 ? `σ = ${o.smoothing_px} px` : 'none'}, straight below ${o.straight_threshold}/px, ${o.point_spacing} spacing`;

route('convention', async () => {
  OPTS = (await api('/info')).defaults;
  const o = OPTS;
  app().innerHTML = `<p class="kicker">Method</p><h1>Curvature sign convention</h1>
    <p class="lead">One rule everywhere (desktop app, Python core, this site): <b style="color:var(--convex)">+1 where the edge bulges out of
      the cell</b> (convex, e.g. a protrusion into a wound), <b style="color:var(--concave)">−1 where it bends into the cell</b> (concave),
      and <b>0 where it is straight</b> (|curvature| below the dead-band).</p>
    <div class="two"><div>${DIAGRAM}</div><div>
      <p><b>Default (${opt(o)}).</b> <code>modules/edge_curvature.py</code> resamples the edge every 0.5 px, smooths it with a Gaussian
        of σ = ${o.smoothing_px} px along its length, and reads the signed curvature at each sample point, in (row, column) coordinates:</p>
      <p class="eq">κ = (c′·r″ − r′·c″) / (r′² + c′²)^1.5 &nbsp; (1/px; 1/radius)</p>
      <p class="eq">sign = +1 if κ &gt; t, −1 if κ &lt; −t, else 0 &nbsp; (t = straight threshold)</p>
      <p class="eq">magnitude = |κ|; &nbsp; normalized = sign · magnitude / max(magnitude)</p>
      <p><b>None (legacy)</b>, smoothing 0: the three-point rule from each point and its neighbours p_left, p_right (diagram):
        cross = v_row·w_col − v_col·w_row with v = p_right − p_left, w = p − p_left; magnitude = |cross| / |v|² (dimensionless);
        with threshold 0 the sign is +1 if cross &gt; 0 else −1, so straight runs count as −1, as before 2026-11.</p></div></div>
    <h2>Why the sign is consistent</h2>
    <p>The edge comes from <code>skimage.measure.find_contours(mask, 0.5)</code>, which always walks with the cell on its right-hand side
      (clockwise round a cell as seen on screen, y pointing down). With that walking direction an edge curving away from the cell
      has positive curvature. The convention therefore holds for every mask, closed or open, and is pinned by tests: a circular mask
      gives +1 at every point (<code>tests/test_edge_curvature.py</code>, <code>edge_engine.test.mjs</code>), and walking the edge
      backwards flips every sign, so points must not be re-ordered before measuring curvature.</p>
    <div class="mini" id="shapes"></div>
    <h2>Options and what the numbers mean</h2>
    <ul><li><b>Smoothing</b> (σ, px). Without it, closely spaced points on a pixel mask see the pixel staircase and single-point signs are
      mostly noise. The default σ = ${o.smoothing_px} px was chosen with <code>scripts/curvature_benchmark.py</code> (seeded masks of known
      curvature: circles R = 20–80 px, an ellipse, notches r = 10 and 20 px, a 5 px-wide finger, a rounded rectangle, a wound edge, with
      0–0.6 px segmentation noise): it gives the best worst-shape sign agreement averaged over the noise levels (at 0.3 px noise and
      no dead-band, ≥ 95 % on every shape, vs 45–63 % on the curved shapes without smoothing); σ ≥ 8 px starts to blur the 5 px finger
      and the r = 20 px notch.</li>
      <li><b>Straight threshold</b> (1/px). |κ| below it is straight (0). Smoothed straight pixel edges still show |κ| up to ~0.02/px from
        the staircase. The default ${o.straight_threshold}/px (radius ${o.straight_threshold > 0 ? Math.round(1 / o.straight_threshold) : '∞'} px)
        gave the best balance in the benchmark between straight points called straight and curved points (|κ| ≥ 0.01/px) signed right,
        averaged over noise levels (0.86, vs 0.84 at 0.008 and 0.83 at 0.015). The price: edges flatter than that radius count as
        straight, e.g. most points of an 80 px-radius circle are still +1 but some read 0. For large cells or fine pixel sizes, lower it.</li>
      <li><b>Magnitude</b>: with smoothing, |κ| in 1/px (divide by µm/px for 1/µm), comparable across analyses. Legacy: |cross| / |v|², about
        κ·Δs/4 (Δs = spacing), so it depends on the number of points.</li>
      <li><b>Open edges</b> (a cell running off the image) are open: the first and last points are not neighbours, have no three-point
        curvature, and are not sampled for intensity. (Before 2026-11 they were paired across the image.)</li>
      <li><b>Point spacing</b>: "equal" puts points at equal arc-length steps. "Legacy" is the old sampler, whose steps were up to one
        contour segment short, leaving one large gap at the end (2.0 px steps and a 42 px gap for 100 points on a 40 px circle).</li></ul>
    <div class="scroll"><table style="max-width:560px"><thead><tr><th class="num">points</th><th class="num">spacing (px)</th>
      <th class="num">+1, legacy options</th><th class="num">+1, defaults</th></tr></thead><tbody>${spacingTable()}</tbody></table></div>
    <p class="small mute">Circular mask, radius 40 px, computed live in your browser with the JS port. A true circle is convex at every point.</p>`;
  const host = document.getElementById('shapes');
  for (const s of SHAPES) {
    for (const [lab, oo] of [['defaults', o], ['legacy', LEGACY]]) {
      const fig = document.createElement('figure');
      fig.innerHTML = `<canvas role="img" aria-label="${s.name}, ${lab} options, sample points coloured by curvature sign"></canvas><figcaption><b>${s.name}</b> (${lab}): <span></span></figcaption>`;
      host.appendChild(fig);
      const c = drawShape(fig.querySelector('canvas'), s, oo);
      fig.querySelector('span').textContent = `${lab === 'defaults' ? s.cap + ' ' : ''}${c.plus} +1, ${c.zero} straight, ${c.minus} −1 of ${c.n}.`;
    }
  }
});
