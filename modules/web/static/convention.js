// Curvature sign page: the convention used by edge_analysis.measure_curvature, shown on test shapes computed live
// with the JS port (the same shapes and assertions are in tests/test_edge_analysis.py and edge_engine.test.mjs).
import { route, app } from './core.js';
import { detectCellEdge, sampleEquidistant } from './contour_engine.js';
import { measureCurvature } from './edge_engine.js';
import { curvColor } from './analyze_view.js';

function maskOf(h, w, inside) {
  const m = new Uint8Array(h * w);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) m[r * w + c] = inside(r, c) ? 1 : 0;
  return m;
}

const SHAPES = [
  { name: 'Circle', n: 24, h: 120, w: 120, f: (r, c) => (r - 60) ** 2 + (c - 60) ** 2 < 40 ** 2,
    cap: 'Convex everywhere: all 24 points +1.' },
  { name: 'Circle with a bite', n: 40, h: 120, w: 120,
    f: (r, c) => (r - 60) ** 2 + (c - 60) ** 2 < 45 ** 2 && !((r - 60) ** 2 + (c - 105) ** 2 < 18 ** 2),
    cap: 'The bite bends into the cell: −1 there; its two corners are convex.' },
  { name: 'Wound edge (open contour)', n: 50, h: 100, w: 160,
    f: (r, c) => r > 50 - 14 * Math.exp(-(((c - 80) / 12) ** 2)) + 10 * Math.exp(-(((c - 30) / 10) ** 2)),
    cap: 'Cell below, wound above: the protrusion tip is +1, the dip −1. Flat runs are −1 (cross product 0).' },
];

function drawShape(canvas, s) {
  const mask = maskOf(s.h, s.w, s.f), pts = sampleEquidistant(detectCellEdge(mask, s.h, s.w), s.n);
  const { sign } = measureCurvature(pts), k = 2;
  canvas.width = s.w * k; canvas.height = s.h * k;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#0b1020'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#1f3b40';
  for (let r = 0; r < s.h; r++) for (let c = 0; c < s.w; c++) if (mask[r * s.w + c]) ctx.fillRect(c * k, r * k, k, k);
  pts.forEach((p, i) => { ctx.beginPath(); ctx.arc((p[1] + 0.5) * k, (p[0] + 0.5) * k, 4, 0, 2 * Math.PI); ctx.fillStyle = curvColor(sign[i]); ctx.fill(); });
  return { plus: sign.filter(v => v === 1).length, n: pts.length };
}

function spacingTable() {
  const h = 120, w = 120, mask = maskOf(h, w, (r, c) => (r - 60) ** 2 + (c - 60) ** 2 < 40 ** 2), edge = detectCellEdge(mask, h, w);
  let L = 0;
  for (let i = 0; i < edge.length - 1; i++) L += Math.hypot(edge[i + 1][0] - edge[i][0], edge[i + 1][1] - edge[i][1]);
  return [12, 24, 36, 48, 60, 80, 100].map(n => {
    const { sign } = measureCurvature(sampleEquidistant(edge, n));
    return `<tr><td class="num">${n}</td><td class="num">${(L / n).toFixed(1)}</td><td class="num">${sign.filter(v => v === 1).length} / ${n}</td></tr>`;
  }).join('');
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

route('convention', () => {
  app().innerHTML = `<p class="kicker">Method</p><h1>Curvature sign convention</h1>
    <p class="lead">One rule everywhere (desktop app, Python core, this site): <b style="color:var(--convex)">+1 where the edge bulges out of
      the cell</b> (convex, e.g. a protrusion into a wound) and <b style="color:var(--concave)">−1 where it bends into the cell</b> (concave)
      or is straight.</p>
    <div class="two"><div>${DIAGRAM}</div><div>
      <p>At each sample point p, with neighbours p_left and p_right along the edge, <code>modules/edge_analysis.py</code> computes the
        cross product of the chord v = p_right − p_left and w = p − p_left, in (row, column) coordinates:</p>
      <p class="eq">cross = v_row·w_col − v_col·w_row</p>
      <p class="eq">sign = +1 if cross &gt; 0 else −1</p>
      <p class="eq">magnitude = |cross| / |v|² = h / |v|</p>
      <p class="eq">normalized = sign · magnitude / max(magnitude)</p></div></div>
    <h2>Why the sign is consistent</h2>
    <p>The edge comes from <code>skimage.measure.find_contours(mask, 0.5)</code>, which always walks with the cell on its right-hand side
      (clockwise round a cell as seen on screen, y pointing down). With that walking direction a point that lies off the chord away
      from the cell gives a positive cross product. The convention therefore holds for every mask, closed or open, and is pinned by
      tests: a circular mask gives +1 at every point (<code>tests/test_edge_analysis.py</code>, <code>edge_engine.test.mjs</code>), and
      walking the same points backwards flips every sign, so points must not be re-ordered before measuring curvature.</p>
    <div class="mini" id="shapes"></div>
    <h2>What the numbers mean</h2>
    <ul><li><b>Magnitude</b> is the height h of the point above the chord divided by the chord length. It is dimensionless and depends
      on the point spacing: for a smooth edge it is about κ·Δs/4 (κ = 1/radius, Δs = spacing); on a circle sampled with n points it
      is tan(π/n)/2 whatever the radius. Compare magnitudes only between analyses with similar spacing.</li>
      <li><b>Normalized</b> divides by the largest magnitude in that frame, so values are relative within a frame (−1..1).</li>
      <li><b>Straight runs</b> (cross product exactly 0, common on pixel masks) get sign −1 and magnitude 0.</li>
      <li><b>Open edges</b> (a cell touching the image border): neighbours wrap round, so the first and last points are paired with each
        other across the image and their curvature is meaningless. "Exclude end points" only removes them from intensity sampling.</li>
      <li>There is no smoothing. On a pixel mask, closely spaced points see the pixel staircase, and single-point signs become noisy:</li></ul>
    <div class="scroll"><table style="max-width:460px"><thead><tr><th class="num">points</th><th class="num">spacing (px)</th>
      <th class="num">points with sign +1</th></tr></thead><tbody>${spacingTable()}</tbody></table></div>
    <p class="small mute">Circular mask, radius 40 px, computed live in your browser with the JS port. A true circle is convex at every point.</p>`;
  const host = document.getElementById('shapes');
  for (const s of SHAPES) {
    const fig = document.createElement('figure');
    fig.innerHTML = `<canvas role="img" aria-label="${s.name} with sample points coloured by curvature sign"></canvas><figcaption><b>${s.name}</b>: <span></span></figcaption>`;
    host.appendChild(fig);
    const { plus, n } = drawShape(fig.querySelector('canvas'), s);
    fig.querySelector('span').textContent = `${s.cap} (${plus} of ${n} points +1)`;
  }
});
