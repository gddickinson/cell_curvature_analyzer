// Home page: hero, facts from the repo (defaults from modules/analysis_defaults.py), and links to the pages.
import { route, app, api, esc, isStatic } from './core.js';

route('home', async () => {
  const info = await api('/info');
  const d = info.defaults;
  app().innerHTML = `<section class="hero"><p class="kicker">Membrane curvature · fluorescence · PIEZO1</p>
    <h1 class="glow">Cell Curvature Analyzer</h1>
    <p class="lead">${esc(info.description)} Written to study how PIEZO1 protein locations relate to membrane
      curvature. Bring your own image and mask: the analysis runs ${isStatic() ? 'in your browser, and nothing is uploaded' : 'in the local Python server'}.</p>
    <a class="btn" href="#/analyze/demo">Try the synthetic demo</a><a class="btn alt" href="#/analyze">Analyse your files</a></section>
    <div class="tiles">
      <div class="tile"><b>${info.curvature_measures.length}</b><span>curvature measures per point (sign, magnitude, normalized)</span></div>
      <div class="tile"><b>${d.n_points}</b><span>edge points by default</span></div>
      <div class="tile"><b>${d.depth} × ${d.width}</b><span>px sampling rectangle by default</span></div>
      <div class="tile"><b>${Math.round(d.min_cell_coverage * 100)}%</b><span>of a rectangle must be inside the cell</span></div></div>
    <div class="callout small"><b>Sign convention:</b> curvature is <b style="color:var(--convex)">+1 where the edge bulges out of the cell</b>
      (convex) and <b style="color:var(--concave)">−1 where it bends into it</b> (concave) or is straight. <a href="#/convention">How and why</a>.</div>
    <div class="grid">
      <a class="card" href="#/analyze"><b>Analyze</b><span>Image + mask (TIFF stacks or PNG) or the synthetic demo: edge overlay coloured by
        curvature, sampling rectangles, profiles along the edge, intensity vs curvature regression, edge movement, CSV export.</span></a>
      <a class="card" href="#/convention"><b>Curvature sign</b><span>The convention, why it is consistent, what magnitude means, and live
        test shapes: a circle, a notch and a wound edge.</span></a>
      <a class="card" href="#/about"><b>About</b><span>The desktop app (${info.desktop_tabs.length} tabs), how to run it and this web app, and how
        the browser port is checked against Python.</span></a>
    </div>`;
});
