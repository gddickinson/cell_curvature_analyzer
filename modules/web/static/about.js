// About page: what the tool is, the method, how to run the desktop app and the web app, credits (from the README).
import { route, app, api, esc } from './core.js';

route('about', async () => {
  const info = await api('/info');
  app().innerHTML = `<p class="kicker">About</p><h1>Cell Curvature Analyzer</h1>
    <p class="lead">A PyQt5 desktop application for analysing the relationship between cell-membrane curvature and PIEZO1
      protein locations from fluorescence microscope recordings, with a web interface for the core analysis.</p>
    <h2>Method</h2>
    <ol><li><b>Edge.</b> The cell edge is the longest 0.5-level contour of the binary mask (<code>skimage.measure.find_contours</code>).</li>
      <li><b>Points.</b> ${info.defaults.n_points} points (default) are placed at equal distances along the edge.</li>
      <li><b>Curvature.</b> At each point, from its two neighbours: sign, magnitude and normalized curvature (<a href="#/convention">convention</a>).</li>
      <li><b>Intensity.</b> An inward normal is chosen by testing which side is inside the mask; a rectangle ${info.defaults.depth} px deep and
        ${info.defaults.width} px wide reaches into the cell and its mean fluorescence is recorded if at least
        ${Math.round(info.defaults.min_cell_coverage * 100)}% of it lies inside the mask (optionally retrying the opposite direction).</li>
      <li><b>Correlation.</b> Intensity is regressed on sign and on normalized curvature, pooling valid points over all frames
        (least squares; r², two-sided p, slope).</li>
      <li><b>Edge movement.</b> Consecutive masks are compared: (gained − lost) / (gained + lost) pixels gives extending, retracting or stable.</li></ol>
    <p>Parameters are in pixels. The analysis itself uses no physical scale; give the pixel size (µm/px) to see lengths in µm.
      TIFF files that carry a scale (ImageJ <code>unit=micron</code>) fill it in automatically.</p>
    <h2>Run the desktop app</h2>
    <pre>pip install -r requirements.txt
python main.py</pre>
    <p>The desktop app adds ${info.desktop_tabs.map(esc).join(', ')} tabs, temporal comparisons between frames, the
      ${info.sampling_methods.desktop_only.map(esc).join(', ')} sampling methods, and figure/report export. The web pages cover the
      core per-frame analysis with equidistant sampling.</p>
    <h2>Run the web app</h2>
    <pre>pip install -r requirements-web.txt
python run_web.py        # http://127.0.0.1:8208</pre>
    <p>The local web app sends the image and mask to its own Python server, which runs the repo's analysis core
      (<code>modules/edge_analysis.py</code>, also used by the desktop app). The static GitHub Pages site runs a JavaScript port of the same
      steps in the browser, including ports of <code>find_contours</code>, <code>draw.polygon</code> and <code>linregress</code>;
      <code>edge_engine.test.mjs</code> checks it against Python results on test masks (agreement about 1e-14).</p>
    <p>The demo data is synthetic (<code>modules/synthetic_cell.py</code>, seeded): the repository contains no measured recordings.</p>
    <h2>Credits</h2>
    <p>Built with AI assistance from <a href="https://claude.com/" target="_blank" rel="noopener">Claude (Anthropic)</a>. Licence: MIT.
      Source: <a href="https://github.com/gddickinson/${esc(info.repo)}" target="_blank" rel="noopener">github.com/gddickinson/${esc(info.repo)}</a>.</p>`;
});
