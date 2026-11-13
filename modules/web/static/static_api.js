// Static-site data layer: answers api() calls without the Python server.
// GET paths come from the JSON exported by scripts/build_static.py; POST /analyze runs the JS port (edge_engine.js),
// which is tested against the Python core (edge_engine.test.mjs).
import { analyzeStack } from './edge_engine.js';
import { fromB64 } from './stack_io.js';

const files = new Map();
function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const clean = path.split('?')[0].replace(/^\/+/, '');
  if (clean === 'analyze') {
    const [T, h, w] = body.shape, n = h * w;
    const im = fromB64(body.image_f32, Float32Array), mk = fromB64(body.mask_u8, Uint8Array);
    const images = [], masks = [];
    for (let t = 0; t < T; t++) { images.push(Float64Array.from(im.subarray(t * n, (t + 1) * n))); masks.push(mk.subarray(t * n, (t + 1) * n)); }
    return analyzeStack(images, masks, h, w, { ...(await file('info')).defaults, ...(body.params || {}) });
  }
  return file(clean);
}
