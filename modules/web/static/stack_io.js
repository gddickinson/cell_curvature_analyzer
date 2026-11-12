// Loading image and mask stacks in the browser (TIFF via tiff_reader.js, PNG/JPEG via a canvas), the demo stack,
// and base64 encoding for the API. A stack is { width, height, frames: [Float64Array], umPerPx, name }.
import { readTiff } from './tiff_reader.js';

export function toB64(typed) {
  const u8 = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i += 0x8000) s += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(s);
}

export function fromB64(b64, Type) {
  const bin = atob(b64), u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Type(u8.buffer);
}

async function readBitmap(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement('canvas');
  c.width = bmp.width; c.height = bmp.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(bmp, 0, 0);
  const px = ctx.getImageData(0, 0, c.width, c.height).data, out = new Float64Array(c.width * c.height);
  for (let i = 0; i < out.length; i++) out[i] = (px[4 * i] + px[4 * i + 1] + px[4 * i + 2]) / 3;  // grey = channel mean
  return { width: c.width, height: c.height, frames: [out], umPerPx: null };
}

export async function readStack(file) {
  const isTiff = /\.tiff?$/i.test(file.name) || file.type === 'image/tiff';
  const s = isTiff ? await readTiff(await file.arrayBuffer()) : await readBitmap(file);
  if (!s.frames.length) throw new Error(`${file.name}: no image data found.`);
  return { ...s, name: file.name };
}

// Pair an image stack with a mask stack: same size; a single mask frame is reused for every image frame,
// a single image frame for every mask frame (as long as the counts otherwise match).
export function pairStacks(img, mask) {
  if (img.width !== mask.width || img.height !== mask.height) {
    throw new Error(`Image is ${img.width} x ${img.height} px but the mask is ${mask.width} x ${mask.height} px.`);
  }
  let images = img.frames, masks = mask.frames;
  if (masks.length === 1 && images.length > 1) masks = images.map(() => mask.frames[0]);
  else if (images.length === 1 && masks.length > 1) images = masks.map(() => img.frames[0]);
  else if (images.length !== masks.length) {
    throw new Error(`The image has ${images.length} frames but the mask has ${masks.length}.`);
  }
  return { width: img.width, height: img.height, images, masks: masks.map(m => Uint8Array.from(m, v => (v > 0 ? 1 : 0))) };
}

export function demoStack(demo) {
  const [T, h, w] = demo.shape, n = h * w;
  const im = fromB64(demo.image_u16, Uint16Array), mk = fromB64(demo.mask_u8, Uint8Array);
  const images = [], masks = [];
  for (let t = 0; t < T; t++) { images.push(Float64Array.from(im.subarray(t * n, (t + 1) * n))); masks.push(mk.slice(t * n, (t + 1) * n)); }
  return { width: w, height: h, images, masks };
}

// Body for POST /api/analyze (float32 image, uint8 mask, both base64 little-endian).
export function analyzeBody(pair, params) {
  const T = pair.images.length, n = pair.width * pair.height;
  const im = new Float32Array(T * n), mk = new Uint8Array(T * n);
  pair.images.forEach((f, t) => im.set(f, t * n));
  pair.masks.forEach((f, t) => mk.set(f, t * n));
  return { shape: [T, pair.height, pair.width], image_f32: toB64(im), mask_u8: toB64(mk), params };
}
