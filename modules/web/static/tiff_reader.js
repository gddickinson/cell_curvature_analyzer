// Minimal TIFF reader for image and mask stacks (no libraries): every page (IFD) is a frame.
// Supports strips and tiles; uncompressed, PackBits, LZW and Deflate (Deflate needs DecompressionStream, present in
// current browsers and Node); horizontal predictor; 1/8/16/32-bit unsigned or signed integers and 32/64-bit floats.
// Multi-sample pixels (RGB) keep the first sample. Returns { width, height, frames: [Float64Array], umPerPx, note }.
// umPerPx comes from the file (ImageJ: XResolution with unit=micron in the description; else ResolutionUnit cm),
// or null when the file has no physical scale.

const T = { W: 256, H: 257, BPS: 258, COMP: 259, DESC: 270, OFFS: 273, SPP: 277, ROWS: 278, COUNTS: 279, XRES: 282,
  PLANAR: 284, RUNIT: 296, PRED: 317, TW: 322, TL: 323, TOFFS: 324, TCOUNTS: 325, FMT: 339 };
const SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8 };

function readIFD(dv, off, le, big) {
  const tags = {};
  const n = big ? Number(dv.getBigUint64(off, le)) : dv.getUint16(off, le);
  let p = off + (big ? 8 : 2);
  const entry = big ? 20 : 12;
  for (let i = 0; i < n; i++, p += entry) {
    const tag = dv.getUint16(p, le), type = dv.getUint16(p + 2, le);
    const count = big ? Number(dv.getBigUint64(p + 4, le)) : dv.getUint32(p + 4, le);
    const bytes = (SIZE[type] || 1) * count, inline = big ? 8 : 4;
    const vo = bytes <= inline ? p + (big ? 12 : 8) : (big ? Number(dv.getBigUint64(p + 12, le)) : dv.getUint32(p + 8, le));
    const vals = [];
    if (type === 2) { tags[tag] = new TextDecoder().decode(new Uint8Array(dv.buffer, dv.byteOffset + vo, count)).replace(/\0+$/, ''); continue; }
    for (let k = 0; k < Math.min(count, 1e7); k++) {
      const q = vo + k * (SIZE[type] || 1);
      vals.push(type === 3 ? dv.getUint16(q, le) : type === 4 ? dv.getUint32(q, le) : type === 16 ? Number(dv.getBigUint64(q, le))
        : type === 5 ? dv.getUint32(q, le) / (dv.getUint32(q + 4, le) || 1) : type === 8 ? dv.getInt16(q, le)
          : type === 9 ? dv.getInt32(q, le) : type === 11 ? dv.getFloat32(q, le) : type === 12 ? dv.getFloat64(q, le) : dv.getUint8(q));
    }
    tags[tag] = vals;
  }
  const next = big ? Number(dv.getBigUint64(p, le)) : dv.getUint32(p, le);
  return { tags, next };
}

function packbits(src) {
  const out = [];
  for (let i = 0; i < src.length;) {
    const n = (src[i++] << 24) >> 24;
    if (n >= 0) { for (let k = 0; k <= n; k++) out.push(src[i++]); } else if (n !== -128) { const v = src[i++]; for (let k = 0; k < 1 - n; k++) out.push(v); }
  }
  return Uint8Array.from(out);
}

function lzw(src) {
  const out = [];
  let dict = [], width = 9, pos = 0, prev = null;
  const reset = () => { dict = []; for (let i = 0; i < 256; i++) dict[i] = [i]; dict[256] = null; dict[257] = null; width = 9; };
  const read = () => {
    let v = 0;
    for (let k = 0; k < width; k++) { const byte = src[(pos + k) >> 3]; v = (v << 1) | (((byte ?? 0) >> (7 - ((pos + k) & 7))) & 1); }
    pos += width; return v;
  };
  reset();
  while (pos + width <= src.length * 8) {
    const code = read();
    if (code === 257) break;
    if (code === 256) { reset(); prev = null; continue; }
    let entry;
    if (code < dict.length && dict[code]) entry = dict[code];
    else if (prev) entry = prev.concat(prev[0]);
    else break;
    for (const b of entry) out.push(b);
    if (prev) dict.push(prev.concat(entry[0]));
    prev = entry;
    if (dict.length + 1 >= (1 << width) && width < 12) width++;
  }
  return Uint8Array.from(out);
}

async function inflate(src) {
  const ds = new DecompressionStream('deflate');
  const stream = new Blob([src]).stream().pipeThrough(ds);
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

async function decompress(bytes, comp) {
  if (comp === 1) return bytes;
  if (comp === 32773) return packbits(bytes);
  if (comp === 5) return lzw(bytes);
  if (comp === 8 || comp === 32946) return inflate(bytes);
  throw new Error(`TIFF compression ${comp} is not supported here; save the file uncompressed, LZW or Deflate.`);
}

function unpredict(u8, w, rows, bps, spp) {
  const step = bps / 8, rowBytes = w * spp * step;
  if (bps === 8) for (let r = 0; r < rows; r++) for (let i = spp; i < w * spp; i++) u8[r * rowBytes + i] = (u8[r * rowBytes + i] + u8[r * rowBytes + i - spp]) & 255;
  else if (bps === 16) {
    const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
    for (let r = 0; r < rows; r++) for (let i = spp; i < w * spp; i++) {
      const q = r * rowBytes + 2 * i;
      dv.setUint16(q, (dv.getUint16(q, true) + dv.getUint16(q - 2 * spp, true)) & 65535, true);
    }
  }
}

// Floating-point predictor (3): per row, bytes were split into planes (most significant first) and differenced.
function unpredictFloat(u8, w, rows, bps, spp, le) {
  const step = bps / 8, rowBytes = w * spp * step, tmp = new Uint8Array(rowBytes);
  for (let r = 0; r < rows; r++) {
    const o = r * rowBytes;
    for (let i = 1; i < rowBytes; i++) u8[o + i] = (u8[o + i] + u8[o + i - 1]) & 255;
    tmp.set(u8.subarray(o, o + rowBytes));
    const n = w * spp;
    for (let k = 0; k < n; k++) for (let b = 0; b < step; b++) {
      u8[o + k * step + (le ? step - 1 - b : b)] = tmp[b * n + k];
    }
  }
}

function sampleReader(dv, bps, fmt, le) {
  if (fmt === 3) return bps === 64 ? q => dv.getFloat64(q, le) : q => dv.getFloat32(q, le);
  if (fmt === 2) return bps === 8 ? q => dv.getInt8(q) : bps === 16 ? q => dv.getInt16(q, le) : q => dv.getInt32(q, le);
  return bps === 8 ? q => dv.getUint8(q) : bps === 16 ? q => dv.getUint16(q, le) : q => dv.getUint32(q, le);
}

async function decodePage(buf, tags, le) {
  const w = tags[T.W][0], h = tags[T.H][0], bps = (tags[T.BPS] || [1])[0], spp = (tags[T.SPP] || [1])[0];
  const comp = (tags[T.COMP] || [1])[0], fmt = (tags[T.FMT] || [1])[0], pred = (tags[T.PRED] || [1])[0];
  // Planar (separate) samples: each sample plane is returned as its own frame. tifffile writes a 3- or 4-frame
  // stack saved without metadata this way, so the planes are usually frames, not colour channels.
  const planar = (tags[T.PLANAR] || [1])[0] === 2 && spp > 1;
  const planes = planar ? spp : 1, perPlane = planar ? 1 : spp;
  const outs = Array.from({ length: planes }, () => new Float64Array(w * h));
  const tiled = !!tags[T.TW];
  const bw = tiled ? tags[T.TW][0] : w, bh = tiled ? tags[T.TL][0] : (tags[T.ROWS] || [h])[0];
  const offs = tags[tiled ? T.TOFFS : T.OFFS], counts = tags[tiled ? T.TCOUNTS : T.COUNTS];
  const across = Math.ceil(w / bw), perImage = offs.length / planes;
  for (let b = 0; b < offs.length; b++) {
    const out = outs[Math.floor(b / perImage)], bi = b % perImage;
    const raw = new Uint8Array(buf, offs[b], counts ? counts[b] : buf.byteLength - offs[b]);
    const u8 = Uint8Array.from(await decompress(raw, comp));
    const r0 = tiled ? Math.floor(bi / across) * bh : bi * bh, c0 = tiled ? (bi % across) * bw : 0;
    const rows = Math.min(bh, h - r0);
    if (pred === 2) unpredict(u8, bw, tiled ? bh : rows, bps, perPlane);
    if (pred === 3) unpredictFloat(u8, bw, tiled ? bh : rows, bps, perPlane, le);
    const dv = new DataView(u8.buffer);
    if (bps === 1) {
      const rowBytes = Math.ceil(bw * perPlane / 8);
      for (let r = 0; r < rows; r++) for (let c = 0; c < bw && c0 + c < w; c++) out[(r0 + r) * w + c0 + c] = (u8[r * rowBytes + (c >> 3)] >> (7 - (c & 7))) & 1;
      continue;
    }
    const step = bps / 8, get = sampleReader(dv, bps, fmt, le);
    for (let r = 0; r < rows; r++) for (let c = 0; c < bw && c0 + c < w; c++) {
      const q = (r * bw + c) * perPlane * step;
      if (q + step <= u8.length) out[(r0 + r) * w + c0 + c] = get(q);
    }
  }
  return { w, h, data: outs };
}

function scaleFrom(tags) {
  const desc = typeof tags[T.DESC] === 'string' ? tags[T.DESC] : '';
  const xres = tags[T.XRES] ? tags[T.XRES][0] : 0;
  if (!(xres > 0)) return null;
  const unit = (desc.match(/unit=([^\n\r]+)/) || [])[1];
  if (unit && /^(micron|um|µm|\\u00B5m)$/i.test(unit.trim())) return 1 / xres;
  if (unit && /^nm$/i.test(unit.trim())) return 1e-3 / xres;
  if (!unit && (tags[T.RUNIT] || [2])[0] === 3 && xres > 1000) return 1e4 / xres;  // pixels per cm
  return null;
}

export async function readTiff(arrayBuffer) {
  const dv = new DataView(arrayBuffer);
  const bo = dv.getUint16(0, false);
  if (bo !== 0x4949 && bo !== 0x4D4D) throw new Error('Not a TIFF file.');
  const le = bo === 0x4949, magic = dv.getUint16(2, le), big = magic === 43;
  if (magic !== 42 && !big) throw new Error('Not a TIFF file.');
  let off = big ? Number(dv.getBigUint64(8, le)) : dv.getUint32(4, le);
  const frames = [];
  let width = 0, height = 0, umPerPx = null, first = true;
  while (off && frames.length < 5000) {
    const { tags, next } = readIFD(dv, off, le, big);
    const page = await decodePage(arrayBuffer, tags, le);
    if (first) { width = page.w; height = page.h; umPerPx = scaleFrom(tags); first = false; }
    if (page.w === width && page.h === height) frames.push(...page.data);
    off = next;
  }
  return { width, height, frames, umPerPx };
}
