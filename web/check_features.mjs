// Check 2 of isl.parity: the JavaScript feature code against features.py, bit for bit.
//
//   python -m isl.parity features && node web/check_features.mjs
//
// For every clip's cached landmarks, isl.js builds the (64, 478) model input and it is compared
// with numpy's, float by float. Exit code 1 if any clip differs.
import { readFileSync, writeFileSync } from 'node:fs';
import { framesToInput } from './isl.js';

const dir = new URL('../data/parity/features/', import.meta.url);
const index = JSON.parse(readFileSync(new URL('index.json', dir)));
const raw = new Float32Array(readFileSync(new URL('raw.f32', dir)).buffer.slice(0));
const xs = new Float32Array(readFileSync(new URL('x.f32', dir)).buffer.slice(0));
const F = index.frame_floats, X = index.x_floats;

let identical = 0, differingValues = 0, maxAbs = 0;
index.clips.forEach((c, i) => {
  const frames = [];
  for (let t = 0; t < c.frames; t++) {
    const row = raw.subarray(c.raw_offset + t * F, c.raw_offset + (t + 1) * F);
    frames.push({ pose: row.subarray(0, 132), left: row.subarray(132, 195), right: row.subarray(195, 258), mask: row.subarray(258, 260) });
  }
  const js = framesToInput(frames), py = xs.subarray(i * X, (i + 1) * X);
  const jb = new Uint32Array(js.buffer), pb = new Uint32Array(py.slice().buffer);
  let diff = 0;
  for (let k = 0; k < X; k++) {
    if (jb[k] !== pb[k]) { diff++; maxAbs = Math.max(maxAbs, Math.abs(js[k] - py[k])); }
  }
  if (diff === 0) identical++; else differingValues += diff;
});

const report = { clips: index.clips.length, bit_identical_clips: identical, differing_values: differingValues,
  values_compared: index.clips.length * X, max_abs_diff: maxAbs };
console.log(JSON.stringify(report, null, 2));
writeFileSync(new URL('../artifacts/feature_parity.json', import.meta.url), JSON.stringify(report, null, 2) + '\n');
process.exit(identical === index.clips.length ? 0 : 1);
