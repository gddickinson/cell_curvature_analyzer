// Linear regression as scipy.stats.linregress computes it (slope, intercept, r, two-sided p from Student's t
// with n-2 degrees of freedom, standard error of the slope), plus the special functions it needs.

// log Gamma, Lanczos approximation (g = 7, n = 9); relative error ~1e-15 for x > 0.
const LG = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
  12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
export function lgamma(x) {
  if (x < 0.5) return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
  x -= 1;
  let a = LG[0];
  const t = x + 7.5;
  for (let i = 1; i < 9; i++) a += LG[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

// Continued fraction for the incomplete beta function (modified Lentz).
function betacf(a, b, x) {
  const FPMIN = 1e-300, EPS = 1e-16;
  let c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < FPMIN) d = FPMIN;
  d = 1 / d;
  let h = d;
  for (let m = 1; m <= 10000; m++) {
    const m2 = 2 * m;
    let aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d; h *= d * c;
    aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1));
    d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
    c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
    d = 1 / d;
    const del = d * c;
    h *= del;
    if (Math.abs(del - 1) < EPS) break;
  }
  return h;
}

// Regularised incomplete beta I_x(a, b).
export function betainc(a, b, x) {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const lbt = lgamma(a + b) - lgamma(a) - lgamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  if (x < (a + 1) / (a + b + 2)) return Math.exp(lbt) * betacf(a, b, x) / a;
  return 1 - Math.exp(lbt) * betacf(b, a, 1 - x) / b;
}

// Two-sided p-value of Student's t with df degrees of freedom: 2 * sf(|t|).
export function tTwoSided(t, df) {
  if (!Number.isFinite(t)) return 0;
  return betainc(df / 2, 0.5, df / (df + t * t));
}

// scipy.stats.linregress(x, y) for n >= 3 (n == 2 gives r = +-1, p = 1, as scipy does).
export function linregress(x, y) {
  const n = x.length;
  let xm = 0, ym = 0;
  for (let i = 0; i < n; i++) { xm += x[i]; ym += y[i]; }
  xm /= n; ym /= n;
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - xm, dy = y[i] - ym;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  sxx /= n; sxy /= n; syy /= n;
  const rden = Math.sqrt(sxx * syy);
  let r = rden === 0 ? 0 : sxy / rden;
  r = Math.max(-1, Math.min(1, r));
  const slope = sxy / sxx, intercept = ym - slope * xm;
  if (n === 2) return { slope, intercept, r: y[0] === y[1] ? 1 : r, p: 1, stderr: 0 };
  const df = n - 2, TINY = 1e-20;
  const t = r * Math.sqrt(df / ((1 - r + TINY) * (1 + r + TINY)));
  const p = tTwoSided(t, df);
  const stderr = Math.sqrt((1 - r * r) * syy / sxx / df);
  return { slope, intercept, r, p, stderr };
}
