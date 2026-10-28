// Sets hints.py's math with MathJax 4 in its default font, New Computer Modern, and writes it to hints_math.json as
// SVG paths, so the page needs neither MathJax nor the font. Each entry: the paths (y up, 1000 units to the em), the
// width, height and depth, and the x centre of the ink of each symbol marked with \cssId.
//
//   npm install --prefix /tmp/mathjax @mathjax/src@4.1.3 @mathjax/mathjax-newcm-font@4.1.3
//   node src/retrieval_blog_post/hints_math.mjs /tmp/mathjax

import {writeFileSync} from 'node:fs';
import {createRequire} from 'node:module';

const MATH = {
  formula: String.raw`\cssId{sigma}{\sigma}[\cssId{l}{\ell}, \cssId{c}{c}, \cssId{p}{p}] \cdot \cssId{h}{h}_t
    + \cssId{beta}{\beta}[\ell, c, p]`,
  ...Object.fromEntries(['in', 'mid', 'out'].map(s => [`alpha_${s}`, String.raw`\boldsymbol{\alpha}_\text{${s}}`])),
};

const MathJax = createRequire(`${process.argv[2]}/package.json`)('@mathjax/src');
await MathJax.init({
  loader: {load: ['input/tex', 'output/svg'], paths: {fonts: '@mathjax'}},
  output: {font: 'mathjax-newcm', fontCache: 'none'},
});
const dom = MathJax.startup.adaptor;

// The x extent of a path's ink: MathJax's glyphs use only absolute M, L, Q, C and Z, so the numbers come in (x, y) pairs.
function inkX(d) {
  const xs = (d.match(/-?\d*\.?\d+/g) ?? []).map(Number).filter((_, i) => i % 2 === 0);
  return [Math.min(...xs), Math.max(...xs)];
}

// The x extent of a node's ink, in the coordinates of its parent.
function extent(node) {
  let lo = Infinity, hi = -Infinity;
  if (dom.kind(node) === 'path') [lo, hi] = inkX(dom.getAttribute(node, 'd'));
  for (const c of dom.childNodes(node)) {
    if (dom.kind(c) === '#text') continue;
    const [a, b] = extent(c);
    lo = Math.min(lo, a), hi = Math.max(hi, b);
  }
  const t = (dom.getAttribute(node, 'transform') ?? '').match(/translate\(([-\d.]+),[-\d.]+\)(?: scale\(([\d.]+)\))?/);
  return t ? [+t[1] + lo * (+t[2] || 1), +t[1] + hi * (+t[2] || 1)] : [lo, hi];
}

// A marked node's ink centre: its own extent, carried up through its ancestors' translations.
function centre(node) {
  let [lo, hi] = extent(node);
  for (let p = dom.parent(node); dom.kind(p) === 'g'; p = dom.parent(p)) {
    const t = (dom.getAttribute(p, 'transform') ?? '').match(/translate\(([-\d.]+),/);
    if (t) lo += +t[1], hi += +t[1];
  }
  return +((lo + hi) / 2).toFixed(1);
}

// The node with the given id, found depth first.
function find(node, id) {
  if (dom.kind(node) === '#text') return null;
  if (dom.getAttribute(node, 'id') === id) return node;
  for (const c of dom.childNodes(node)) {
    const f = find(c, id);
    if (f) return f;
  }
  return null;
}

const out = {};
for (const [name, tex] of Object.entries(MATH)) {
  const svg = dom.firstChild(await MathJax.tex2svgPromise(tex.replace(/\s+/g, ' '), {display: true}));
  const [, top, w, h] = dom.getAttribute(svg, 'viewBox').split(' ').map(Number);
  const marks = Object.fromEntries(
    [...tex.matchAll(/\\cssId\{(\w+)\}/g)].map(([, id]) => [id, centre(find(svg, id))]),
  );
  const paths = dom.innerHTML(dom.firstChild(svg))
    .replace(/ (data-[\w-]+|id)="[^"]*"/g, '')
    .replace(/<g( transform="translate\(0,0\)")?><\/g>/g, '');
  out[name] = {svg: paths, width: w, height: -top, depth: +(h + top).toFixed(1), marks};
}
writeFileSync(new URL('hints_math.json', import.meta.url), JSON.stringify(out, null, 1) + '\n');
