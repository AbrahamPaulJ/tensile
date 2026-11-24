// The built status page (demo/dist/tensile-fee-live.html) in headless Chromium, served over http
// as GitHub Pages would, at phone width, against the live chains (needs the network). It passes when
// the page has read Sepolia, re-run every claim, compared the relayed ticks with mainnet with no
// difference, shows no error and does not scroll sideways. A node that refuses a request (HTTP 4xx,
// after which the page asks the next node) is printed, not failed.
// usage: node demo/test/tensile-fee-live-page.mjs [screenshot.png]      (playwright's chromium)
import { createServer } from 'node:http';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const started = Date.now();
const html = readFileSync(fileURLToPath(new URL('../dist/tensile-fee-live.html', import.meta.url)));
const server = createServer((req, res) => {
  res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
  res.end(html);
});
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 900 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => {
  if (m.type() === 'error' && !/status of 4\d\d/.test(m.text())) errors.push(m.text());
});
page.on('response', (r) => {
  if (r.status() >= 400) console.log(`(a node refused a request: HTTP ${r.status()} from ${new URL(r.url()).host})`);
});
await page.goto(`http://127.0.0.1:${server.address().port}/fee-live/`);
await page.waitForFunction(() => {
  const relay = document.getElementById('s-relay').textContent;
  const tags = [...document.querySelectorAll('#claims .tag')];
  return document.getElementById('error').textContent
    || (tags.length && tags.every((t) => !t.classList.contains('wait')) && /equal|DIFFERENT/.test(relay) && document.getElementById('relay-note').textContent.includes('relayed at the start'));
}, null, { timeout: 150000 }).catch((e) => errors.push(`the page did not finish: ${e.message}`));
const seen = await page.evaluate(() => ({
  fee: document.getElementById('s-fee').textContent,
  relay: document.getElementById('s-relay').textContent,
  claims: document.getElementById('s-claims').textContent,
  lag: document.getElementById('s-lag').textContent,
  tags: [...document.querySelectorAll('#claims li')].map((li) => `${li.children[0].textContent} ${li.children[1].textContent}: ${li.children[2].textContent}`),
  different: document.querySelectorAll('#ticks td.no').length,
  sideways: document.documentElement.scrollWidth > window.innerWidth,
  error: document.getElementById('error').textContent,
}));
if (process.argv[2]) await page.screenshot({ path: process.argv[2], fullPage: true });
await browser.close();
server.close();
console.log(`fee ${seen.fee}; relay against mainnet ${seen.relay}; relay behind ${seen.lag}; claims ${seen.claims}`);
for (const t of seen.tags) console.log(`  ${t}`);
if (seen.error) errors.push(`the page shows an error: ${seen.error}`);
if (seen.different) errors.push(`${seen.different} relayed tick(s) differ from mainnet`);
if (seen.sideways) errors.push('the page scrolls sideways at 390 px');
if (seen.tags.some((t) => /FINAL$/.test(t))) errors.push('a false claim is final');
for (const e of errors) console.log(`WRONG: ${e}`);
console.log(`tensile-fee-live page: ${errors.length} wrong, ${((Date.now() - started) / 1000).toFixed(1)} s`);
process.exit(errors.length ? 1 : 0);
