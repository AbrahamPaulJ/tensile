// Records the demo video: drives the real built page in Chrome, with no edits to what it computes, and
// captures every frame the browser paints through the DevTools screencast. Each beat of script.json
// starts its action and its narration together, and the next beat waits for both to finish, so nothing
// the page shows is sped up or faked. Writes work/frames/*.jpg and work/timeline.json for assemble.py.
// usage: node demo/video/record.mjs      (after tts.py)
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = new URL('.', import.meta.url);
const path = (p) => fileURLToPath(new URL(p, here));
const script = JSON.parse(readFileSync(path('script.json'), 'utf8'));
const { seconds } = JSON.parse(readFileSync(path('work/durations.json'), 'utf8'));
rmSync(path('work/frames'), { recursive: true, force: true });
mkdirSync(path('work/frames'), { recursive: true });
const VIEW = { width: 1440, height: 720 }; // 1920 x 960 on screen; the strip below it holds the captions
const GAP_MS = 350;

const browser = await chromium.launch({ headless: true, channel: 'chrome' });
const context = await browser.newContext({ viewport: VIEW, deviceScaleFactor: 4 / 3, offline: true, colorScheme: 'light' });
const page = await context.newPage();
page.on('pageerror', (e) => { console.error('page error', e); process.exit(1); });
await page.goto(pathToFileURL(fileURLToPath(new URL('../dist/tensile-demo.html', here))).href);
await page.waitForFunction(() => window.tensileDemo?.ready === true, null, { timeout: 60000 });

// A visible pointer, and the full-frame cards. Added by this recorder only, never part of the page.
await page.addStyleTag({ content: `
  #rec-cursor { position: fixed; left: 0; top: 0; width: 22px; height: 22px; z-index: 99; pointer-events: none; transition: none; }
  #rec-card { position: fixed; inset: 0; z-index: 90; background: #0c1117; color: #e8edf2; display: flex; flex-direction: column; justify-content: center; padding: 0 130px; opacity: 0; transition: opacity .45s; pointer-events: none; }
  #rec-card.on { opacity: 1; }
  #rec-card h1 { font: 600 70px Bahnschrift, "Segoe UI", sans-serif; letter-spacing: .02em; margin: 0 0 14px; }
  #rec-card p { font: 400 25px/1.4 "Segoe UI", sans-serif; margin: 0 0 8px; color: #c3ced8; max-width: 1040px; }
  #rec-card table { border-collapse: collapse; font: 400 23px/1.3 "Segoe UI", sans-serif; }
  #rec-card td { padding: 11px 44px 11px 0; border-top: 1px solid #2d3844; vertical-align: baseline; }
  #rec-card td:first-child { font: 500 36px "Cascadia Mono", Consolas, monospace; color: #79b0ee; white-space: nowrap; }
  #rec-card small { font: 400 17px "Segoe UI", sans-serif; color: #9aa8b5; margin-top: 28px; display: block; }
` });
await page.evaluate(() => {
  const c = document.createElement('div');
  c.id = 'rec-cursor';
  c.innerHTML = '<svg viewBox="0 0 22 22" width="22" height="22"><path d="M2 1 L2 18 L7 13.5 L10.5 21 L13.5 19.6 L10 12.3 L17 12 Z" fill="#111" stroke="#fff" stroke-width="1.4"/></svg>';
  document.body.append(c);
  const card = document.createElement('div');
  card.id = 'rec-card';
  document.body.append(card);
  window.addEventListener('mousemove', (e) => { c.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`; }, true);
});
let mouse = { x: 720, y: 500 };
await page.mouse.move(mouse.x, mouse.y);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function glide(x, y, ms = 420) {
  const steps = Math.max(6, Math.round(ms / 16));
  const from = mouse;
  for (let i = 1; i <= steps; i++) {
    const t = i / steps;
    const e = t * t * (3 - 2 * t);
    await page.mouse.move(from.x + (x - from.x) * e, from.y + (y - from.y) * e);
    await sleep(ms / steps);
  }
  mouse = { x, y };
}
async function point(selector) {
  const box = await page.locator(selector).first().boundingBox();
  await glide(box.x + box.width / 2, box.y + box.height / 2);
}
async function click(selector, settle = 250) {
  await point(selector);
  await sleep(120);
  await page.mouse.down();
  await page.mouse.up();
  await sleep(settle);
}
const card = (html) => page.evaluate((h) => {
  const el = document.getElementById('rec-card');
  if (h === null) el.classList.remove('on');
  else { el.innerHTML = h; el.classList.add('on'); }
}, html);
const waitNote = (id, text) => page.waitForFunction(([i, t]) => document.getElementById(i).textContent.includes(t), [id, text], { timeout: 30000 });

async function drawSeven() {
  const box = await page.locator('#pad').boundingBox();
  const k = box.width / 280;
  const at = (x, y) => [box.x + x * k, box.y + y * k];
  await glide(...at(60, 62), 500);
  await page.mouse.down();
  await glide(...at(215, 60), 750);
  await glide(...at(125, 235), 850);
  await page.mouse.up();
  await waitNote('run-note', 'gas');
}

// What each beat does on screen. The narration for it starts at the same moment.
const ACT = {
  title: () => card('<h1>Tensile</h1><p>One kernel in Fe. Native to classify, on the EVM to settle a dispute.</p>'),
  page: async () => {
    await sleep(1400);
    await card(null);
    await sleep(900);
    await point('#l-engine');
  },
  draw: drawSeven,
  cost: async () => {
    await point('#answer');
    await sleep(2200);
    await point('#run-note');
  },
  claim: async () => {
    await click('#to-claim', 500);
    await sleep(1200);
    await point('#lies');
  },
  lie: async () => {
    // The lie first: the answer buttons are disabled while "Tell the truth" is selected.
    await sleep(500);
    await click('input[name="lie"][value="hidden"]', 200);
    await page.waitForFunction(() => !document.getElementById('post').disabled);
    await sleep(500);
    await click('#targets button:nth-child(2)', 200);
    await page.waitForFunction(() => !document.getElementById('post').disabled
      && document.getElementById('c-claim-class').textContent.trim() === '1', null, { timeout: 15000 });
    await sleep(900);
    await point('#c-claim-hidden');
    await sleep(1500);
    await point('#c-claim-logits');
  },
  post: async () => {
    await click('#post', 200);
    await page.waitForSelector('#act-dispute.on');
    // The narration says the claim is that this seven is a one. Stop the recording if it is not.
    if (!(await page.textContent('#l-claim')).includes('says 1')) throw new Error('the claim on screen is not the narrated one');
  },
  finding: async () => {
    await sleep(1200);
    await point('#finding');
    await sleep(2500);
    await point('#send-named');
  },
  wrong: async () => {
    await sleep(800);
    await click('#tries summary', 500);
    await click('#try-class', 100);
    await waitNote('dispute-note', 'NoFraud');
    await sleep(600);
    await point('#dispute-note');
  },
  right: async () => {
    // Say what the contract will do first; the click lands so that the stamp follows the sentence.
    await page.evaluate(() => { document.getElementById('tries').open = false; });
    await point('#checks');
    await sleep(Math.max(0, seconds.right * 1000 - 2800));
    await click('#send-named', 100);
    await page.waitForSelector('.verdict.slashed.show', { timeout: 30000 });
    await sleep(200);
  },
  result: async () => {
    await point('#gas-bars');
    await sleep(1800);
    await point('#l-watcher');
  },
  honest: async () => {
    await click('#to-settle', 250);
    await click('#again', 400);
    await click('input[name="lie"][value="honest"]', 400);
    await click('#post', 200);
    await page.waitForSelector('#act-dispute.on');
    await sleep(500);
    await click('#try-l1', 100);
    await waitNote('dispute-note', 'NoFraud');
  },
  final: async () => {
    await click('#to-settle', 450);
    await click('#forward', 500);
    await click('#finalize', 200);
    await waitNote('settle-out', 'final');
    await sleep(400);
    await click('#read-result', 200);
    await waitNote('settle-out', 'returns');
    await sleep(1100);
    await click('#w-watcher', 200);
    await waitNote('settle-out', 'Watcher pulled');
    await sleep(500);
    await click('#w-claimant', 200);
    await waitNote('settle-out', 'Claimant pulled');
  },
  types: async () => {
    await click('#steps button[data-act="fe"]', 500);
    await sleep(2400);
    await click('#fe-tabs button:nth-child(2)', 300);
    await sleep(2600);
    await click('#fe-tabs button:nth-child(3)', 300);
    await sleep(1500);
    await click('#fe-tabs button:nth-child(4)', 300);
  },
  effects: async () => {
    await point('#fe-effects .hl');
    await sleep(4800);
    await point('#fe-costs');
  },
  evidence: () => card(`<table>
    <tr><td>0 / 10,000</td><td>mismatches: NumPy reference, native build and EVM build, bit for bit</td></tr>
    <tr><td>45 tests</td><td>Foundry fuzz and invariant tests on the arbiter's properties P1 to P6</td></tr>
    <tr><td>53 of 53</td><td>hand-chosen faults caught: 45 by those tests, 8 by compile-time assertions</td></tr>
    <tr><td>1.26x</td><td>Fe's gas per claim, against a Solidity twin that passes the same tests</td></tr>
  </table>`),
  limits: () => card(`<h1 style="font-size:46px">Limits</h1>
    <p>Needs at least one honest watcher. A claimant can slash its own false claim and recover the bond.</p>
    <p>A small dense model. Not audited, not deployed. Listed in full in the README.</p>`),
  end: () => card(`<h1>Tensile</h1><p>A false answer, caught by one neuron.</p>
    <small>Fe 26.5.1. Page recorded in real time. Narration is a synthetic voice (Kokoro).</small>`),
};

// --------------------------------------------------------------------------------------------------
const cdp = await context.newCDPSession(page);
const frames = [];
cdp.on('Page.screencastFrame', async ({ data, metadata, sessionId }) => {
  const n = frames.length;
  const file = `f${String(n).padStart(6, '0')}.jpg`;
  frames.push({ file, t: metadata.timestamp });
  writeFileSync(path(`work/frames/${file}`), Buffer.from(data, 'base64'));
  try { await cdp.send('Page.screencastFrameAck', { sessionId }); } catch { /* closing */ }
});
await card('<h1>Tensile</h1><p>One kernel in Fe. Native to classify, on the EVM to settle a dispute.</p>');
await sleep(600);
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1920, maxHeight: 960, everyNthFrame: 1 });
await sleep(400);

const beats = [];
for (const beat of script.beats) {
  const started = Date.now() / 1000;
  const action = ACT[beat.id];
  if (!action) throw new Error(`no action for beat ${beat.id}`);
  await Promise.all([action(), sleep(seconds[beat.id] * 1000)]);
  const ended = Date.now() / 1000;
  beats.push({ id: beat.id, text: beat.text, start: started, end: ended, speech: seconds[beat.id] });
  console.log(`${beat.id.padEnd(10)} speech ${seconds[beat.id].toFixed(1)} s, on screen ${(ended - started).toFixed(1)} s`);
  await sleep(GAP_MS);
}
await sleep(1500);
const finished = Date.now() / 1000;
await cdp.send('Page.stopScreencast');
await sleep(300);
writeFileSync(path('work/timeline.json'), JSON.stringify({ frames, beats, finished }, null, 0));
console.log(`${frames.length} frames, ${(finished - frames[0].t).toFixed(1)} s`);
await browser.close();
