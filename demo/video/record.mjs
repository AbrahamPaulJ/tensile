// Records the demo video: drives the real built page in Chrome, with no edits to what it computes, and
// captures every frame the browser paints through the DevTools screencast. Each beat of script.json
// starts its action and its narration together, and the next beat waits for both to finish, so nothing
// the page shows is sped up or faked. The recorder adds only presentation on top: a pointer, a camera
// (a CSS zoom of the whole page toward what is being talked about), and full-frame title and number
// cards. Writes work/frames/*.jpg and work/timeline.json for assemble.py.
// usage: node demo/video/record.mjs      (after tts.py)
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { chromium } from 'playwright';

const here = new URL('.', import.meta.url);
const path = (p) => fileURLToPath(new URL(p, here));
const script = JSON.parse(readFileSync(path('script.json'), 'utf8'));
const { seconds } = JSON.parse(readFileSync(path('work/durations.json'), 'utf8'));
// The figures on the cards come from the generated docs, never typed here.
const doc = (f) => readFileSync(path(`../../docs/${f}`), 'utf8');
const grab = (text, re, what) => { const m = text.match(re); if (!m) throw new Error(`no ${what} in docs`); return m; };
const numbersMd = doc('numbers.md'), mutants = doc('mutation-results.md');
const faults = grab(mutants, /Totals: (\d+) compiler, (\d+) killed \(of (\d+)\)/, 'mutant totals');
if (faults[3] !== '59' || Number(faults[1]) + Number(faults[2]) !== 59) throw new Error('the narration says 59 of 59 faults caught');
const numbers = {
  classify_gas: Number(grab(numbersMd, /`Network\.classify`[^|]*\| ([\d,]+) \|/, 'classify gas')[1].replace(/,/g, '')),
  claim_vs_solidity: grab(numbersMd, /\| `claim` \| [\d,]+ \| [\d,]+ \| ([\d.]+)x \|/, 'claim ratio')[1],
  faults_by_compiler: faults[1],
  faults_by_tests: faults[2],
  fe_version: grab(numbersMd, /^fe (\d+\.\d+\.\d+)/m, 'fe version')[1],
  wasm_commit: grab(readFileSync(path('../../tools/build_wasm.sh'), 'utf8'), /\b([0-9a-f]{7})[0-9a-f]{33}\b/, 'wasm commit')[1],
};
rmSync(path('work/frames'), { recursive: true, force: true });
mkdirSync(path('work/frames'), { recursive: true });
const VIEW = { width: 1440, height: 810 }; // 1920 x 1080 on screen
const GAP_MS = 250;
const font = (f) => readFileSync(path(`fonts/${f}`)).toString('base64');

const browser = await chromium.launch({ headless: true, channel: 'chrome' });
const context = await browser.newContext({ viewport: VIEW, deviceScaleFactor: 4 / 3, offline: true, colorScheme: 'light' });
const page = await context.newPage();
page.on('pageerror', (e) => { console.error('page error', e); process.exit(1); });
await page.goto(pathToFileURL(fileURLToPath(new URL('../dist/tensile-demo.html', here))).href);
await page.waitForFunction(() => window.tensileDemo?.ready === true, null, { timeout: 60000 });

// Presentation layer. Everything here sits outside <body> or only transforms it; the page's own code and
// numbers are untouched.
await page.addStyleTag({ content: `
  @font-face { font-family: Geist; src: url(data:font/ttf;base64,${font('Geist.ttf')}) format('truetype'); font-weight: 100 900; }
  @font-face { font-family: GeistMono; src: url(data:font/ttf;base64,${font('GeistMono.ttf')}) format('truetype'); font-weight: 100 900; }
  html { overflow: hidden; }
  html.rec-carded #rec-cursor { opacity: 0; }
  body { transform-origin: 0 0; transition: transform var(--cam-ms, 900ms) cubic-bezier(.65, 0, .35, 1); }
  #rec-cursor { position: fixed; left: 0; top: 0; width: 26px; height: 26px; z-index: 99; pointer-events: none; }
  #rec-card { position: fixed; inset: 0; z-index: 90; background: radial-gradient(1200px 700px at 50% 38%, #16202c 0%, #0b0f14 70%); color: #eef2f6;
    display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 0 110px 120px;
    opacity: 0; transition: opacity .5s; pointer-events: none; font-family: Geist, sans-serif; }
  #rec-card.on { opacity: 1; }
  #rec-card .in { opacity: 0; transform: translateY(26px); animation: rec-in .7s cubic-bezier(.2, .7, .2, 1) forwards; animation-delay: var(--d, 0s); }
  @keyframes rec-in { to { opacity: 1; transform: none; } }
  #rec-card .out { animation: rec-out .45s ease forwards; animation-delay: var(--d, 0s); }
  @keyframes rec-out { to { opacity: .22; } }
  #rec-card .kicker { font: 500 22px Geist; letter-spacing: .16em; text-transform: uppercase; color: #8aa0b4; margin: 0 0 18px; }
  #rec-card .big { font: 700 150px/1 GeistMono; letter-spacing: -.03em; font-variant-numeric: tabular-nums; margin: 0; }
  #rec-card .big small { font: 600 54px Geist; letter-spacing: 0; color: #8aa0b4; margin-left: 18px; }
  #rec-card .line { font: 600 50px/1.18 Geist; letter-spacing: -.01em; margin: 10px 0; }
  #rec-card .dim { color: #8aa0b4; }
  #rec-card .red { color: #ff5d5d; font-size: 92px; font-weight: 750; margin-top: 34px; }
  #rec-card .mark { font: 750 168px/1 Geist; letter-spacing: -.045em; margin: 0; }
  #rec-card .sub { font: 500 38px/1.3 Geist; color: #c5d1dc; margin: 22px 0 0; max-width: 1100px; }
  #rec-card .chip { display: inline-block; font: 600 24px GeistMono; color: #0b0f14; background: #7cc4ff; border-radius: 999px; padding: 8px 22px; margin-top: 36px; }
  #rec-card .stats { display: grid; grid-template-columns: auto 1fr; gap: 26px 44px; align-items: baseline; text-align: left; }
  #rec-card .stats .n { font: 700 84px/1 GeistMono; font-variant-numeric: tabular-nums; color: #7cc4ff; text-align: right; white-space: nowrap; }
  #rec-card .stats .t { font: 500 31px/1.3 Geist; color: #d6dee6; max-width: 760px; }
  #rec-card .url { font: 500 30px GeistMono; color: #7cc4ff; margin-top: 40px; }
  #rec-card .fine { position: absolute; left: 0; right: 0; bottom: 34px; font: 400 17px/1.5 Geist; color: #7f8e9c; padding: 0 160px; }
  .rec-focus { outline: 4px solid #2f81f7 !important; outline-offset: 6px; border-radius: 6px; transition: outline-color .3s; }
` });
await page.evaluate(() => {
  const root = document.documentElement;
  // Below a short page, the page's own background, not a black band.
  root.style.background = getComputedStyle(document.body).backgroundColor;
  const c = document.createElement('div');
  c.id = 'rec-cursor';
  c.innerHTML = '<svg viewBox="0 0 22 22" width="26" height="26"><path d="M2 1 L2 18 L7 13.5 L10.5 21 L13.5 19.6 L10 12.3 L17 12 Z" fill="#111" stroke="#fff" stroke-width="1.4"/></svg>';
  root.append(c);
  const card = document.createElement('div');
  card.id = 'rec-card';
  root.append(card);
  window.addEventListener('mousemove', (e) => { c.style.transform = `translate(${e.clientX}px, ${e.clientY}px)`; }, true);
  // Count-up numbers on the cards: <span data-to="1896043" data-ms="1800" data-delay="0.2">
  window.recCount = () => {
    for (const el of document.querySelectorAll('#rec-card [data-to]')) {
      const to = Number(el.dataset.to), ms = Number(el.dataset.ms || 1500), delay = Number(el.dataset.delay || 0) * 1000;
      const t0 = performance.now() + delay;
      const tick = (now) => {
        const t = Math.min(1, Math.max(0, (now - t0) / ms));
        const e = 1 - Math.pow(1 - t, 3);
        el.textContent = Math.round(to * e).toLocaleString('en-US');
        if (t < 1) requestAnimationFrame(tick);
      };
      el.textContent = '0';
      requestAnimationFrame(tick);
    }
  };
});
let mouse = { x: 720, y: 560 };
await page.mouse.move(mouse.x, mouse.y);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// Where a phrase falls in a beat's narration, in seconds, by its share of the characters spoken.
function when(id, phrase) {
  const beat = script.beats.find((b) => b.id === id);
  const said = beat.say ?? beat.text;
  const at = said.indexOf(phrase);
  if (at < 0) throw new Error(`"${phrase}" is not in beat ${id}`);
  return (seconds[id] * at) / said.length;
}

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
async function point(selector, ms) {
  const box = await page.locator(selector).first().boundingBox();
  await glide(box.x + box.width / 2, box.y + box.height / 2, ms);
}
async function click(selector, settle = 250) {
  await point(selector);
  await sleep(120);
  await page.mouse.down();
  await page.mouse.up();
  await sleep(settle);
}

// The camera: a scale and translate on <body>. `cam(selectors, opts)` frames the union of the elements,
// zoomed as far as fits (up to opts.max), placed a little above centre so the captions do not cover it.
// `cam(null)` returns to the whole page. Clicks wait for the move to finish, since boxes move with it.
let view = { s: 1, x: 0, y: 0 };
async function cam(selectors, { ms = 900, max = 2.2, fill = 0.78, s: fixed } = {}) {
  let next = { s: 1, x: 0, y: 0 };
  if (selectors) {
    const list = Array.isArray(selectors) ? selectors : [selectors];
    const r = await page.evaluate((sels) => {
      const boxes = sels.map((s) => document.querySelector(s).getBoundingClientRect());
      const x = Math.min(...boxes.map((b) => b.left)), y = Math.min(...boxes.map((b) => b.top));
      const r = Math.max(...boxes.map((b) => b.right)), b = Math.max(...boxes.map((b) => b.bottom));
      return { x, y, w: r - x, h: b - y, docH: document.body.scrollHeight };
    }, list);
    // Back to page coordinates, out of the current camera.
    const u = { x: (r.x - view.x) / view.s, y: (r.y - view.y) / view.s, w: r.w / view.s, h: r.h / view.s };
    const s = fixed ?? Math.min(max, (VIEW.width * fill) / u.w, (VIEW.height * 0.66) / u.h);
    const cx = u.x + u.w / 2, cy = u.y + u.h / 2;
    const docH = Math.max(VIEW.height, r.docH / view.s);
    const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
    next = {
      s,
      x: clamp(VIEW.width / 2 - s * cx, VIEW.width - s * VIEW.width, 0),
      y: clamp(VIEW.height * 0.43 - s * cy, VIEW.height - s * docH, 0),
    };
  }
  await page.evaluate(([v, ms]) => {
    document.body.style.setProperty('--cam-ms', `${ms}ms`);
    document.body.style.transform = `translate(${v.x}px, ${v.y}px) scale(${v.s})`;
  }, [next, ms]);
  view = next;
  await sleep(ms + 60);
}

const card = (html) => page.evaluate((h) => {
  const el = document.getElementById('rec-card');
  if (h === null) el.classList.remove('on');
  else { el.innerHTML = h; el.classList.add('on'); window.recCount(); }
  document.documentElement.classList.toggle('rec-carded', h !== null);
}, html);
const waitNote = (id, text) => page.waitForFunction(([i, t]) => document.getElementById(i).textContent.includes(t), [id, text], { timeout: 30000 });
const fmt = (n) => n.toLocaleString('en-US');

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
  hook: () => card(`
    <p class="kicker in">One neural network answer, on Ethereum</p>
    <p class="big in" style="--d:.15s"><span data-to="${numbers.classify_gas}" data-ms="2300" data-delay="0.2">0</span><small>gas</small></p>
    <p class="line in" style="--d:${when('hook', 'For one').toFixed(2)}s">for one answer.</p>`),
  dodge: () => card(`
    <p class="line in">So nobody runs it.</p>
    <p class="line in dim" style="--d:${when('dodge', 'Someone').toFixed(2)}s">Someone posts the answer, and a bond.</p>
    <p class="line red in" style="--d:${when('dodge', 'But what').toFixed(2)}s">What if it's a lie?</p>`),
  title: () => card(`
    <p class="mark in">Tensile</p>
    <p class="sub in" style="--d:${when('title', 'It catches').toFixed(2)}s">Catch a false answer by re-running one neuron.</p>
    <p class="chip in" style="--d:${when('title', 'And it').toFixed(2)}s">written in Fe</p>`),
  page: async () => {
    await card(null);
    await sleep(700);
    await sleep(when('page', 'a real') * 1000 - 700);
    await cam('#engine', { max: 2.4, fill: 0.6 });
    await point('#l-engine');
  },
  draw: async () => {
    await cam(['#pad', '#clear'], { max: 1.9, ms: 800 });
    await drawSeven();
  },
  answer: async () => {
    await cam(['#hidden', '#answer'], { max: 1.8, ms: 800 });
    await point('#answer');
    if ((await page.textContent('#answer')).trim() !== '7') throw new Error('the network did not answer 7: the narration says it does');
    await sleep(Math.max(0, when('answer', 'Right') * 1000 - 1500));
    await cam(['#run-note'], { max: 2.0, fill: 0.7, ms: 800 });
    await point('#run-note');
  },
  lie: async () => {
    await cam(null, { ms: 600 });
    await click('#to-claim', 500);
    await cam(['#lies', '#targets'], { max: 1.9, ms: 800 });
    // The lie first: the answer buttons are disabled while "Tell the truth" is selected.
    await click('input[name="lie"][value="hidden"]', 200);
    await page.waitForFunction(() => !document.getElementById('post').disabled);
    await sleep(300);
    await click('#targets button:nth-child(2)', 200);
    await page.waitForFunction(() => !document.getElementById('post').disabled
      && document.getElementById('c-claim-class').textContent.trim() === '1', null, { timeout: 15000 });
    await sleep(Math.max(0, when('lie', 'we forge') * 1000 - 3500));
    await cam(['#c-claim-hidden', '#c-claim-logits'], { max: 1.9, ms: 900 });
    await point('#c-claim-hidden');
    await sleep(1600);
    await point('#c-claim-logits');
  },
  post: async () => {
    await cam(['#post'], { max: 1.8, fill: 0.5, ms: 700 });
    await click('#post', 200);
    await page.waitForSelector('#act-dispute.on');
    // The narration says the claim is that this seven is a one. Stop the recording if it is not.
    if (!(await page.textContent('#l-claim')).includes('says 1')) throw new Error('the claim on screen is not the narrated one');
    await cam(['#ledger'], { max: 1.6, fill: 0.95, ms: 800 });
    await point('#l-arbiter');
  },
  watcher: async () => {
    await cam(['#finding'], { max: 2.1, fill: 0.62, ms: 900 });
    await sleep(Math.max(0, when('watcher', 'and finds') * 1000 - 1000));
    await point('#finding');
  },
  wrong: async () => {
    await cam(['#tries', '#verdict', '#checks'], { max: 1.5, fill: 0.95, ms: 800 });
    await page.evaluate(() => { document.getElementById('tries').open = true; });
    await sleep(400);
    await click('#try-class', 100);
    await waitNote('dispute-note', 'NoFraud');
    await sleep(300);
    await cam(['#checks', '#tree'], { max: 1.8, fill: 0.7, ms: 800 });
  },
  right: async () => {
    // Say what the contract will do first; the click lands so that the stamp follows the sentence.
    await page.evaluate(() => { document.getElementById('tries').open = false; });
    await cam(['#finding', '#checks'], { max: 1.5, fill: 0.95, ms: 800 });
    await point('#send-named');
    await sleep(Math.max(0, seconds.right * 1000 - 2400));
    await click('#send-named', 100);
    await page.waitForSelector('.verdict.slashed.show', { timeout: 30000 });
  },
  slashed: async () => {
    await cam(['#verdict'], { s: 2.4, ms: 420 });
    await sleep(1300);
    await cam(['#ledger'], { max: 1.6, fill: 0.95, ms: 800 });
    await point('#l-watcher');
    await sleep(Math.max(0, when('slashed', 'It cost') * 1000 - 3000));
    await cam(['#gas-bars'], { max: 2.0, fill: 0.7, ms: 800 });
  },
  honest: async () => {
    await cam(null, { ms: 600 });
    await click('#to-settle', 200);
    await click('#again', 300);
    await click('input[name="lie"][value="honest"]', 300);
    await click('#post', 150);
    await page.waitForSelector('#act-dispute.on');
    await cam(['#tries', '#verdict', '#checks'], { max: 1.5, fill: 0.95, ms: 600 });
    await page.evaluate(() => { document.getElementById('tries').open = true; });
    await sleep(200);
    await click('#try-l1', 100);
    await waitNote('dispute-note', 'NoFraud');
    await sleep(500);
    await cam(null, { ms: 500 });
    await click('#to-settle', 300);
    await click('#forward', 300);
    await click('#finalize', 150);
    await waitNote('settle-out', 'final');
    await cam(['#settle-out'], { max: 1.8, fill: 0.7, ms: 600 });
  },
  types: async () => {
    await cam(null, { ms: 500 });
    await click('#steps button[data-act="fe"]', 400);
    await cam(['#fe-tabs', '#fe-diag'], { max: 1.8, fill: 0.62, ms: 800 });
    await sleep(Math.max(0, when('types', 'Wire up') * 1000 - 2400));
    await click('#fe-tabs button:nth-child(2)', 300);
    await sleep(1800);
    await click('#fe-tabs button:nth-child(3)', 300);
  },
  effects: async () => {
    await cam(['#fe-effects'], { max: 1.8, fill: 0.62, ms: 800 });
    await point('#fe-effects .hl');
    await sleep(Math.max(0, when('effects', 'And the') * 1000 - 1400));
    await cam(['#fe-costs'], { max: 1.8, fill: 0.7, ms: 800 });
    await point('#fe-costs tr:nth-child(2)');
  },
  evidence: () => card(`
    <div class="stats">
      <div class="n in"><span data-to="10000" data-ms="1500" data-delay=".3">0</span></div>
      <div class="t in">of 10,000 test images classified bit for bit the same by the NumPy reference, native Fe, WebAssembly and the EVM</div>
      <div class="n in" style="--d:${when('evidence', 'We planted').toFixed(2)}s"><span data-to="59" data-ms="1300" data-delay="${when('evidence', 'We planted').toFixed(2)}">0</span>/59</div>
      <div class="t in" style="--d:${when('evidence', 'We planted').toFixed(2)}s">deliberate faults in the contract caught: ${numbers.faults_by_tests} by tests, ${numbers.faults_by_compiler} refused by the compiler</div>
      <div class="n in" style="--d:${when('evidence', 'And F E').toFixed(2)}s">${numbers.claim_vs_solidity}x</div>
      <div class="t in" style="--d:${when('evidence', 'And F E').toFixed(2)}s">Fe's gas per claim, against a hand-written Solidity twin that passes the same tests</div>
    </div>`),
  end: () => card(`
    <p class="mark in">Tensile</p>
    <p class="sub in" style="--d:.4s">A false answer, caught by one neuron.</p>
    <p class="url in" style="--d:.9s">github.com/AbrahamPaulJ/tensile</p>
    <p class="fine in" style="--d:1.2s">Limits: needs at least one honest watcher; a small dense model; not audited, not deployed.<br>
    Fe ${numbers.fe_version}; WebAssembly with Fe master ${numbers.wasm_commit}. The page was recorded in real time; the zoom and the cards are added by the recorder.
    Narration: Kokoro, a synthetic voice. Captions: ffmpeg-skill.</p>`),
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
await card('');
await sleep(600);
await cdp.send('Page.startScreencast', { format: 'jpeg', quality: 92, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
await sleep(300);

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
await sleep(2500);
const finished = Date.now() / 1000;
await cdp.send('Page.stopScreencast');
await sleep(300);
writeFileSync(path('work/timeline.json'), JSON.stringify({ frames, beats, finished }, null, 0));
console.log(`${frames.length} frames, ${(finished - frames[0].t).toFixed(1)} s`);
await browser.close();
