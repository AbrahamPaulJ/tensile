// Drives the built page in a real browser, from file://, with no network: draws with the mouse, runs
// every lie through the whole dispute, and fails on any console error or failed expectation.
// usage: node test/ui.mjs [screenshot-dir]        (needs Chrome, or CHROME=/path/to/chrome)
import { mkdirSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const shots = process.argv[2] ?? null;
if (shots) mkdirSync(shots, { recursive: true });
const page_ = pathToFileURL(fileURLToPath(new URL('../dist/tensile-demo.html', import.meta.url))).href;
// Chrome as installed; PLAYWRIGHT_BUNDLED=1 uses Playwright's own Chromium (CI); CHROME=path a specific one.
const launch = process.env.CHROME
  ? { executablePath: process.env.CHROME }
  : process.env.PLAYWRIGHT_BUNDLED ? {} : { channel: 'chrome' };
const browser = await chromium.launch({ headless: true, ...launch });

let failures = 0;
const expect = (ok, what) => {
  if (!ok) {
    failures++;
    console.log(`FAIL  ${what}`);
  } else {
    console.log(`ok    ${what}`);
  }
};

// Polylines in pad coordinates (0..280), one list per stroke.
const STROKES = {
  7: [[[60, 60], [215, 60], [125, 235]]],
  1: [[[140, 45], [140, 240]]],
  0: [[[140, 45], [205, 85], [215, 150], [190, 215], [140, 240], [90, 215], [65, 150], [75, 85], [140, 45]]],
  4: [[[90, 45], [70, 150], [210, 150]], [[175, 50], [175, 245]]],
};

async function drawDigit(page, digit) {
  await page.click('#clear');
  const box = await page.locator('#pad').boundingBox();
  const k = box.width / 280;
  for (const stroke of STROKES[digit]) {
    const start = stroke[0];
    await page.mouse.move(box.x + start[0] * k, box.y + start[1] * k);
    await page.mouse.down();
    for (let s = 1; s < stroke.length; s++) {
      await page.mouse.move(box.x + stroke[s][0] * k, box.y + stroke[s][1] * k, { steps: 10 });
    }
    await page.mouse.up();
  }
  await page.waitForFunction(() => document.getElementById('run-note').textContent.includes('gas'));
}

async function newPage(viewport) {
  const context = await browser.newContext({ viewport, offline: true });
  const page = await context.newPage();
  page.on('console', (m) => {
    if (m.type() === 'error') expect(false, `console error: ${m.text()}`);
  });
  page.on('pageerror', (e) => expect(false, `page error: ${e.message}`));
  await page.goto(page_);
  await page.waitForFunction(() => window.tensileDemo?.ready === true, null, { timeout: 60000 });
  return page;
}

const shot = async (page, name) => {
  if (shots) await page.screenshot({ path: `${shots}/${name}.png` });
};

const page = await newPage({ width: 1280, height: 720 });
expect((await page.textContent('#l-engine')).includes('5 Fe contracts'), 'the EVM starts and deploys from file://, offline');
await shot(page, '00-start');

// The drawing path: digits drawn with the mouse.
for (const digit of [7, 1, 0, 4]) {
  await page.evaluate(() => { document.getElementById('run-note').textContent = ''; });
  await drawDigit(page, digit);
  const got = (await page.textContent('#answer')).trim();
  expect(got === String(digit), `a drawn ${digit} is classified as ${got}`);
  const note = await page.textContent('#run-note');
  expect(note.includes(`WebAssembly answered ${digit}`) && note.includes("ten outputs equal to the EVM's"),
    `the WebAssembly build answers ${digit} too, with the EVM's outputs`);
}
await shot(page, '01-draw');

// Every lie through the whole dispute, on a test image.
const stories = [
  ['hidden', 'SLASHED'],
  ['logit', 'SLASHED'],
  ['class', 'SLASHED'],
];
await page.locator('#samples canvas').nth(21).click();
await page.waitForFunction(() => document.getElementById('run-note').textContent.includes('gas'));
const truth = (await page.textContent('#answer')).trim();
expect(/^[0-9]$/.test(truth), `a test image is classified (${truth})`);
await page.click('#to-claim');
for (const [kind, stamp] of stories) {
  await page.check(`input[name="lie"][value="${kind}"]`);
  await page.locator('#targets button').nth(1).click();
  await page.waitForFunction(() => !document.getElementById('post').disabled);
  await shot(page, `02-claim-${kind}`);
  await page.click('#post');
  await page.waitForSelector('#act-dispute.on');
  const finding = await page.textContent('#finding');
  expect(finding.includes('wrong'), `${kind}: the watcher names a value: ${finding}`);
  await page.click('#send-named');
  await page.waitForSelector('.verdict.show');
  expect((await page.textContent('#verdict')) === stamp, `${kind}: stamped ${stamp}`);
  expect((await page.textContent('#l-claim')).includes('slashed'), `${kind}: ledger shows slashed`);
  await page.waitForTimeout(400);
  await shot(page, `03-dispute-${kind}`);
  await page.click('#to-settle');
  await page.click('#read-result');
  await page.waitForFunction(() => document.getElementById('settle-out').textContent.includes('NotFinal'));
  await page.click('#again');
  await page.waitForSelector('#act-claim.on');
}

// The honest path: challenges refused, finalize too early, then final and paid.
await page.check('input[name="lie"][value="honest"]');
await page.waitForFunction(() => !document.getElementById('post').disabled);
await page.click('#post');
await page.waitForSelector('#act-dispute.on');
expect((await page.textContent('#finding')).includes('every value matches'), 'honest claim: the watcher finds nothing');
await page.click('#try-class');
await page.waitForSelector('.verdict.stands.show');
expect((await page.textContent('#dispute-note')).includes('NoFraud'), 'honest claim: a class challenge gets NoFraud');
await page.fill('#pick-l1', '3');
await page.click('#try-l1');
await page.waitForFunction(() => document.getElementById('dispute-note').textContent.includes('NoFraud')
  && document.querySelector('#log tbody tr td.mono').textContent.includes('challengeNeuron1'));
await page.waitForTimeout(1300);
await shot(page, '04-honest-refused');
await page.click('#to-settle');
await page.click('#finalize');
await page.waitForFunction(() => document.getElementById('settle-out').textContent.includes('TooEarly'));
expect(true, 'finalize before the deadline gets TooEarly');
await page.click('#forward');
await page.waitForFunction(() => document.getElementById('settle-out').textContent.includes('clock moved'));
await page.click('#finalize');
await page.waitForFunction(() => document.getElementById('settle-out').textContent.includes('final'));
await page.click('#read-result');
await page.waitForFunction(() => document.getElementById('settle-out').textContent.includes('returns'));
expect((await page.textContent('#settle-out')).includes(`returns ${truth}`), 'result(id) returns the class after finalize');
await page.click('#w-watcher');
await page.waitForFunction(() => document.getElementById('settle-out').textContent.includes('Watcher pulled'));
await page.click('#w-claimant');
await page.waitForFunction(() => document.getElementById('settle-out').textContent.includes('Claimant pulled'));
await page.waitForTimeout(150);
expect((await page.textContent('#l-watcher')).includes('103'), `the watcher ends with three bonds: ${await page.textContent('#l-watcher')}`);
expect((await page.textContent('#l-claimant')).includes('97'), `the claimant ends three bonds down: ${await page.textContent('#l-claimant')}`);
expect((await page.textContent('#l-arbiter')).trim() === '0 ETH', 'the arbiter ends empty');
await shot(page, '05-settle');

await page.click('#steps button[data-act="fe"]');
expect((await page.textContent('#fe-diag')).includes('expected'), 'the compiler output is shown');
await shot(page, '06-fe');

// Nothing overflows sideways on a phone.
const phone = await newPage({ width: 390, height: 844 });
for (const act of ['draw', 'claim', 'dispute', 'settle', 'fe']) {
  await phone.click(`#steps button[data-act="${act}"]`);
  const overflow = await phone.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow <= 0, `phone: no sideways scroll on ${act} (${overflow}px)`);
  if (shots) await phone.screenshot({ path: `${shots}/phone-${act}.png`, fullPage: true });
}

await browser.close();
console.log(`ui test: ${failures} failed`);
process.exit(failures === 0 ? 0 : 1);
