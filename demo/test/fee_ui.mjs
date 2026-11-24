// Drives the built fee-tier page in a real browser, from file://, with no network: picks windows,
// feeds the ticks through FeeOracle, runs every kind of lie through the whole dispute, shows FeeOracle
// refusing a claim about another window, applies an honest claim, and fails on any console error,
// failed expectation or sideways scroll. The whole story runs at desktop and at phone width.
// usage: node test/fee_ui.mjs [screenshot-dir]        (needs Chrome, or CHROME=/path/to/chrome)
import { mkdirSync } from 'node:fs';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const shots = process.argv[2] ?? null;
if (shots) mkdirSync(shots, { recursive: true });
const url = pathToFileURL(fileURLToPath(new URL('../dist/tensile-fee.html', import.meta.url))).href;
// Chrome as installed; PLAYWRIGHT_BUNDLED=1 uses Playwright's own Chromium (CI); CHROME=path a specific one.
const launch = process.env.CHROME
  ? { executablePath: process.env.CHROME }
  : process.env.PLAYWRIGHT_BUNDLED ? {} : { channel: 'chrome' };
const browser = await chromium.launch({ headless: true, ...launch });
const started = performance.now();

let failures = 0;
const expect = (ok, what) => {
  if (!ok) failures++;
  console.log(`${ok ? 'ok  ' : 'FAIL'}  ${what}`);
};

async function newPage(viewport) {
  const context = await browser.newContext({ viewport, offline: true });
  const page = await context.newPage();
  page.setDefaultTimeout(90000);
  page.on('console', (m) => {
    if (m.type() === 'error') expect(false, `console error: ${m.text()}`);
  });
  page.on('pageerror', (e) => expect(false, `page error: ${e.message}`));
  page.on('request', (r) => {
    if (!r.url().startsWith('file:') && !r.url().startsWith('data:')) expect(false, `network request: ${r.url()}`);
  });
  await page.goto(url);
  await page.waitForFunction(() => window.tensileFee?.ready === true);
  return page;
}

const text = (page, selector) => page.textContent(selector);
const waitText = (page, id, part) => page.waitForFunction(([i, p]) => document.getElementById(i).textContent.includes(p), [id, part]);
const state = (page, fn) => page.evaluate(fn);

async function noSideways(page, tag, what) {
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  expect(overflow <= 0, `${tag}: no sideways scroll on ${what} (${overflow}px)`);
}

async function shot(page, tag, name) {
  if (shots) await page.screenshot({ path: `${shots}/${tag}-${name}.png`, fullPage: true });
}

/** Waits until the EVM has run the network on the window the page shows. */
async function windowReady(page) {
  await page.waitForFunction(() => window.tensileFee.S.evm !== null && !document.getElementById('to-observe').disabled);
}

async function story(page, tag, kinds) {
  expect((await text(page, '#l-engine')).includes('5 Fe contracts'), `${tag}: the EVM starts and deploys from file://, offline`);
  await windowReady(page);
  await page.click('#w-day-fwd');
  await windowReady(page);
  expect((await state(page, () => window.tensileFee.S.i)) > 0, `${tag}: one day later is another window`);
  await page.click('[data-next-tier="2"]');
  await windowReady(page);
  const picked = await state(page, () => ({ i: window.tensileFee.S.i, label: window.tensileFee.S.w.label, cls: window.tensileFee.S.evm.cls, gas: String(window.tensileFee.S.evm.gas) }));
  expect(picked.label === 2, `${tag}: "next window where the real tier is 2" picks window ${picked.i}, real tier ${picked.label}`);
  const note = await text(page, '#window-note');
  expect(note.includes('gas') && note.includes('to the last bit'), `${tag}: Network.classify ran in the tab and the page's arithmetic agrees with Network.trace`);
  expect((await text(page, '#tiers')).includes(`tier ${picked.cls}`), `${tag}: the model's tier from the EVM is shown (${picked.cls})`);
  expect((await page.locator('#bytes span').count()) === 144, `${tag}: 144 input bytes are shown`);
  await noSideways(page, tag, 'the window');
  await shot(page, tag, '01-window-picked');

  // The contract builds the input from the ticks.
  await page.click('#to-observe');
  await page.click('#feed');
  await waitText(page, 'words-out', 'byte for byte');
  expect((await text(page, '#words-out')).includes("equals the page's 144 bytes"), `${tag}: inputWords(144) equals the page's bytes`);
  expect((await page.locator('#words .okk').count()) === 5, `${tag}: all five packed words are equal`);
  expect((await text(page, '#observe-out')).includes('144 moves stored'), `${tag}: 144 moves stored after 145 ticks`);
  await noSideways(page, tag, 'observe');
  await shot(page, tag, '02-observed');
  await page.click('#to-claim');
  await page.waitForSelector('#act-claim.on');

  // Every lie, slashed; and a slashed claim cannot set the fee.
  for (const kind of kinds) {
    await page.check(`input[name="lie"][value="${kind}"]`);
    await page.locator('#targets button:not([disabled])').first().click();
    await page.waitForFunction(() => !document.getElementById('post').disabled);
    if (kind === kinds[0]) {
      await noSideways(page, tag, 'the claim');
      await shot(page, tag, `03-claim-${kind}`);
    }
    await page.click('#post');
    await page.waitForSelector('#act-dispute.on');
    const finding = await text(page, '#finding');
    expect(finding.includes('wrong'), `${tag}: ${kind}: the watcher names a value: ${finding}`);
    if (kind === kinds[0]) {
      await noSideways(page, tag, 'the posted lie');
      await shot(page, tag, `04-lie-posted-${kind}`);
    }
    await page.click('#send-named');
    await page.waitForSelector('.verdict.show');
    expect((await text(page, '#verdict')) === 'SLASHED', `${tag}: ${kind}: stamped SLASHED`);
    expect((await text(page, '#l-claim')).includes('slashed'), `${tag}: ${kind}: the ledger shows slashed`);
    if (kind === kinds[0]) {
      await page.waitForTimeout(500);
      await noSideways(page, tag, 'the slash');
      await shot(page, tag, `05-slashed-${kind}`);
    }
    await page.click('#to-apply');
    await page.click('#apply');
    await page.waitForFunction(() => document.getElementById('apply-out').textContent.includes('refused'));
    expect((await text(page, '#fee-value')).trim() === '3,000', `${tag}: ${kind}: a slashed claim leaves fee() at the default`);
    await page.click('#again');
    await page.waitForSelector('#act-claim.on');
  }

  // Right arithmetic about another window: nothing to slash, and FeeOracle refuses it.
  await page.check('input[name="lie"][value="input"]');
  await page.locator('#targets button:not([disabled])').first().click();
  await page.waitForFunction(() => !document.getElementById('post').disabled);
  await page.click('#post');
  await page.waitForSelector('#act-dispute.on');
  expect((await text(page, '#finding')).includes('not the oracle'), `${tag}: another window's claim: the watcher finds the arithmetic right and the input foreign`);
  expect(await page.isDisabled('#send-named'), `${tag}: another window's claim: there is no challenge to send`);
  await page.click('#try-class');
  await page.waitForSelector('.verdict.stands.show');
  expect((await text(page, '#dispute-note')).includes('NoFraud'), `${tag}: another window's claim: a class challenge gets NoFraud`);
  await page.click('#to-apply');
  await page.click('#forward');
  await waitText(page, 'apply-out', 'observed the next 6 real ticks');
  expect((await text(page, '#apply-out')).includes('60 minutes'), `${tag}: six observations are one hour of chain time`);
  await page.click('#finalize');
  await waitText(page, 'apply-out', 'final');
  await page.click('#apply');
  await waitText(page, 'apply-out', 'NotThisWindow');
  expect((await text(page, '#fee-value')).trim() === '3,000', `${tag}: the refused claim leaves fee() at the default`);
  await page.waitForTimeout(350);
  await noSideways(page, tag, 'the refusal');
  await shot(page, tag, '06-mismatch-refused');

  // The honest claim: challenges refused, finalize too early, then final and applied.
  await page.click('#again');
  await page.check('input[name="lie"][value="honest"]');
  await page.waitForFunction(() => !document.getElementById('post').disabled);
  await page.click('#post');
  await page.waitForSelector('#act-dispute.on');
  expect((await text(page, '#finding')).includes('every value matches'), `${tag}: honest claim: the watcher finds nothing`);
  await page.selectOption('#pick-layer', '2');
  await page.fill('#pick-j', '3');
  await page.click('#try-neuron');
  await page.waitForSelector('.verdict.stands.show');
  await waitText(page, 'dispute-note', 'NoFraud');
  expect(true, `${tag}: honest claim: a layer 2 challenge gets NoFraud`);
  await page.click('#to-apply');
  await page.click('#finalize');
  await waitText(page, 'apply-out', 'TooEarly');
  expect(true, `${tag}: finalize inside the window gets TooEarly`);
  await page.click('#forward');
  await waitText(page, 'apply-out', 'observed the next');
  await page.click('#finalize');
  await waitText(page, 'apply-out', 'final');
  await page.click('#apply');
  await waitText(page, 'apply-out', 'set the fee');
  const want = await state(page, () => window.tensileFee.S.current.claim.cls);
  const fees = await state(page, () => window.TENSILE_FEE_ASSETS.fees.by_tier);
  const shown = (await text(page, '#fee-value')).trim();
  expect(shown === fees[want].toLocaleString('en-US'), `${tag}: fee() after apply is ${shown}, the fee of tier ${want}`);
  expect(want === picked.cls, `${tag}: the applied tier is the model's tier on the EVM (${want})`);
  expect((await text(page, '#route-note')).includes('gas'), `${tag}: the claim route's gas is tallied from the receipts`);
  await noSideways(page, tag, 'the applied fee');
  await shot(page, tag, '07-fee-applied');
}

const desktop = await newPage({ width: 1280, height: 800 });
await story(desktop, 'desktop', ['class', 'layer1', 'layer2', 'layer3']);
const phone = await newPage({ width: 390, height: 844 });
await story(phone, 'phone', ['layer1']);
for (const act of ['window', 'observe', 'claim', 'dispute', 'apply']) {
  await phone.click(`#steps button[data-act="${act}"]`);
  await noSideways(phone, 'phone', `step ${act}`);
}

await browser.close();
console.log(`fee ui test: ${failures} failed, ${((performance.now() - started) / 1000).toFixed(1)} s`);
process.exit(failures === 0 ? 0 : 1);
