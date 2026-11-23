// Builds dist/tensile-fee.html: the fee-tier page, one self-contained file like dist/tensile-demo.html,
// with the EVM, the Fe bytecode of examples/amm-fee, the model, the tick series and the page inlined,
// so it opens from disk or any static host with nothing to install and no network.
// usage: npm run build:fee      (generated/fee-assets.json comes from fee_assets.sh)
import { build } from 'esbuild';
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';

const at = (path) => new URL(path, import.meta.url);
const result = await build({
  entryPoints: [at('src/fee-app.js').pathname.replace(/^\/([A-Za-z]:)/, '$1')],
  bundle: true,
  minify: true,
  format: 'iife',
  platform: 'browser',
  target: 'es2022',
  write: false,
  legalComments: 'none',
  logLevel: 'warning',
  metafile: true,
});
// `</script` must not appear inside an inline script, whatever the bundle or the data contains.
const inline = (text) => text.replace(/<\/(script)/gi, '<\\/$1');
const script = inline(result.outputFiles[0].text);
const assets = inline(JSON.stringify(JSON.parse(readFileSync(at('generated/fee-assets.json'), 'utf8'))));
// The digit page's stylesheet as it is, then this page's additions.
const style = readFileSync(at('src/style.css'), 'utf8') + readFileSync(at('src/fee.css'), 'utf8');
const page = readFileSync(at('src/fee.html'), 'utf8')
  .replace('/*STYLE*/', () => style)
  .replace('/*ASSETS*/null', () => assets)
  .replace('/*SCRIPT*/', () => script);
// What the bundle contains, from the bundler's own record, with each package's declared licence.
const packages = new Map();
for (const input of Object.keys(result.metafile.inputs)) {
  const match = input.match(/node_modules\/((?:@[^/]+\/)?[^/]+)\//g);
  if (!match) continue;
  const name = match[match.length - 1].replace('node_modules/', '').replace(/\/$/, '');
  if (!packages.has(name)) {
    const meta = JSON.parse(readFileSync(at(`node_modules/${name}/package.json`), 'utf8'));
    packages.set(name, `${name}@${meta.version} ${meta.license}`);
  }
}
const notice = [...packages.values()].sort();
mkdirSync(at('dist/'), { recursive: true });
const target = at('dist/tensile-fee.html');
writeFileSync(target, page.replace('<!doctype html>', `<!doctype html>\n<!-- Tensile fee-tier demo. Bundles, unmodified: ${notice.join(', ')}. -->`));
console.log(`dist/tensile-fee.html: ${statSync(target).size} bytes (bundle ${script.length}, assets ${assets.length})`);
