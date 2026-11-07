// Builds dist/tensile-demo.html: one self-contained file, with the EVM, the Fe bytecode, the model
// data and the page inlined, so it opens from disk or any static host with nothing to install and no
// network. usage: npm run build (generated/assets.json comes from tools/build_demo.sh).
import { build } from 'esbuild';
import { mkdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';

const at = (path) => new URL(path, import.meta.url);
const result = await build({
  entryPoints: [at('src/app.js').pathname.replace(/^\/([A-Za-z]:)/, '$1')],
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
const data = JSON.parse(readFileSync(at('generated/assets.json'), 'utf8'));
data.numbers = JSON.parse(readFileSync(at('generated/numbers.json'), 'utf8'));
const assets = inline(JSON.stringify(data));
const page = readFileSync(at('src/index.html'), 'utf8')
  .replace('/*STYLE*/', () => readFileSync(at('src/style.css'), 'utf8'))
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
writeFileSync(at('THIRD_PARTY.txt'), `Third-party packages bundled, unmodified, into dist/tensile-demo.html (their source is on npm):\n\n${notice.join('\n')}\n`);
mkdirSync(at('dist/'), { recursive: true });
const target = at('dist/tensile-demo.html');
writeFileSync(target, page.replace('<!doctype html>', `<!doctype html>\n<!-- Tensile demo. Bundles, unmodified: ${notice.join(', ')}. -->`));
console.log(`dist/tensile-demo.html: ${statSync(target).size} bytes (bundle ${script.length}, assets ${assets.length})`);
