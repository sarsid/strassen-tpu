// Local browser checks; writes only to the scoped execution's artifacts.
const { chromium } = require('/Users/bagheera/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');

(async () => {
  const out = path.join(process.env.STRASSEN_EXECUTION_DIR, 'artifacts');
  fs.mkdirSync(out);
  const checks = [], errors = [];
  const browser = await chromium.launch({headless:true,
    executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  try {
    const page = await browser.newPage({viewport:{width:1280,height:1000},deviceScaleFactor:1});
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto('http://127.0.0.1:8766/', {waitUntil:'networkidle'});
    await page.waitForFunction(() => document.getElementById('connection-state').textContent.startsWith('Connected'));
    const response = await page.request.get('http://127.0.0.1:8766/api/progress');
    assert.equal(response.status(), 200);
    const data = await response.json();
    fs.writeFileSync(path.join(out,'api-snapshot.json'), JSON.stringify(data,null,2));
    assert.equal((await page.locator('#experiment-state').innerText()).toLowerCase(), 'not started');
    assert.equal(await page.locator('#models article').count(), 3);
    assert(data.models.every(m => m.state === 'not_started'));
    assert(data.gemma.passed < data.gemma.total);
    checks.push('Live evidence loads; new model experiment remains not started; failed Gemma gate visible');
    await page.locator('#auto').uncheck();
    assert((await page.locator('#connection-state').innerText()).includes('paused'));
    await page.locator('#refresh').click();
    await page.waitForFunction(() => !document.getElementById('refresh').disabled);
    checks.push('Pause and manual refresh work');
    await page.locator('#filter').selectOption('attention');
    assert(await page.locator('#events details').count() > 0);
    await page.locator('#expand').click();
    assert.equal(await page.locator('#events details[open]').count(), await page.locator('#events details').count());
    await page.locator('#collapse').click();
    assert.equal(await page.locator('#events details[open]').count(),0);
    await page.locator('#filter').selectOption('all');
    checks.push('Failure filter and expandable evidence log work');
    const link = await page.locator('#gemma-evidence a').first().getAttribute('href');
    assert.equal((await page.request.get('http://127.0.0.1:8766'+link)).status(),200);
    assert.equal((await page.request.get('http://127.0.0.1:8766/files/.runtime_private/secret.json')).status(),404);
    checks.push('Evidence accessible; private paths rejected');
    await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
    await page.setViewportSize({width:390,height:844});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth));
    await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
    checks.push('390px layout has no horizontal overflow');
    await page.setViewportSize({width:1280,height:1000});
    await page.route('**/api/progress', route => route.abort());
    await page.locator('#refresh').click();
    await page.waitForFunction(() => document.getElementById('connection-state').textContent.startsWith('Disconnected'));
    assert.equal(await page.locator('#models article').count(),3);
    await page.unroute('**/api/progress');
    checks.push('Disconnected state retains and labels the previous snapshot');
    const complete = JSON.parse(JSON.stringify(data));
    complete.grid.state='complete'; complete.grid.updated_utc='2020-01-01T00:00:00+00:00';
    await page.route('**/api/progress', route => route.fulfill({json:complete}));
    await page.locator('#refresh').click();
    await page.waitForFunction(() => document.getElementById('grid-state').textContent.toLowerCase()==='complete');
    await page.unroute('**/api/progress');
    checks.push('Confirmed completion is not mislabeled stale');
    await page.locator('#refresh').click();
    await page.waitForFunction(() => !document.getElementById('refresh').disabled);
    await page.locator('#theme').click();
    await page.locator('#theme').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'),'dark');
    await page.screenshot({path:path.join(out,'dark.png'),fullPage:true});
    assert.deepEqual(errors, []);
    checks.push('Dark theme works; no JavaScript runtime errors');
    fs.writeFileSync(path.join(out,'summary.json'), JSON.stringify({passed:true,checks,errors},null,2));
    console.log(JSON.stringify({passed:true,checks}));
  } catch(error) {
    fs.writeFileSync(path.join(out,'summary.json'), JSON.stringify({passed:false,checks,errors,error:String(error)},null,2));
    throw error;
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode=1; });
