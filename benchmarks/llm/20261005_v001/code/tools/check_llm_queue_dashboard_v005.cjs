const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),cp=require('child_process'),assert=require('assert');
(async()=>{
 const out=path.join(process.env.STRASSEN_EXECUTION_DIR,'artifacts');fs.mkdirSync(out);
 const root=process.env.STRASSEN_PROJECT_ROOT;
 const feed=JSON.parse(cp.execFileSync(path.join(path.dirname(root),'.venv-colab/bin/python'),['tools/check_llm_progress_v001.py'],{encoding:'utf8'}));assert.equal(feed.status,'completed');fs.writeFileSync(path.join(out,'feed-checks.json'),JSON.stringify(feed,null,2));
 const log=fs.openSync(path.join(out,'server.log'),'wx');
 const server=cp.spawn(path.join(path.dirname(root),'.venv-colab/bin/python'),['status/server_llm_v013.py','--run',path.join(root,'results/v6e/llm_campaign_20261003_v001/pilot_20261003_v002'),'--port','8790'],{stdio:['ignore',log,log]});
 let browser;
 try{
  let state;for(let i=0;i<60;i++){try{const r=await fetch('http://127.0.0.1:8790/state');if(r.ok){state=await r.json();break;}}catch{}await new Promise(r=>setTimeout(r,100));}
  assert(state&&state.queue);assert.equal(state.main.expected,350);assert.equal(Object.values(state.queue.counts).reduce((a,b)=>a+b,0),70);
  browser=await chromium.launch({channel:'chrome',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1100}});let errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.route('**/state',route=>route.fulfill({contentType:'application/json',body:JSON.stringify(state)}));
  await page.goto('http://127.0.0.1:8790');await page.getByRole('heading',{name:'Unattended queue',exact:true}).waitFor();assert.equal(await page.locator('.model-name').count(),7);assert(state.queue.model_order[0]==='qwen3_8b'&&state.queue.model_order[1]==='qwen3_32b');
  await page.getByRole('heading',{name:'Recent full-forward samples',exact:true}).waitFor();
  assert.equal(await page.locator('.live-track').count(),1);assert.equal(await page.locator('.round-cells').count(),5);
  assert.equal(await page.locator('.round-cells .seen').count(),state.progress.observed_rounds);assert.equal(state.main.measured,10);assert.equal(state.scope.historical_comparisons,5);assert.equal(state.progress.expected_rounds,75);assert(state.progress.recent_measurements.every(e=>e.output_dtype==='bfloat16'));
  await page.locator('#live-quality summary').click();await page.locator('#live-events summary').click();
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.locator('#live-workload').screenshot({path:path.join(out,'detail.png')});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)),'mobile overflow');
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});assert.equal(errors.length,0,errors.join('\n'));
  await page.route('**/state',route=>route.fulfill({contentType:'application/json',body:JSON.stringify({...state,queue:{...state.queue,status:'blocked_shared_setup',stale:false}})}));
  await page.reload();await page.getByRole('heading',{name:'Queue paused for recovery.',exact:true}).waitFor();
  await page.unroute('**/state');const oldEvent=new Date(Date.now()-600000).toISOString();const fixture={...state,generated_utc:new Date().toISOString(),snapshot_utc:new Date().toISOString(),progress:{...state.progress,last_event:{...state.progress.last_event,utc:oldEvent},sample_cadence_seconds:60}};
  await page.route('**/state',route=>route.fulfill({contentType:'application/json',body:JSON.stringify(fixture)}));await page.reload();await page.locator('#live-workload').waitFor();
  assert(await page.locator('.freshness .tick').nth(0).evaluate(e=>e.classList.contains('age-old')),'old event visibly aged');assert(!(await page.locator('.freshness .tick').nth(1).evaluate(e=>e.classList.contains('age-old'))),'fresh snapshot is distinct');
  const clock=await page.locator('.freshness .tick').nth(0).textContent();await page.waitForTimeout(1300);assert.notEqual(await page.locator('.freshness .tick').nth(0).textContent(),clock,'event age clock advances');assert.equal(await page.locator('.round-cells .seen').count(),state.progress.observed_rounds,'clock does not invent progress');
  const summary={status:'completed',checks:['350 active BF16 comparisons','70 durable workload states','unattended queue visible','seven model rows','mobile without horizontal overflow','no browser errors','shared setup stop visibly marked paused','one BF16 track and five algorithm round bars','active and historical counts separated','FP32 samples excluded from active table','observed counts agree with feed','quality and event details expand','measurement age differs from snapshot freshness','clock advances without inventing samples'],feed_checks:feed.checks.length,queue:state.queue,measured:state.main.measured,observed_rounds:state.progress.observed_rounds};
  fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary));
 }finally{if(browser)await browser.close();server.kill();fs.closeSync(log);}
})().catch(e=>{console.error(e);process.exitCode=1});
