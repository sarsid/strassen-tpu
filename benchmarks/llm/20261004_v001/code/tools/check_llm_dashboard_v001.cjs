// Browser verification of the dashboard against saved evidence; no experiment execution.
const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),cp=require('child_process'),assert=require('assert');
(async()=>{
 const out=path.join(process.env.STRASSEN_EXECUTION_DIR,'artifacts');fs.mkdirSync(out);
 const root=process.env.STRASSEN_PROJECT_ROOT;
 const run=path.join(root,'results/v6e/llm_campaign_20261003_v001/pilot_20261003_v002');
 const log=fs.openSync(path.join(out,'server.log'),'wx');
 const server=cp.spawn(path.join(path.dirname(root),'.venv-colab/bin/python'),['status/server_llm_v003.py','--run',run,'--port','8790'],{stdio:['ignore',log,log]});
 let browser;
 try{
  const base='http://127.0.0.1:8790';let state;
  for(let i=0;i<60;i++){try{let r=await fetch(base+'/state');if(r.ok){state=await r.json();break;}}catch{}await new Promise(r=>setTimeout(r,100));}
  assert(state);assert.equal(state.main.measured,0);assert.equal(state.main.expected,700);assert.equal(state.kernels.passed,54);assert.equal(state.semantics.passed,36);
  browser=await chromium.launch({channel:'chrome',headless:true});
  const page=await browser.newPage({viewport:{width:1440,height:1100},deviceScaleFactor:1});let errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto(base);await page.getByRole('heading',{name:'First look at performance'}).waitFor();
  assert.equal(await page.locator('.chart-row').count(),5);assert.equal(await page.locator('.model-name').count(),7);
  assert((await page.locator('.chart-row.native_default .time').innerText()).includes('0.659'));
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.getByRole('button',{name:'FP32',exact:true}).click();assert((await page.locator('.chart-row.native_default .time').innerText()).includes('0.667'));
  await page.locator('#tiles summary').click();await page.locator('#errors summary').click();
  assert(await page.locator('#tiles').getAttribute('open')!==null);
  await page.screenshot({path:path.join(out,'details.png'),fullPage:true});
  for(const link of ['/evidence/report.md','/evidence/pilot.json','/evidence/profiles.json','/evidence/protocol.md'])assert.equal((await fetch(base+link)).status,200,link);
  assert.equal((await fetch(base+'/evidence/../../etc/passwd')).status,404);
  await page.setViewportSize({width:390,height:844});await page.getByRole('button',{name:'BF16',exact:true}).click();
  await page.locator('#tiles summary').click();await page.locator('#errors summary').click();
  const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert(!overflow,'Mobile page overflows horizontally');
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  assert.equal(errors.length,0,errors.join('\n'));
  const result={passed:true,checks:['saved evidence counters','five algorithms and seven models','BF16/FP32 toggle','expandable errors and selected tiles','evidence links','file route allowlist','390px layout without page overflow','no browser errors'],screenshots:['desktop.png','details.png','mobile.png'],source_scope:'dashboard only; no TPU allocation'};
  fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
 }finally{if(browser)await browser.close();server.kill();fs.closeSync(log);}
})().catch(e=>{console.error(e);process.exitCode=1});
