const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),cp=require('child_process'),assert=require('assert');
(async()=>{
 const out=path.join(process.env.STRASSEN_EXECUTION_DIR,'artifacts');fs.mkdirSync(out);
 const root=process.env.STRASSEN_PROJECT_ROOT;
 const log=fs.openSync(path.join(out,'server.log'),'wx');
 const server=cp.spawn(path.join(path.dirname(root),'.venv-colab/bin/python'),['status/server_llm_v009.py','--run',path.join(root,'results/v6e/llm_campaign_20261003_v001/pilot_20261003_v002'),'--port','8790'],{stdio:['ignore',log,log]});
 let browser;
 try{
  let state;for(let i=0;i<60;i++){try{const r=await fetch('http://127.0.0.1:8790/state');if(r.ok){state=await r.json();break;}}catch{}await new Promise(r=>setTimeout(r,100));}
  assert(state&&state.queue);assert.equal(state.main.expected,700);assert.equal(Object.values(state.queue.counts).reduce((a,b)=>a+b,0),70);
  browser=await chromium.launch({channel:'chrome',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1100}});let errors=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('http://127.0.0.1:8790');await page.getByRole('heading',{name:'Unattended queue',exact:true}).waitFor();assert.equal(await page.locator('.model-name').count(),7);
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth)),'mobile overflow');
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});assert.equal(errors.length,0,errors.join('\n'));
  const summary={status:'completed',checks:['700 planned comparisons','70 durable workload states','unattended queue visible','seven model rows','mobile without horizontal overflow','no browser errors'],queue:state.queue,measured:state.main.measured};
  fs.writeFileSync(path.join(out,'summary.json'),JSON.stringify(summary,null,2));console.log(JSON.stringify(summary));
 }finally{if(browser)await browser.close();server.kill();fs.closeSync(log);}
})().catch(e=>{console.error(e);process.exitCode=1});
