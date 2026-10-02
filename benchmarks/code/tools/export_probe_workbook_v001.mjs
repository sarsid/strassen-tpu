import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const root = process.env.STRASSEN_PROJECT_ROOT || process.cwd();
const run = process.env.STRASSEN_EXECUTION_DIR;
if (!run) throw new Error('Run through archive_scoped_v001.py');
const output = path.join(run, 'artifacts');
await fs.mkdir(output, {recursive:true});
const currentSource = 'runs/20260920-region-grid-cohort-v001/analysis/confirmed_results.json';
const previousSource = 'reports/probe_comparison_v001/previous_normalized.json';
const load = async p => JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const digest = async file => {
  const hash=crypto.createHash('sha256');
  for await (const chunk of createReadStream(file)) hash.update(chunk);
  return hash.digest('hex');
};
const current = await load(currentSource);
const previous = await load(previousSource);
const manifest = JSON.parse(await fs.readFile(current.manifest_path,'utf8'));
const shapeMeta = new Map(manifest.shapes.map(s=>[s.id,s]));
const phase = await load('runs/20260920-region-grid-cohort-v001/GRID-confirm-finished.json');
assert.equal(phase.status,'completed');
const artifactDir = path.join(phase.run,'artifacts');
const seal = JSON.parse(await fs.readFile(path.join(artifactDir,'artifact_manifest.json'),'utf8'));
const consumed = ['results.jsonl','confirmation.json','selection_input.json','environment.json'];
const sourceHashes={};
for(const f of consumed){
  const sha=await digest(path.join(artifactDir,f));
  assert.equal(sha,seal.sha256[f],`Current ${f} seal mismatch`);
  sourceHashes[path.relative(root,path.join(artifactDir,f))]=sha;
}
const selected=JSON.parse(await fs.readFile(path.join(artifactDir,'selection_input.json'),'utf8'));
const environment=JSON.parse(await fs.readFile(path.join(artifactDir,'environment.json'),'utf8'));
assert.equal(environment.identity.allocation_id,current.allocation_id);
const observed=new Map();
for await(const line of readline.createInterface({input:createReadStream(path.join(artifactDir,'results.jsonl')),crlfDelay:Infinity})){
  if(!line)continue;const r=JSON.parse(line);
  if(r.event==='case_result' && r.group_id?.endsWith('__headline_confirmation')){
    const key=[r.shape_id,r.candidate_id,r.scope].join('|');
    assert(!observed.has(key),`Duplicate ${key}`);observed.set(key,r);
  }
}
const lattice=n=>n>0 && ((n&(n-1))===0 || (n%3===0 && ((n/3)&(n/3-1))===0));
const currentRows=current.complete_call_rows.map(a=>{
  const r={...a,probe:'Current',source_file:currentSource,on_lattice:shapeMeta.get(a.shape_id).on_lattice};
  r.sampling_stratum=shapeMeta.get(a.shape_id).sampling_stratum;
  assert.equal(r.on_lattice,[r.m,r.n,r.k].every(lattice));
  for(const family of ['strassen','native','cubic','native_default']){
    const cid=r[family+'_candidate'];
    if(family!=='native_default')assert.equal(cid,selected.by_shape[r.shape_id][family].winner.candidate_id);
    for(const scope of ['call','prepared_kernel']){
      const evidence=observed.get([r.shape_id,cid,scope].join('|'));
      assert(evidence?.status==='ok' && evidence.correctness?.pass,`Missing eligible ${r.shape_id}/${cid}/${scope}`);
      if(scope==='call')assert.equal(evidence.timing.mean_ms,r[family+'_ms']);
      else r[family+'_prepared_ms']=evidence.timing.mean_ms;
    }
  }
  assert(Math.abs(r.native_ms/r.strassen_ms-r.strassen_vs_native)<1e-12);
  return r;
});
const previousRows=(previous.rows||previous.complete_call_rows).map(a=>({...a,probe:'Previous'}));
assert.equal(currentRows.length,120);assert.equal(previousRows.length,60);
const sort=(a,b)=>a.m-b.m||a.n-b.n||a.k-b.k||a.probe.localeCompare(b.probe);
currentRows.sort(sort);previousRows.sort(sort);
for(const rows of [currentRows,previousRows]){
  assert.equal(new Set(rows.map(r=>`${r.m},${r.n},${r.k}`)).size,rows.length);
  for(const r of rows){
    assert.equal(r.on_lattice,[r.m,r.n,r.k].every(lattice));
    for(const f of ['strassen','native','cubic','native_default'])assert(Number.isFinite(r[f+'_ms'])&&r[f+'_ms']>0,'Complete source timings are required');
    assert(Number.isFinite(r.native_ci_low)&&Number.isFinite(r.native_ci_high),'Complete confidence intervals are required');
    assert.equal(r.native_classification,r.native_ci_low>1?'win':r.native_ci_high<1?'loss':'inconclusive');
  }
}
assert.notEqual(currentRows[0].allocation_id,previousRows[0].allocation_id);
const joined=[...currentRows,...previousRows].sort(sort);
const overlap=currentRows.filter(r=>previousRows.some(p=>p.m===r.m&&p.n===r.n&&p.k===r.k));
assert.equal(overlap.length,12);
const headers=['M','N','K','Probe','Strassen (ms)','Native tuned (ms)','Cubic (ms)','Native default (ms)',
  'Time saved vs native (%)','Time saved vs cubic (%)','Time saved vs default (%)','Native / Strassen (x)',
  '95% CI low (x)','95% CI high (x)','Result vs native','Grid membership','Sampling group',
  'Strassen configuration','Native configuration','Cubic configuration','Strassen prepared (ms)',
  'Native prepared (ms)','Cubic prepared (ms)','Default prepared (ms)','Allocation','Shape ID'];
const resultNames={win:'Strassen wins',loss:'Native wins',inconclusive:'Inconclusive'};
const fraction=(r,b)=>r[b+'_ms']==null?null:1-r.strassen_ms/r[b+'_ms'];
function values(r){return [r.m,r.n,r.k,r.probe,r.strassen_ms,r.native_ms,r.cubic_ms,r.native_default_ms,
  fraction(r,'native'),fraction(r,'cubic'),fraction(r,'native_default'),r.native_ms/r.strassen_ms,
  r.native_ci_low,r.native_ci_high,resultNames[r.native_classification],r.on_lattice?'Target grid':'Off-grid diagnostic',
  r.sampling_stratum||r.sampling_role,r.strassen_candidate,r.native_candidate,r.cubic_candidate,
  r.strassen_prepared_ms??null,r.native_prepared_ms??null,r.cubic_prepared_ms??null,r.native_default_prepared_ms??null,
  r.allocation_id,r.shape_id];}
const csvEscape=v=>v==null?'':String(v).match(/[",\r\n]/)?'"'+String(v).replaceAll('"','""')+'"':String(v);
for(const [name,rows] of [['current_probe',currentRows],['previous_probe',previousRows],['joined_by_shape',joined]]){
  const csvRows=rows.map(r=>{const v=values(r);for(const i of [8,9,10])if(v[i]!=null)v[i]*=100;return v;});
  await fs.writeFile(path.join(output,name+'.csv'),[headers,...csvRows].map(row=>row.map(csvEscape).join(',')).join('\r\n')+'\r\n',{flag:'wx'});
}
await fs.writeFile(path.join(output,'normalized_results.json'),JSON.stringify({current:currentRows,previous:previousRows},null,2)+'\n',{flag:'wx'});
const workbook=Workbook.create();
const sheetSpecs=[['Current probe',currentRows,'CurrentProbe'],['Previous probe',previousRows,'PreviousProbe'],['Joined by shape',joined,'JoinedByShape']];
const sheets=new Map(sheetSpecs.map(([name])=>[name,workbook.worksheets.add(name)]));
const rowMaps={Current:new Map(),Previous:new Map()};
const start=9;
for(const [name,rows,tableName] of sheetSpecs){
  const sheet=sheets.get(name),end=start+rows.length-1;
  sheet.showGridLines=false;sheet.tabColor=name==='Joined by shape'?'#233F64':'#6C86A7';
  sheet.getRange(`A1:Z${end}`).format.font={name:'Arial',size:10,color:'#20262E'};
  sheet.getRange(`A1:Z${end}`).format.verticalAlignment='center';
  sheet.getRange('A2').values=[[name+' — confirmation timings']];sheet.getRange('A2').format.font={size:15,bold:true};
  sheet.getRange('A3').values=[['Time saved = (baseline − Strassen) / baseline. Positive is faster; negative is slower.']];
  sheet.getRange('A4').values=[['Primary: device padding + matmul + crop; host transfer and compilation excluded. Prepared timings exclude padding and crop.']];
  sheet.getRange('A5').values=[['Configurations were selected during screening, then timed in confirmation. CI: pointwise paired 95%; no multiplicity adjustment.']];
  const counts=rows.reduce((c,r)=>(c[r.on_lattice?'grid':'offgrid']++,c),{grid:0,offgrid:0});
  sheet.getRange('A6').values=[[`${rows.length} shape/probe rows: ${counts.grid} on grid, ${counts.offgrid} off grid. Grid dimensions are 2^i or 3·2^i. Joined rows retain separate allocations.`]];
  const source=name==='Current probe'?currentSource:name==='Previous probe'?'Previous probe: sealed GRID-confirm-v5e-v004-00a15b headline confirmation results':'Sources: Current probe and Previous probe sheets. Sorted numerically by M, N, K, then probe; no averaging.';
  sheet.getRange('A7').values=[['Source: '+source]];
  sheet.getRange('A3:Z7').format.font={italic:true,size:10,color:'#526171'};
  sheet.getRange('A2:Z2').format.borders={bottom:{style:'thin',color:'#BAC6D3'}};
  sheet.getRange(`A8:Z${end}`).values=[headers,...rows.map(values)];
  if(name!=='Joined by shape'){
    rows.forEach((r,i)=>rowMaps[r.probe].set(r.shape_id,start+i));
    for(const [col,baseline] of [['I','F'],['J','G'],['K','H']]){
      sheet.getRange(`${col}${start}`).formulas=[[`=IF(${baseline}${start}="","",1-E${start}/${baseline}${start})`]];
      sheet.getRange(`${col}${start}:${col}${end}`).fillDown();
    }
    sheet.getRange(`L${start}`).formulas=[[`=F${start}/E${start}`]];
    sheet.getRange(`L${start}:L${end}`).fillDown();
  }else{
    const linked=rows.map(r=>Array.from({length:26},(_,c)=>{
      const col=String.fromCharCode(65+c),tab=r.probe+' probe',sr=rowMaps[r.probe].get(r.shape_id);
      return `=IF('${tab}'!${col}${sr}="","",'${tab}'!${col}${sr})`;
    }));
    sheet.getRange(`A${start}:Z${end}`).formulas=linked;
  }
  const table=sheet.tables.add(`A8:Z${end}`,true,tableName);table.style='TableStyleMedium2';table.showFilterButton=true;
  sheet.getRange('A8:Z8').format={fill:'#233F64',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:45,horizontalAlignment:'center',verticalAlignment:'center'};
  sheet.getRange(`A${start}:Z${end}`).format.rowHeight=21;
  sheet.getRange(`A${start}:C${end}`).setNumberFormat('#,##0');
  sheet.getRange(`E${start}:H${end}`).setNumberFormat('0.0000');
  sheet.getRange(`I${start}:K${end}`).setNumberFormat('+0.00%;-0.00%;0.00%');
  sheet.getRange(`L${start}:N${end}`).setNumberFormat('0.0000');
  sheet.getRange(`U${start}:X${end}`).setNumberFormat('0.0000');
  for(const area of [`A${start}:C${end}`,`E${start}:N${end}`,`U${start}:X${end}`])sheet.getRange(area).format.horizontalAlignment='right';
  for(const col of ['D','O','Y'])sheet.getRange(`${col}${start}:${col}${end}`).format.horizontalAlignment='center';
  const widths=[10,10,10,12,15,15,15,15,17,17,17,15,13,13,17,22,28,30,24,30,17,17,17,17,54,40];
  widths.forEach((width,i)=>sheet.getRangeByIndexes(0,i,end,1).format.columnWidth=width);
  sheet.getRange(`I${start}:K${end}`).conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#E4F1E8',font:{color:'#205B38'}}});
  sheet.getRange(`I${start}:K${end}`).conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{fill:'#F8E9E4',font:{color:'#8A3524'}}});
  sheet.freezePanes.freezeRows(8);sheet.freezePanes.freezeColumns(4);
}
workbook.recalculate();
for(const [name,rows] of sheetSpecs){
  const sheet=sheets.get(name);
  const numeric=sheet.getRange(`E${start}:N${start+rows.length-1}`).values;
  rows.forEach((r,i)=>{
    const expected=values(r).slice(4,14);
    expected.forEach((v,j)=>{
      if(v==null)assert(numeric[i][j]==null||numeric[i][j]==='');
      else assert(Math.abs(numeric[i][j]-v)<1e-10,`${name} row ${i} column ${j}`);
    });
  });
}
const currentSheet=sheets.get('Current probe');
const testRow=rowMaps.Current.get(currentRows.find(r=>r.m===8192&&r.n===8192&&r.k===8192).shape_id);
const original=currentSheet.getRange(`E${testRow}`).values[0][0];
currentSheet.getRange(`E${testRow}`).values=[[original*1.1]];workbook.recalculate();
const changed=currentSheet.getRange(`I${testRow}`).values[0][0];
assert(Math.abs(changed-(1-original*1.1/currentSheet.getRange(`F${testRow}`).values[0][0]))<1e-12);
const jRow=start+joined.findIndex(r=>r.probe==='Current'&&r.m===8192&&r.n===8192&&r.k===8192);
assert.equal(sheets.get('Joined by shape').getRange(`I${jRow}`).values[0][0],changed);
currentSheet.getRange(`E${testRow}`).values=[[original]];workbook.recalculate();
const errors=await workbook.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:100},summary:'Final formula error scan',maxChars:6000});
await fs.writeFile(path.join(output,'formula-error-scan.ndjson'),errors.ndjson+'\n');
console.log('ERROR_SCAN',errors.ndjson);
for(const [name] of sheetSpecs){
  const check=await workbook.inspect({kind:'table',range:`'${name}'!A8:Q12`,include:'values,formulas',tableMaxRows:5,tableMaxCols:17,maxChars:5000});
  await fs.writeFile(path.join(output,name.replaceAll(' ','_')+'-inspection.ndjson'),check.ndjson+'\n');
  for(const [label,range] of [['main','A1:Q16'],['details','R8:Z14']]){
    const preview=await workbook.render({sheetName:name,range,scale:1.5,format:'png'});
    await fs.writeFile(path.join(output,name.replaceAll(' ','_')+'-'+label+'.png'),new Uint8Array(await preview.arrayBuffer()));
  }
}
const xlsx=await SpreadsheetFile.exportXlsx(workbook);await xlsx.save(path.join(output,'strassen_probe_comparison.xlsx'));
sourceHashes[currentSource]=await digest(path.join(root,currentSource));sourceHashes[previousSource]=await digest(path.join(root,previousSource));
const validation={status:'passed',current_rows:120,previous_rows:60,joined_rows:180,unique_shapes:new Set(joined.map(r=>`${r.m},${r.n},${r.k}`)).size,overlapping_shapes:overlap.length,all_numeric_cells_checked:true,recalculation_and_join_update_checked:true,source_hashes:sourceHashes,allocations:[currentRows[0].allocation_id,previousRows[0].allocation_id],pooling:false};
await fs.writeFile(path.join(output,'validation.json'),JSON.stringify(validation,null,2)+'\n',{flag:'wx'});
const readme=`# Strassen probe comparison\n\nThree tables are supplied as Excel sheets and CSV files: current (${currentRows.length} rows), previous (${previousRows.length}), and joined (${joined.length}). Rows are sorted numerically by M, N, K; repeated shapes retain separate probe rows. There are ${overlap.length} repeated shapes and ${validation.unique_shapes} distinct shapes. No runtime is averaged across machines.\n\nMatrices are A[M,K] × B[K,N]. Primary times are arithmetic means of the confirmation complete-call measurements, in milliseconds, including device-side padding, matrix multiplication and output cropping. Host transfer and compilation are excluded. Configurations were selected by screening and frozen before confirmation. Prepared-kernel timings are separate supporting columns and exclude preparation/cropping. They must not be substituted for complete-call latency.\n\nTime saved (%) = 100 × (baseline time − Strassen time) / baseline time. Positive means Strassen takes less time; negative means Strassen takes more time. Native / Strassen is a speedup ratio, which is a different metric. Percentage columns in CSV use percentage points (e.g. 6.0 means 6%); Excel uses numeric percentage cells. Comparisons against tuned native, cubic and native default are supplied.\n\nThe native baseline is the screening-selected XLA preset, and the default-native column is also retained. The per-algorithm time is for that algorithm's screen-selected configuration, not the minimum reselected from confirmation. All recorded current shapes, including 24 off-grid diagnostics, remain in the table. Filter Grid membership to Target grid for the user's 2^i or 3·2^i domain. Sampling groups remain labeled. Confidence intervals are pointwise paired 95% intervals, without multiplicity adjustment, and do not capture variability between allocations. Reserved holdouts were not measured.\n\nCurrent allocation: ${currentRows[0].allocation_id}\nPrevious allocation: ${previousRows[0].allocation_id}\n\nSources (relative to the project):\n- ${currentSource}\n- ${previousSource}\n- ${path.relative(root,artifactDir)}/results.jsonl\n- The previous normalization includes exact original sources and sealed provenance.\n\nNo original measurement files were modified. The workbook is self-contained; joined cells link to the two probe sheets. Filtering and sorting are available in each table. Figures are exported separately as PNG and SVG.\n`;
await fs.writeFile(path.join(output,'README.md'),readme,{flag:'wx'});
console.log(JSON.stringify(validation));
