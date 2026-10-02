import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const input=process.argv[2];
const run=process.env.STRASSEN_EXECUTION_DIR;
if(!input||!run)throw new Error('Use scoped archive execution with a results.json argument');
const out=path.join(run,'artifacts');await fs.mkdir(out,{recursive:false});
const data=JSON.parse(await fs.readFile(input,'utf8'));
assert.equal(data.shapes.length,140);assert.equal(data.algorithms.length,700);assert.equal(data.pairwise.length,560);
const methods=['native_default','native','cubic','one_level','two_level'];
const labels=data.summary.labels;
const wb=Workbook.create();
const results=wb.worksheets.add('Results');
const algorithms=wb.worksheets.add('Algorithm detail');
const comparisons=wb.worksheets.add('S2 comparisons');
const start=10,end=149;

function table(sheet,headers,values,tableName,widths){
  const last=start+values.length-1,n=headers.length;
  sheet.showGridLines=false;
  sheet.getRangeByIndexes(0,0,last,n).format.font={name:'Arial',size:10,color:'#243243'};
  sheet.getRangeByIndexes(0,0,last,n).format.verticalAlignment='center';
  sheet.getRangeByIndexes(8,0,values.length+1,n).values=[headers,...values];
  let column='',number=n;while(number>0){number--;column=String.fromCharCode(65+number%26)+column;number=Math.floor(number/26);}
  const t=sheet.tables.add('A9:'+column+last,true,tableName);
  t.style='TableStyleMedium2';t.showFilterButton=true;
  sheet.getRangeByIndexes(8,0,1,n).format={fill:'#294563',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:43,horizontalAlignment:'center',verticalAlignment:'center'};
  sheet.getRangeByIndexes(9,0,values.length,n).format.rowHeight=21;
  widths.forEach((width,i)=>sheet.getRangeByIndexes(0,i,last,1).format.columnWidth=width);
  sheet.getRange('A2').format.font={name:'Arial',size:15,bold:true,color:'#243243'};
  sheet.getRangeByIndexes(1,0,1,n).format.borders={bottom:{style:'thin',color:'#CBD5DF'}};
  sheet.getRangeByIndexes(2,0,4,n).format.font={name:'Arial',size:10,italic:true,color:'#59697A'};
  sheet.freezePanes.freezeRows(9);sheet.freezePanes.freezeColumns(4);
}

const headers=['Order','M','K','N','Classical work (GFLOP)','Geometry','Native default (ms)','Native tuned (ms)','Cubic (ms)','Strassen 1 (ms)','Strassen 2 (ms)','Lowest mean','Native tuned / S2','95% CI low','95% CI high','S2 vs tuned Native','Native default / S2','Strassen 1 / S2','Cubic / S2','S1 max relative L2','S2 max relative L2','All five eligible','Sampling group','Shape ID'];
const resultValues=data.shapes.map(r=>[r.order,r.m,r.k,r.n,r.classical_gflop,r.geometry,...methods.map(m=>r.methods[m].mean_ms),labels[r.fastest_mean_method],
  r.comparisons.native.speedup,...r.comparisons.native.ci95,r.comparisons.native.result,r.comparisons.native_default.speedup,
  r.comparisons.one_level.speedup,r.comparisons.cubic.speedup,r.methods.one_level.max_relative_l2,r.methods.two_level.max_relative_l2,
  r.all_five_eligible?'Yes':'No',r.sampling_group,r.shape_id]);
table(results,headers,resultValues,'First140Results',[8,10,10,10,16,14,16,16,15,15,15,19,16,13,13,20,17,16,14,17,17,13,43,42]);
results.tabColor='#294563';
results.getRange('A2').values=[['First 140 confirmed shapes']];
results.getRange('A3').values=[['A[M,K] × B[K,N]. Times include device padding and crop. Lower milliseconds is better.']];
results.getRange('A4').values=[['Each configuration was selected in screening and measured on 3 fresh inputs × 30 rounds. Speedup > 1 favors S2.']];
results.getRange('A5').values=[['These are the first 140 of 168 shapes, ordered by volume. Synthetic Gaussian inputs; actual-model results are excluded.']];
results.getRange('A6').values=[['Lowest mean is descriptive. Confidence intervals are pointwise 95%, with no correction across shapes.']];
results.getRange('A7').values=[['S2 vs tuned Native:']];
results.getRange('D7').formulas=[['=COUNTIFS(P10:P149,"S2 faster")']];results.getRange('E7').values=[['CI wins']];
results.getRange('G7').formulas=[['=COUNTIFS(P10:P149,"S2 slower")']];results.getRange('H7').values=[['CI losses']];
results.getRange('J7').formulas=[['=COUNTIFS(P10:P149,"Inconclusive")']];results.getRange('K7').values=[['Inconclusive']];
results.getRange('M7').formulas=[['=GEOMEAN(M10:M149)']];results.getRange('N7').values=[['Geomean speedup']];
results.getRange('M7').setNumberFormat('0.000"x"');
results.getRange('A10:E149').setNumberFormat('#,##0');results.getRange('E10:E149').setNumberFormat('#,##0.000');
results.getRange('G10:K149').setNumberFormat('0.0000');
results.getRange('M10:O149').setNumberFormat('0.000"x"');results.getRange('Q10:S149').setNumberFormat('0.000"x"');
results.getRange('T10:U149').setNumberFormat('0.0000%');
for(const [col,base] of [['M','H'],['Q','G'],['R','J'],['S','I']]){
  results.getRange(col+'10').formulas=[['='+base+'10/K10']];results.getRange(col+'10:'+col+'149').fillDown();
}
results.getRange('P10').formulas=[['=IF(N10>1,"S2 faster",IF(O10<1,"S2 slower","Inconclusive"))']];results.getRange('P10:P149').fillDown();
for(const area of ['M10:M149','Q10:S149']){
  results.getRange(area).conditionalFormats.add('cellIs',{operator:'greaterThan',formula:1,format:{fill:'#E6F1E9',font:{color:'#265B3B'}}});
  results.getRange(area).conditionalFormats.add('cellIs',{operator:'lessThan',formula:1,format:{fill:'#F8EAE5',font:{color:'#8C3E2B'}}});
}

const detailHeaders=['Order','M','K','N','Algorithm','Mean time (ms)','Candidate ID','Tile BM','Tile BN','Tile BK','Padded volume / useful','Max relative L2','Max normwise error','Max absolute error','Max RMSE','Max p99 absolute error','Reference scope','Reference outputs / input','Fresh inputs','Timing samples','Eligible','Confirmed alternatives','CI overlaps winner','CI faster than winner','', 'Source phase'];
const detailValues=data.algorithms.map(r=>[r.order,r.m,r.k,r.n,r.label,r.mean_ms,r.candidate_id,...(r.tile||[null,null,null]),
 r.padded_volume_ratio,r.max_relative_l2,r.max_normwise_error,r.max_max_abs_error,r.max_rmse,r.max_p99_abs_error,r.reference_scope,
 r.reference_sample_count,r.fresh_inputs,r.samples,r.eligible?'Yes':'No',r.confirmed_alternatives,r.indistinguishable_alternatives,r.faster_alternatives,null,r.phase]);
table(algorithms,detailHeaders,detailValues,'AlgorithmDetails',[8,10,10,10,18,16,40,11,11,11,19,17,20,20,19,22,29,23,13,14,12,19,20,21,3,23]);
algorithms.getRange('A2').values=[['Selected configurations and accuracy']];
algorithms.getRange('A3').values=[['700 algorithm labels. Native tuned and default can refer to the same measured candidate; they are not independent repeats.']];
algorithms.getRange('A4').values=[['Errors are worst across 3 inputs, compared with FP64 multiplication of exact BF16 inputs. Outputs are FP32.']];
algorithms.getRange('A5').values=[['Sampled references use all K but only selected output entries. Native tiles are compiler-managed, so BM/BN/BK are blank.']];
algorithms.getRange('A6').values=[['Tile alternatives cover the confirmation shortlist only; no global optimality or post-confirmation winner reselection.']];
algorithms.getRange('A7').values=[['Source: 20260921-mlsys-main-v5e-v001, sealed MAIN-01 through MAIN-10 confirmation artifacts.']];
algorithms.getRange('A10:D709').setNumberFormat('#,##0');algorithms.getRange('F10:F709').setNumberFormat('0.0000');
algorithms.getRange('H10:J709').setNumberFormat('#,##0');algorithms.getRange('K10:K709').setNumberFormat('0.000"x"');
algorithms.getRange('L10:P709').setNumberFormat('0.000E+00');algorithms.getRange('R10:T709').setNumberFormat('#,##0');
algorithms.getRange('V10:X709').setNumberFormat('0');algorithms.getRange('Y9:Y709').format.fill='#FFFFFF';
algorithms.getRange('L10:L709').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0.02,format:{fill:'#F8EAE5',font:{color:'#8C3E2B',bold:true}}});

const pairHeaders=['Order','M','K','N','Baseline','Baseline time (ms)','S2 time (ms)','Baseline / S2','95% CI low','95% CI high','Time saved by S2','Result','', 'Source phase'];
const pairValues=data.pairwise.map(r=>[r.order,r.m,r.k,r.n,r.label,r.baseline_mean_ms,r.strassen2_mean_ms,r.speedup,
 ...r.ci95,1-r.strassen2_mean_ms/r.baseline_mean_ms,r.result,null,r.phase]);
table(comparisons,pairHeaders,pairValues,'S2Pairwise',[8,10,10,10,20,20,18,18,15,15,19,20,3,24]);
comparisons.getRange('A2').values=[['Strassen 2 against every baseline']];
comparisons.getRange('A3').values=[['Speedup = baseline time / S2 time. Time saved = 1 − S2 time / baseline time. Positive savings favor S2.']];
comparisons.getRange('A4').values=[['Pointwise paired 95% intervals: 4,000 hierarchical bootstrap replicates over 3 inputs and 30 paired rounds.']];
comparisons.getRange('A5').values=[['Tuned Native intervals are the recorded frozen statistics; the other pairs are exploratory calculations from the same samples.']];
comparisons.getRange('A6').values=[['An interval crossing 1 is inconclusive. Intervals do not include variation between allocations or simultaneous uncertainty over all shapes.']];
comparisons.getRange('A7').values=[['Source: the same sealed confirmation phases as Algorithm detail. Screen winners remain fixed for every comparison.']];
comparisons.getRange('A10:D569').setNumberFormat('#,##0');comparisons.getRange('F10:G569').setNumberFormat('0.0000');
comparisons.getRange('H10:J569').setNumberFormat('0.000"x"');comparisons.getRange('K10:K569').setNumberFormat('+0.00%;-0.00%;0.00%');
comparisons.getRange('H10').formulas=[['=F10/G10']];comparisons.getRange('H10:H569').fillDown();
comparisons.getRange('K10').formulas=[['=1-G10/F10']];comparisons.getRange('K10:K569').fillDown();
comparisons.getRange('L10').formulas=[['=IF(I10>1,"S2 faster",IF(J10<1,"S2 slower","Inconclusive"))']];comparisons.getRange('L10:L569').fillDown();
comparisons.getRange('M9:M569').format.fill='#FFFFFF';
comparisons.getRange('K10:K569').conditionalFormats.add('cellIs',{operator:'greaterThan',formula:0,format:{fill:'#E6F1E9',font:{color:'#265B3B'}}});
comparisons.getRange('K10:K569').conditionalFormats.add('cellIs',{operator:'lessThan',formula:0,format:{fill:'#F8EAE5',font:{color:'#8C3E2B'}}});
for(const [sheet,cols,last] of [[results,['A:E','G:K','M:O','Q:U'],149],[algorithms,['A:D','F:F','H:P','R:T','V:X'],709],[comparisons,['A:D','F:K'],569]]){
  for(const cs of cols){const [a,b]=cs.split(':');sheet.getRange(a+'10:'+b+last).format.horizontalAlignment='right';}
}

wb.recalculate();
assert.equal(results.getRange('D7').values[0][0],data.summary.comparisons.native.ci_wins);
assert.equal(results.getRange('G7').values[0][0],data.summary.comparisons.native.ci_losses);
assert.equal(results.getRange('J7').values[0][0],data.summary.comparisons.native.inconclusive);
assert(Math.abs(results.getRange('M7').values[0][0]-data.summary.comparisons.native.geomean_speedup)<1e-10);
for(let i=0;i<140;i++){
  const actual=results.getRange('M'+(start+i)+':S'+(start+i)).values[0],r=data.shapes[i];
  for(const [j,method] of [[0,'native'],[4,'native_default'],[5,'one_level'],[6,'cubic']])assert(Math.abs(actual[j]-r.comparisons[method].speedup)<1e-10);
  assert.equal(actual[3],r.comparisons.native.result);
}
const old=comparisons.getRange('G10').values[0][0];comparisons.getRange('G10').values=[[old*1.1]];wb.recalculate();
assert(Math.abs(comparisons.getRange('H10').values[0][0]-data.pairwise[0].speedup/1.1)<1e-10);
comparisons.getRange('G10').values=[[old]];wb.recalculate();
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:60},summary:'Final formula error scan',maxChars:2000});
await fs.writeFile(path.join(out,'formula-error-scan.ndjson'),errors.ndjson+'\n');console.log(errors.ndjson);
const views=[['Results','A1:L17','results'],['Results','M9:X17','results_right'],['Algorithm detail','A1:L15','details'],['Algorithm detail','M9:Z15','details_right'],['S2 comparisons','A1:N15','comparisons']];
for(const [name,range,file] of views){
 const png=await wb.render({sheetName:name,range,scale:1.3,format:'png'});await fs.writeFile(path.join(out,file+'.png'),new Uint8Array(await png.arrayBuffer()));
}
const inspection=await wb.inspect({kind:'table',range:'Results!A9:U12',include:'values,formulas',tableMaxRows:4,tableMaxCols:21,maxChars:3500});
await fs.writeFile(path.join(out,'inspection.ndjson'),inspection.ndjson+'\n');
const xlsx=await SpreadsheetFile.exportXlsx(wb);await xlsx.save(path.join(out,'first_140_shapes.xlsx'));
await fs.writeFile(path.join(out,'validation.json'),JSON.stringify({status:'passed',shapes:140,algorithm_rows:700,pairwise_rows:560,formula_checks:'All 140 speedup rows and summary reconciled; edited timing recalculated and restored',input,source_summary:data.summary},null,2)+'\n');
console.log(JSON.stringify({status:'exported',path:path.join(out,'first_140_shapes.xlsx')}));
