"""Validate sealed chart outputs and render every PDF page for visual review."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from PIL import Image, ImageDraw


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--chart-run',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args()
    run=args.chart_run.resolve(); source=run/'artifacts'
    out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=False)
    assert json.loads((run/'completion.json').read_text())['status']=='completed'
    seal=json.loads((source/'plot_manifest.json').read_text())
    outer=json.loads((run/'artifact-manifest.json').read_text())
    files=seal['files']
    for name,record in files.items():
        path=source/name
        actual={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        assert actual==record==outer['artifacts/'+name],name
    m=json.loads((source/'manifest.json').read_text())
    assert len(m['charts'])==seal['chart_count']
    assert len(set(c['id'] for c in m['charts']))==len(m['charts'])
    assert {c['experiment'] for c in m['charts']}=={'N'+str(i) for i in range(1,10)}
    assert all(not v['missing'] and v['case_results']==v['represented'] for v in m['coverage'].values())
    point_count=0
    for c in m['charts']:
        assert json.loads((source/c['data_file']).read_text())==c
        assert c['source_run_ids']
        for pt in c['points']:
            assert pt['row_refs'],(c['id'],'missing journal refs')
            assert all(ref.split(':')[0] in c['source_run_ids'] for ref in pt['row_refs'])
            point_count+=1
        for mode,paths in c['files'].items():
            for fmt,name in paths.items():
                assert name in files,(c['id'],mode,fmt)
                if fmt=='png':
                    with Image.open(source/name) as im: im.verify()
    pdf=source/'N1_N9_key_comparisons.pdf'
    info=subprocess.check_output(['pdfinfo',str(pdf)],text=True)
    pages=int(re.search(r'^Pages:\s+(\d+)',info,re.M).group(1))
    assert pages==m['pdf_pages']==sum(not c['detail'] for c in m['charts'])
    (out/'pdfinfo.txt').write_text(info)
    subprocess.run(['pdftotext',str(pdf),str(out/'pdf-text.txt')],check=True)
    subprocess.run(['pdftoppm','-r','55','-png',str(pdf),str(out/'pdf-page')],check=True)
    rendered=sorted(out.glob('pdf-page-*.png'))
    assert len(rendered)==pages
    def sheets(paths,prefix,labels):
        for start in range(0,len(paths),12):
            chunk=paths[start:start+12]
            canvas=Image.new('RGB',(1600,4*330),'#dce2e8')
            draw=ImageDraw.Draw(canvas)
            for i,path in enumerate(chunk):
                with Image.open(path) as im:
                    im=im.convert('RGB'); im.thumbnail((525,298))
                    x=(i%3)*533; y=(i//3)*330
                    canvas.paste(im,(x+(525-im.width)//2,y+22))
                    draw.text((x+8,y+5),labels[start+i],fill='#172433')
            canvas.save(out/(prefix+f'-{start//12+1:02d}.png'))
    sheets(rendered,'pdf-contact',[f'PDF page {i+1}' for i in range(pages)])
    figures=[]; labels=[]
    for c in m['charts']:
        for mode,paths in c['files'].items():
            figures.append(source/paths['png']); labels.append(c['id']+' '+mode)
    sheets(figures,'figure-contact',labels)
    summary={'chart_run':run.name,'sealed_files_verified':len(files),'logical_charts':len(m['charts']),
             'image_variants':len(figures),'pdf_pages':pages,'plotted_points':point_count,
             'charts_per_experiment':dict(Counter(c['experiment'] for c in m['charts'])),
             'case_results_represented':sum(v['represented'] for v in m['coverage'].values()),
             'coverage_complete':True,'note':'Automated structure/provenance checks passed; raster contact sheets require human visual inspection.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
