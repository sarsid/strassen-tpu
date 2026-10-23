"""Check preserved comparison data and render every explained PDF page."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw
from pypdf import PdfReader


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    base=a.run.resolve()/'artifacts';old=a.source_run.resolve()/'artifacts'
    assert json.loads((a.run/'completion.json').read_text())['status']=='completed'
    seal=json.loads((base/'explained_artifact_manifest.json').read_text())['files']
    outer=json.loads((a.run/'artifact-manifest.json').read_text())
    for name,record in seal.items():
        path=base/name;actual={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        assert actual==record==outer['artifacts/'+name],name
    m=json.loads((base/'explained_manifest.json').read_text())
    original={c['id']:c for c in json.loads((old/'manifest.json').read_text())['charts'] if not c['detail']}
    assert len(m['charts'])==len(original)==47
    for c in m['charts']:
        for key in ['points','categories','series','source_run_ids','title','scope','cohort']:
            assert c[key]==original[c['id']][key],(c['id'],key)
        assert c['explanation']['observation'] and c['explanation']['interpretation']
        for pair in c['files'].values():
            for name in pair.values():assert (base/name).is_file()
    pdf=base/'N1_N9_explained_comparisons.pdf';reader=PdfReader(pdf);assert len(reader.pages)==49
    texts=[p.extract_text() for p in reader.pages]
    assert all('N'+str(i) in texts[0] for i in range(1,10)) and 'N6a' in texts[0]
    assert all('WHAT THIS GRAPH SHOWS' in s and 'OUR INTERPRETATION' in s for s in texts[2:])
    (out/'extracted_text.txt').write_text('\n\f\n'.join(texts))
    subprocess.run(['pdftoppm','-r','70','-png',str(pdf),str(out/'page')],check=True)
    pages=sorted(out.glob('page-*.png'));assert len(pages)==49
    for start in range(0,len(pages),9):
        sheet=Image.new('RGB',(1680,1440),'#DCE2E8');draw=ImageDraw.Draw(sheet)
        for i,path in enumerate(pages[start:start+9]):
            with Image.open(path) as im:
                im=im.convert('RGB');im.thumbnail((552,442))
                x=(i%3)*560;y=(i//3)*480
                sheet.paste(im,(x+(552-im.width)//2,y+22));draw.text((x+10,y+5),f'Page {start+i+1}',fill='#172433')
        sheet.save(out/f'contact-{start//9+1:02d}.png')
    summary={'run':a.run.name,'pdf_pages':49,'original_graphs_preserved':47,'all_values_and_sources_unchanged':True,
             'all_graphs_have_observation_and_interpretation':True,'sealed_files_checked':len(seal),'visual_review':'Required on rendered pages'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
