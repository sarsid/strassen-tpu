"""Check saved report content and render every page for visual review."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from pypdf import PdfReader
import pdfplumber


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf',required=True,type=Path)
    parser.add_argument('--render-dir',required=True,type=Path)
    parser.add_argument('--pdftoppm',required=True)
    args=parser.parse_args()
    root=Path(os.environ['STRASSEN_PROJECT_ROOT'])
    output=Path(os.environ['STRASSEN_EXECUTION_DIR'])/'artifacts'
    output.mkdir(exist_ok=False)
    reader=PdfReader(str(args.pdf))
    assert len(reader.pages)==11
    pages=[p.extract_text() for p in reader.pages]
    text='\n'.join(pages)
    report=json.loads((root/'runs/20260921T025202Z-llm-mini-final-report-v001-639727/artifacts/report/comparisons.json').read_text())
    for r in report['complete_call_rows']:
        for f in ('native','cubic','strassen'):
            assert f'{r[f+"_call_ms"]:.3f}' in text
            assert f'{r[f+"_prepared_kernel_ms"]:.3f}' in text
        for f in ('native','cubic'):
            assert f'{r["strassen_vs_"+f]:.3f}x' in text
    for term in ('13 / 18','18 / 18','gate_up_concat','588','29,056','2026092103','128 sampled rows','two-level Strassen'):
        assert term in text, term
    issues=[]
    with pdfplumber.open(args.pdf) as pdf:
        for i,p in enumerate(pdf.pages,1):
            assert abs(p.width-612)<.1 and abs(p.height-792)<.1
            for char in p.chars:
                if char['x0'] < 40 or char['x1'] > 572 or char['top'] < 20 or char['bottom'] > 773:
                    issues.append({'page':i,'text':char['text'],'bbox':[char['x0'],char['top'],char['x1'],char['bottom']]})
    assert not issues, issues[:5]
    args.render_dir.mkdir(parents=True,exist_ok=False)
    cache=args.render_dir/'cache';cache.mkdir()
    config=args.render_dir/'fonts.conf'
    fonts='/Users/bagheera/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/fonts'
    config.write_text('<fontconfig><dir>'+fonts+'</dir><cachedir>'+str(cache)+'</cachedir></fontconfig>')
    subprocess.run([args.pdftoppm,'-r','120','-png',str(args.pdf),str(args.render_dir/'page')],check=True,
                   env=dict(os.environ,FONTCONFIG_FILE=str(config),XDG_CACHE_HOME=str(cache)))
    renders=sorted(args.render_dir.glob('page-*.png'));assert len(renders)==11
    result={'passed':True,'pages':11,'confirmed_rows_checked':18,'expected_numeric_values_present':True,
            'text_inside_page_bounds':True,'visual_inspection':'pending','pdf_sha256':hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
            'rendered_pages':[str(p) for p in renders]}
    with (output/'summary.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
