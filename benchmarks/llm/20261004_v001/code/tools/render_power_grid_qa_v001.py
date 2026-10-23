"""Render an immutable chartbook for visual QA; never alter scientific output."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--figures', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pdftoppm', required=True)
    parser.add_argument('--pdfinfo', required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pages = out / 'pages'
    pages.mkdir()
    source = args.pdf.resolve()
    figures = json.loads(args.figures.read_text())
    info = subprocess.run([args.pdfinfo, str(source)], check=True, capture_output=True, text=True)
    (out / 'pdfinfo.txt').write_text(info.stdout)
    match = re.search(r'^Pages:\s+(\d+)$', info.stdout, re.MULTILINE)
    if not match or int(match.group(1)) != len(figures):
        raise ValueError('PDF page count does not match the figure inventory')
    completed = subprocess.run([args.pdftoppm, '-r', '65', '-scale-to', '1000', '-png',
                                str(source), str(pages / 'page')], check=True, capture_output=True, text=True)
    (out / 'renderer.log').write_text(completed.stdout + completed.stderr)
    rendered = sorted(pages.glob('page-*.png'), key=lambda p: int(p.stem.rsplit('-', 1)[-1]))
    if len(rendered) != len(figures):
        raise ValueError('Renderer page count mismatch')
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.load_default(size=16)
    sheets = []
    for offset in range(0, len(rendered), 16):
        batch = rendered[offset:offset + 16]
        cell_w, cell_h = 430, 510
        sheet = Image.new('RGB', (cell_w * 4, cell_h * 4), '#d9dee3')
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(batch):
            x, y = (index % 4) * cell_w, (index // 4) * cell_h
            with Image.open(path) as original:
                thumb = original.convert('RGB')
                thumb.thumbnail((cell_w - 16, cell_h - 45), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y + 30))
            page = offset + index + 1
            draw.text((x + 8, y + 6), f"Page {page}: {figures[page - 1]['category']}", fill='#152536', font=font)
        destination = out / f'contact_{offset // 16 + 1:02d}.png'
        sheet.save(destination)
        sheets.append({'file': destination.name, 'pages': list(range(offset + 1, offset + len(batch) + 1))})
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'purpose': 'Visual QA renders; not new performance measurements or scientific figures',
                'source_pdf': str(source), 'source_pdf_sha256': digest(source),
                'figure_inventory': str(args.figures.resolve()), 'figure_inventory_sha256': digest(args.figures),
                'page_count': len(rendered), 'contact_sheets': sheets,
                'manual_visual_review_required': True,
                'sha256': {str(p.relative_to(out)): digest(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    with (out / 'qa_manifest.json').open('x') as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'page_count': len(rendered), 'contact_sheet_count': len(sheets), 'output': str(out)}))


if __name__ == '__main__':
    main()
