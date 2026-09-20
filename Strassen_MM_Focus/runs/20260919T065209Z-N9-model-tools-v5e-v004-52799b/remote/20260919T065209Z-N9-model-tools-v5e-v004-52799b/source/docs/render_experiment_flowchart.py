from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import math

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
W, H = 2000, 3820
img = Image.new('RGB', (W, H), '#FAFBFD')
draw = ImageDraw.Draw(img)
FONT_DIR = Path('/System/Library/Fonts/Supplemental')
def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ('Arial Bold.ttf' if bold else 'Arial.ttf')), size)

INK = '#172A43'
TEXT = '#384B64'
LINE = '#74879E'
BLUE = '#285D9A'
FILL = '#F0F5FC'
BORDER = '#C3D2E5'

def center_text(text, cy, f, color=INK):
    box = draw.textbbox((0, 0), text, font=f)
    draw.text(((W - (box[2] - box[0])) / 2, cy), text, font=f, fill=color)

center_text('Fast matrix multiplication on TPU', 57, font(66, True))
center_text('Proposed experiment sequence · Updated', 144, font(39), TEXT)
center_text('Shared suite: square, rectangular, LLM and partial-tile shapes', 210, font(33), TEXT)

# Preserve the original numbers; insert N6a between explanation and validation.
nodes = {
    'N1': (425, 315, 1150, 228),
    'N2': (425, 633, 1150, 222),
    'N3': (90, 960, 865, 272),
    'N4': (1045, 960, 865, 272),
    'N5': (425, 1337, 1150, 380),
    'N6': (425, 1817, 1150, 246),
    'N6a': (425, 2163, 1150, 300),
    'N7': (425, 2563, 1150, 280),
    'N8': (425, 2943, 1150, 246),
    'N9': (425, 3289, 1150, 246),
}
edges = [('N1', 'N2'), ('N2', 'N3'), ('N2', 'N4'), ('N3', 'N5'),
         ('N4', 'N5'), ('N5', 'N6'), ('N6', 'N6a'), ('N6a', 'N7'),
         ('N7', 'N8'), ('N8', 'N9')]

def arrow(points):
    draw.line(points, fill=LINE, width=5, joint='curve')
    x, y = points[-1]
    px, py = points[-2]
    a = math.atan2(y-py, x-px)
    length, half = 20, 11
    bx, by = x-length*math.cos(a), y-length*math.sin(a)
    draw.polygon([(x, y), (bx+half*math.sin(a), by-half*math.cos(a)),
                  (bx-half*math.sin(a), by+half*math.cos(a))], fill=LINE)

for source, dest in edges:
    sx, sy, sw, sh = nodes[source]
    dx, dy, dw, dh = nodes[dest]
    start = (sx + sw/2, sy + sh)
    end = (dx + dw/2, dy)
    mid = (start[1]+end[1])/2
    # Separate incoming ports make the two prerequisites clearly visible.
    if dest == 'N5':
        end = (dx + dw*(0.30 if start[0] < W/2 else 0.70), dy)
    arrow([start, (start[0], mid), (end[0], mid), end])

content = {
    'N1': ('What happens with basic implementations?',
           ['Plain cubic, one-level Strassen and native XLA.',
            'Check correctness and timing across the shape suite.']),
    'N2': ('How much does tile choice matter?',
           ['Sweep matching tiles across shapes.',
            'Identify promising regions and poor fits.']),
    'N3': ('Which MM optimizations\nactually help?',
           ['Change one thing at a time, then combine.',
            'Separate individual gains from interactions.']),
    'N4': ('Where does numerical\naccuracy deteriorate?',
           ['Test random, difficult and real matrix values.',
            'Identify accuracy limits and needed fallbacks.']),
    'N5': ('Who wins after fair tuning and preparation?',
           ['Retune Strassen and strong full-tile cubic fairly.',
            'Compare both with native XLA.',
            'Time the kernel alone and the complete MM call,',
            'including padding and required layout changes.',
            'Separate per-call work from reusable preparation.']),
    'N6': ('Why do some shapes win and others lose?',
           ['Profile representative wins, ties and losses.',
            'Explain compute, memory and scheduling costs.']),
    'N6a': ('Can we predict when Strassen will help?',
            ['Build a rule that selects cubic or Strassen',
             'and, optionally, a tile configuration.',
             'Freeze the rule before testing on new shapes.']),
    'N7': ('Does the frozen rule generalize?',
           ['Test shapes not used to build or tune the rule.',
            'Compare its choices with measured alternatives.',
            'Repeat in fresh v5e/v6e runs; report wins and losses.']),
    'N8': ('Does subsequent computation add a gain?',
           ['Compare MM plus activation or residual work.',
            'Rerun fusion and early-finalization experiments.']),
    'N9': ('Do these gains help actual LLMs?',
           ['Qwen plus Mistral and Gemma.',
            'Measure layer/model performance and model quality.']),
}

for key, (x, y, width, height) in nodes.items():
    accent, fill, border = BLUE, FILL, BORDER
    if key == 'N4':
        accent, fill, border = '#936117', '#FBF6EB', '#DAC9A8'
    elif key == 'N9':
        accent, fill, border = '#176B63', '#EBF6F3', '#B8D7CF'
    draw.rounded_rectangle((x,y,x+width,y+height), radius=22, fill=fill, outline=border, width=3)
    draw.rounded_rectangle((x+27,y+27,x+118,y+82), radius=11, fill=accent)
    draw.text((x+42,y+34), key, font=font(32 if key == 'N6a' else 36,True), fill='white')
    title, lines = content[key]
    tx, ty = x+145, y+32
    heading = font(39, True)
    for line in title.split('\n'):
        assert draw.textlength(line, font=heading) <= width-172, (key, line)
        draw.text((tx,ty), line, font=heading, fill=INK)
        ty += 49
    body_y = max(y+105,ty+14)
    body = font(34)
    for line in lines:
        assert draw.textlength(line, font=body) <= width-60, (key, line)
        draw.text((x+30,body_y), line, font=body, fill=TEXT)
        body_y += 47
    assert body_y < y+height, key

center_text('Optional later study: try two-level Strassen.', 3600, font(34, True), TEXT)
center_text('Compare 0, 1 and 2 levels on a small selection of shapes.', 3655, font(32), TEXT)
center_text('Questions to investigate — no outcomes are assumed.', 3750, font(31), TEXT)
path = OUT / 'proposed-experiment-flowchart-v2.png'
img.save(path, dpi=(220,220), optimize=True)
print(path)
print(f'{W} × {H} pixels; {len(nodes)} nodes; {len(edges)} arrows')
