"""Version 002: corrected white diamond fill in the explanatory key."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import math
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from pypdf import PdfReader, PdfWriter, Transformation

W,H=1008,792
INK='#172A43'; MUTED='#475467'; BLUE='#3772C8'; ORANGE='#DD8734'; GREEN='#24936D'


def para(c,text,x,top,width,font=11,leading=15,bold=False,color=INK,max_height=None):
    style=ParagraphStyle('p',fontName='Helvetica-Bold' if bold else 'Helvetica',fontSize=font,
                         leading=leading,textColor=HexColor(color))
    p=Paragraph(escape(text),style);pw,ph=p.wrap(width,H)
    if max_height is not None:assert ph<=max_height,(text,ph,max_height)
    p.drawOn(c,x,top-ph)
    return ph


def footer(c,page,title='Fast matrix multiplication on TPU | v5e results'):
    c.setStrokeColor(HexColor('#D9E1EB'));c.line(40,36,W-40,36)
    c.setFillColor(HexColor(MUTED));c.setFont('Helvetica',8)
    c.drawString(40,22,title);c.drawRightString(W-40,22,f'{page} / 49')


def arrow(c,points):
    c.setStrokeColor(HexColor('#8296AB'));c.setFillColor(HexColor('#8296AB'));c.setLineWidth(1.6)
    p=c.beginPath();p.moveTo(*points[0])
    for xy in points[1:]:p.lineTo(*xy)
    c.drawPath(p)
    x,y=points[-1];px,py=points[-2];a=math.atan2(y-py,x-px);length=7;half=3.2
    p=c.beginPath();p.moveTo(x,y);p.lineTo(x-length*math.cos(a)+half*math.sin(a),y-length*math.sin(a)-half*math.cos(a));p.lineTo(x-length*math.cos(a)-half*math.sin(a),y-length*math.sin(a)+half*math.cos(a));p.close();c.drawPath(p,fill=1,stroke=0)


def node(c,x,top,width,height,title,body,accent=BLUE):
    c.setFillColor(HexColor('#F4F7FB'));c.setStrokeColor(HexColor('#C8D6E5'))
    c.roundRect(x,top-height,width,height,8,fill=1,stroke=1)
    c.setFillColor(HexColor(accent));c.roundRect(x,top-height,4,height,2,fill=1,stroke=0)
    th=para(c,title,x+14,top-12,width-28,font=12,leading=15,bold=True,max_height=38)
    para(c,body,x+14,top-18-th,width-28,font=10.5,leading=13.5,max_height=height-28-th)


def flowchart(c):
    para(c,'The study, question by question',40,760,930,font=26,leading=30,bold=True)
    para(c,'Completed v5e evidence. Arrows show the research logic, not the exact launch order.',40,720,930,font=11,color=MUTED)
    # Left column: initial MM questions, optimization/accuracy branch, fair tuning.
    node(c,40,686,420,84,'N1  What happens with basic implementations?',
         'Fixed tile, broad shape suite. Strassen beats the matched quadrant control on some shapes, but has no supported win over Native or full-tile Cubic.')
    node(c,40,574,420,84,'N2  How much does tile choice matter?',
         'Six tile choices per custom algorithm. Performance depends on shape; the largest tile can exceed on-chip memory.')
    node(c,40,462,200,114,'N3  Which changes help?',
         'Hold the tile fixed. Product order and output accumulation help selectively; gains depend on the algorithm and shape.')
    node(c,260,462,200,114,'N4  When does accuracy fail?',
         'Stress the inputs. All 24 cancellation cases fail for Strassen; ordinary and actual-weight tests can still pass.',accent='#AD731F')
    node(c,40,316,420,124,'N5  Who wins after independent tuning?',
         'Tune tile and variant separately for Cubic and Strassen on 16 shapes, then confirm with fresh samples. Strassen versus Native: 6 supported complete-call wins, 4 inconclusive, 6 losses.')
    arrow(c,[(250,602),(250,574)])
    arrow(c,[(250,490),(250,476),(140,476),(140,462)])
    arrow(c,[(250,490),(250,476),(360,476),(360,462)])
    arrow(c,[(140,348),(140,333),(166,333),(166,316)])
    arrow(c,[(360,348),(360,333),(334,333),(334,316)])
    # Right column: diagnostics, frozen selection, validation, applications.
    node(c,550,686,418,84,'N6  Do device traces support the timing story?',
         'Original representatives show ranking reversals. A post-hoc large-shape supplement also shows Strassen gains inside the TPU module.')
    node(c,550,574,418,84,'N6a  Freeze a rule from the N5 results',
         'Choose Native, Cubic or Strassen, including its tile and variant. Freeze before any N7 measurements. This is a selection step, not another timing race.')
    node(c,550,462,418,100,'N7  Does the frozen rule transfer?',
         'Evaluate on 16 reserved shapes against independently tuned comparators. All 3 custom choices beat Native; the same gains repeat on a fresh v5e allocation.')
    node(c,550,334,418,100,'N8  Do gains survive additional operations?',
         'Test activation and residual work. Fused Strassen is strong against custom Cubic; Native comparisons depend on the operation and preparation scope.',accent=GREEN)
    node(c,550,206,418,100,'N9  Do the gains help actual models?',
         'Qwen and Mistral pass the measured quality gates. Mistral resident layers improve, but transfer-inclusive model timing is near parity. Gemma is access-blocked.',accent=GREEN)
    arrow(c,[(460,254),(506,254),(506,644),(550,644)])
    for y0,y1 in [(602,574),(490,462),(362,334),(234,206)]:arrow(c,[(759,y0),(759,y1)])
    para(c,'How to follow the tree',40,157,420,font=12,leading=16,bold=True)
    para(c,'N1-N7 form the matrix-multiplication study. N8-N9 test whether those results survive progressively more realistic application work. N6a uses N5 data; N7 never refits it.',40,135,420,font=10.5,leading=14,max_height=66)
    para(c,'Future work: v6e replication and optional two-level Strassen. Neither is included in these results.',40,62,928,font=10,leading=13,color=MUTED)
    footer(c,1);c.showPage()


def guide(c,charts):
    para(c,'How to read the graphs',40,758,920,font=26,leading=30,bold=True)
    para(c,'Start with the question, then read the bar heights, then the interpretation beneath the graph.',40,717,920,font=12,leading=17,color=MUTED)
    left=40;cw=435
    items=[
      ('1. Compare within one shape','Each cluster is one matrix shape, written M x K x N for (M x K) times (K x N). Blue is Native, orange is Cubic and green is Strassen. Extra controls and N3 optimization variants use the legend shown on their graph.'),
      ('2. Find the lowest measured bar','For timing graphs, lower is faster. A short black horizontal segment marks the lowest eligible mean for that shape; a diamond identifies the bar. Different shapes get separate segments. These marks show the observed ordering, not statistical certainty.'),
      ('3. Read uncertainty separately','Ordinary timing whiskers show sample standard deviation, not confidence intervals. The explanations classify supported comparisons using the archived paired 95% intervals, without multiple-comparison correction. Inconclusive does not mean equal.'),
      ('4. Keep scopes and accuracy separate','Complete call includes required device preparation and finishing. Prepared kernel reuses prepared buffers. Neither is full inference. Device traces, resident layers and streamed forwards are separate scopes. In N4 the black marker means lowest median error; red dashed lines are tolerance gates.'),
    ]
    top=667
    for title,body in items:
        para(c,title,left,top,cw,font=13,leading=17,bold=True)
        bh=para(c,body,left,top-25,cw,font=11,leading=15,max_height=100)
        top-=bh+52
    x=550
    para(c,'A marker is not a significance test',x,667,418,font=13,leading=17,bold=True)
    # Schematic only: makes the new segment and diamond concrete.
    bx=[605,725,845]; heights=[80,68,58];base=494
    for xx,hh,co,lab in zip(bx,heights,[BLUE,ORANGE,GREEN],['Native','Cubic','Strassen']):
        c.setFillColor(HexColor(co));c.rect(xx,base,64,hh,fill=1,stroke=0)
        para(c,lab,xx-8,base-10,80,font=10,leading=13)
    c.setStrokeColor(HexColor(INK));c.setLineWidth(1.2);c.setDash(4,3);c.line(590,base+58,921,base+58);c.setDash()
    c.setFillColor(HexColor('#FFFFFF'));c.saveState();c.translate(877,base+58);c.rotate(45);c.rect(-3,-3,6,6,fill=1,stroke=1);c.restoreState()
    para(c,'Illustration only; these are not experiment measurements.',x,459,418,font=9.5,leading=13,color=MUTED)
    para(c,'Experiment index',x,409,418,font=13,leading=17,bold=True)
    labels={'N1':'Basic baselines','N2':'Tile sweeps','N3':'Optimization changes','N4':'Numerical accuracy',
            'N5':'Independent tuning and confirmation','N6':'Device profiling and its supplement',
            'N7':'Held-out evaluation and fresh-v5e repeat','N8':'MM plus application operations','N9':'Real-model timing and quality'}
    for i,exp in enumerate(labels):
        page=3+next(j for j,q in enumerate(charts) if q['experiment']==exp)
        y=377-i*27
        para(c,exp+'  '+labels[exp],x,y,354,font=10.5,leading=14)
        c.setFont('Helvetica-Bold',10.5);c.setFillColor(HexColor(INK));c.drawRightString(968,y-11,str(page))
    para(c,'Failures are retained. Selection screens are exploratory. Different allocations are never pooled. All 47 original overview comparisons are included.',x,112,418,font=10.5,leading=14,max_height=61)
    footer(c,2);c.showPage()


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);args=p.parse_args();out=args.artifacts
    manifest=json.loads((out/'explained_manifest.json').read_text());charts=manifest['charts']
    annotation=out/'annotation_pages.pdf';assert not annotation.exists()
    c=canvas.Canvas(str(annotation),pagesize=(W,H),pageCompression=1)
    c.setTitle('N1-N9: experiment flowchart and explained comparisons')
    flowchart(c);guide(c,charts)
    for i,q in enumerate(charts,3):
        c.setFillColor(HexColor(MUTED));c.setFont('Helvetica',9)
        c.drawString(40,774,q['experiment']+' | '+q['scope'].replace('_',' ')+' | '+q['cohort'])
        for x,title,body in [(40,'WHAT THIS GRAPH SHOWS',q['explanation']['observation']),
                             (526,'OUR INTERPRETATION',q['explanation']['interpretation'])]:
            para(c,title,x,233,442,font=10,leading=13,bold=True,color=BLUE)
            para(c,body,x,213,442,font=11,leading=15,max_height=165)
        footer(c,i);c.showPage()
    c.save()
    bg=PdfReader(annotation);figs=PdfReader(out/'N1_N9_key_comparisons.pdf');assert len(figs.pages)==47 and len(bg.pages)==49
    writer=PdfWriter()
    for i,page in enumerate(bg.pages):
        if i>=2:
            figure=figs.pages[i-2];fw=float(figure.mediabox.width);fh=float(figure.mediabox.height)
            scale=min(928/fw,501/fh);x=(W-fw*scale)/2;y=759-fh*scale
            assert y>=255
            page.merge_transformed_page(figure,Transformation().scale(scale).translate(x,y),expand=False)
        writer.add_page(page)
    writer.add_outline_item('Experiment flowchart',0);writer.add_outline_item('How to read the graphs',1)
    for exp in dict.fromkeys(q['experiment'] for q in charts):
        first=next(i for i,q in enumerate(charts) if q['experiment']==exp)
        writer.add_outline_item(exp+' comparisons',first+2)
    writer.add_metadata({'/Title':'Fast MM on TPU: N1-N9 explained results','/Author':'Strassen MM Focus',
                         '/Subject':'Completed v5e study; per-shape observed-best markers, uncertainty and interpretation'})
    dest=out/'N1_N9_explained_comparisons.pdf'
    with dest.open('xb') as handle:writer.write(handle)
    check=PdfReader(dest);assert len(check.pages)==49
    for i,page in enumerate(check.pages):
        txt=page.extract_text()
        if i>=2:assert 'WHAT THIS GRAPH SHOWS' in txt and 'OUR INTERPRETATION' in txt
    print(json.dumps({'pdf':str(dest),'pages':49,'all_47_figures_have_explanations':True}),flush=True)


if __name__=='__main__':main()
