"""Build the visual Phase 4 deck, PDF companion and plain-English notes.

Run: python scripts/build_presentation.py
Uses only credited presentation derivatives and saved results. Photos preserve
aspect ratio and full framing; diagrams, labels and chart bars remain editable.
The fallback PDF uses the same geometry as the PPTX. With --office, the final
PDF is exported from the actual PowerPoint by headless LibreOffice. Neither
this builder nor its checks train models or change the paper.
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN
from pptx.util import Pt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/presentation'
W,H=960,540
INK,MUTED,GREEN,BG,LINE,LIME='17382D','52645C','176A4C','F3F6F2','D8E2D9','D8ED9D'
FROZEN,FINE,WRONG='526D94','168468','AF4C32'
FONT_DIR=Path('/usr/share/fonts/truetype/dejavu')
def register_fonts():
    """Use the existing Linux fonts or Matplotlib's bundled copies on other OSes."""
    import matplotlib
    candidates = [FONT_DIR, Path(matplotlib.get_data_path()) / 'fonts' / 'ttf']
    filenames = [('Deck', 'DejaVuSans.ttf'), ('DeckBold', 'DejaVuSans-Bold.ttf')]
    for directory in candidates:
        if all((directory / filename).is_file() for _, filename in filenames):
            for name, filename in filenames:
                pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
            return directory
    raise FileNotFoundError('DejaVu Sans fonts are missing. Reinstall matplotlib in the project environment.')


def wrap(text,width,size,bold=False):
    lines=[]
    for paragraph in text.split('\n'):
        line=''
        for word in paragraph.split():
            proposed=f'{line} {word}'.strip()
            if line and pdfmetrics.stringWidth(proposed,'DeckBold' if bold else 'Deck',size)>width:
                lines.append(line);line=word
            else:line=proposed
        lines.append(line)
    return lines


class Deck:
    def __init__(self,manifest):
        register_fonts()
        self.manifest=manifest
        self.ppt=Presentation();self.ppt.slide_width=Pt(W);self.ppt.slide_height=Pt(H)
        self.pdf=canvas.Canvas(str(OUT/'Phase-4-Presentation.pdf'),pagesize=(W,H))
        self.pdf.setTitle('Phase 4 — Plant Disease Classification Across Image Datasets')
        self.pdf.setAuthor('coolTeamName.ipynb')
        self.bounds=[];self.slide_number=0

    def color(self,color): return tuple(int(color[i:i+2],16)/255 for i in (0,2,4))

    def shape(self,x,y,w,h,color,ellipse=False):
        if x<0 or y<0 or x+w>W+.1 or y+h>H+.1:raise ValueError('Shape outside slide')
        s=self.current.shapes.add_shape(MSO_SHAPE.OVAL if ellipse else MSO_SHAPE.RECTANGLE,Pt(x),Pt(y),Pt(w),Pt(h))
        s.fill.solid();s.fill.fore_color.rgb=RGBColor.from_string(color);s.line.fill.background()
        self.pdf.setFillColorRGB(*self.color(color))
        if ellipse:self.pdf.ellipse(x,H-y-h,x+w,H-y,fill=1,stroke=0)
        else:self.pdf.rect(x,H-y-h,w,h,fill=1,stroke=0)
        return s

    def line(self,x1,y1,x2,y2,color=GREEN,width=2):
        s=self.current.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Pt(x1),Pt(y1),Pt(x2),Pt(y2))
        s.line.color.rgb=RGBColor.from_string(color);s.line.width=Pt(width)
        self.pdf.setStrokeColorRGB(*self.color(color));self.pdf.setLineWidth(width)
        self.pdf.line(x1,H-y1,x2,H-y2)

    def arrow(self,x1,y1,x2,y2,color=GREEN):
        # Arrowheads are ordinary editable lines and work in both renderers.
        import math
        self.line(x1,y1,x2,y2,color)
        theta=math.atan2(y2-y1,x2-x1)
        for offset in [-.55,.55]:
            self.line(x2,y2,x2-9*math.cos(theta+offset),y2-9*math.sin(theta+offset),color)

    def text(self,text,x,y,w,h,size=20,color=INK,bold=False,center=False):
        lines=wrap(str(text),w,size,bold);leading=size*1.22
        if len(lines)*leading>h+.5:raise ValueError(f'Slide {self.slide_number}: text too tall: {text!r}')
        if any(pdfmetrics.stringWidth(s,'DeckBold' if bold else 'Deck',size)>w+.5 for s in lines):
            raise ValueError(f'Slide {self.slide_number}: text too wide: {text!r}')
        if min(x,y)<0 or x+w>W+.1 or y+h>H+.1:raise ValueError('Text outside slide')
        box=self.current.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(h));f=box.text_frame
        f.margin_left=f.margin_right=f.margin_top=f.margin_bottom=0;f.word_wrap=False
        for i,line in enumerate(lines):
            p=f.paragraphs[0] if i==0 else f.add_paragraph();p.text=line
            p.font.name='Arial';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
            p.line_spacing=Pt(leading);p.space_before=Pt(0);p.space_after=Pt(0)
            p.alignment=PP_ALIGN.CENTER if center else PP_ALIGN.LEFT
        self.pdf.setFont('DeckBold' if bold else 'Deck',size);self.pdf.setFillColorRGB(*self.color(color))
        for i,line in enumerate(lines):
            if center:self.pdf.drawCentredString(x+w/2,H-y-size-i*leading,line)
            else:self.pdf.drawString(x,H-y-size-i*leading,line)
        self.bounds.append({'slide':self.slide_number,'text':str(text),'x':x,'y':y,'w':w,'h':h,'font':size})
        return box

    def photo(self,name,x,y,w,h):
        asset=self.manifest['assets'][name];path=OUT/'assets'/asset['file']
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=asset['sha256']:raise ValueError(f'Changed asset: {name}')
        self.shape(x,y,w,h,'FFFFFF')
        with Image.open(path) as im:iw,ih=im.size
        scale=min((w-12)/iw,(h-12)/ih);dw,dh=iw*scale,ih*scale
        px=x+(w-dw)/2;py=y+(h-dh)/2
        pic=self.current.shapes.add_picture(str(path),Pt(px),Pt(py),width=Pt(dw),height=Pt(dh))
        pic._element.nvPicPr.cNvPr.set('descr',f"{name}: {asset.get('label',asset['changes'])}")
        self.pdf.drawImage(str(path),px,H-py-dh,dw,dh,mask='auto')

    def tag(self,text,x,y,w,color=GREEN):
        self.shape(x,y,w,29,color);self.text(text,x+8,y+6,w-16,20,12,'FFFFFF',True)

    def new_slide(self,item,total):
        self.slide_number+=1
        self.current=self.ppt.slides.add_slide(self.ppt.slide_layouts[6])
        self.shape(0,0,W,H,INK if item['kind'] in ['cover','conclusion'] else BG)
        if item['kind'] not in ['cover','conclusion']:
            self.shape(40,28,32,4,GREEN)
            self.text('PHASE 4  /  '+('BACKUP' if self.slide_number>10 else 'RESEARCH FINDINGS'),82,22,730,20,10,GREEN,True)
            self.text(item['title'],40,55,880,78,32,INK,True)
            self.text(item['subtitle'],40,139,880,32,17,MUTED)
        photos=[p for p in item['photos'] if p in self.manifest['assets']]
        credit=''
        if photos:
            credit='Images: '+', '.join(photos)+'. '
            datasets={self.manifest['assets'][p]['dataset'] for p in photos}
            if 'plantvillage' in datasets:credit+='PV: Mohanty, CC BY-SA 3.0. '
            if 'plantdoc' in datasets:credit+='PD: Singh/Kayal et al., CC BY 4.0.'
        else:credit='Original experiment 9619975 · measured results and full sources in the paper.'
        color='D0E2D5' if item['kind'] in ['cover','conclusion'] else MUTED
        self.text(credit,40,506,820,28,8.5,color)
        self.text(f'{self.slide_number:02} / {total}',876,510,60,18,9,color)
        notes=item['notes']
        if photos:
            notes+='\n\nImage credits (full frames retained unless identified as model processing):'
            for name in photos:
                asset=self.manifest['assets'][name];source=self.manifest['sources'][asset['dataset']]
                original=self.manifest['assets'].get(asset.get('source_photo'),asset)
                notes+=f"\n{name}: {source['creator']}; {source['license']} ({source['license_url']}). Source: {source['url']}. File: {original['source_path']}. Changes: {asset['changes']}"
        self.current.notes_slide.notes_text_frame.text=f"Suggested speaker: {item['presenter']}\nSuggested time: {item['seconds']} seconds (provisional)\n\n{notes}"
        return notes

    def finish(self):self.pdf.showPage()

    def save(self):
        self.ppt.save(OUT/'Phase-4-Presentation.pptx');self.pdf.save()
        checks=ROOT/'outputs/presentation-redesign';checks.mkdir(exist_ok=True,parents=True)
        (checks/'layout.json').write_text(json.dumps(self.bounds,indent=2)+'\n', encoding='utf-8')


def build_slide(d,item,results):
    k=item['kind']
    if k=='cover':
        d.text('coolTeamName.ipynb',42,42,400,25,15,LIME,True)
        d.text(item['title'],42,124,420,150,42,'FFFFFF',True)
        d.text(item['subtitle'],42,293,410,65,27,LIME)
        d.photo('pd-apple-01',490,65,430,355)
        d.text('A familiar leaf. A different image collection.',490,434,430,50,18,'FFFFFF')
        d.text('Jacob Ayoub · Max Nanasy · Troy Waters · Shriyansh Singh',42,454,870,30,16,'FFFFFF')
    elif k=='question':
        for x,name,title,sub in [(40,'pv-apple-01','PlantVillage','Training collection'),(510,'pd-apple-02','PlantDoc','Different collection')]:
            d.photo(name,x,190,410,238);d.text(title,x,438,410,30,23,INK,True)
            d.text(sub,x,470,410,25,15,MUTED)
        d.shape(458,279,44,44,GREEN,True);d.text('?',461,282,38,39,28,'FFFFFF',True,True)
    elif k=='datasets':
        for x,names,label in [(40,['pv-apple-01','pv-grape-01'],'PlantVillage'),(510,['pd-apple-01','pd-grape-01'],'Cropped PlantDoc')]:
            d.tag(label,x,186,410)
            for j,(name,condition) in enumerate(zip(names,['Apple scab','Grape black rot'])):
                d.photo(name,x+j*210,226,200,195);d.text(condition,x+j*210,431,200,26,16,INK,True,True)
        d.text('22 shared categories  •  5 uncertain pairs excluded',40,472,880,29,20,GREEN,True,True)
    elif k=='protocol':
        d.tag('DEVELOPMENT · PLANTVILLAGE',40,188,445)
        d.tag('FINAL TESTS',640,188,280)
        for x,y,title,count in [(40,241,'Train','18,023'),(275,241,'Validate','4,529')]:
            d.shape(x,y,210,145,'FFFFFF');d.text(title,x+15,y+16,180,25,18,MUTED)
            d.text(count,x+15,y+53,180,54,36,GREEN,True)
        d.arrow(250,312,270,312)
        d.text('Fit weights',40,399,210,28,17,MUTED,center=True)
        d.text('Select checkpoints',260,399,230,28,17,MUTED,center=True)
        d.arrow(492,312,610,312)
        d.text('Saved\nmodels',504,245,105,55,17,INK,True,True)
        d.line(620,268,620,392);d.arrow(610,312,620,312)
        for y,label,count in [(233,'PlantVillage','5,487'),(351,'PlantDoc','183')]:
            d.arrow(620,y+45,640,y+45);d.shape(645,y,275,97,'FFFFFF')
            d.text(label,662,y+12,235,27,19,INK,True);d.text(count,662,y+43,235,42,30,GREEN,True)
        d.text('Fixed image lists · 3 seeds · no test-driven model selection',40,468,880,29,18,MUTED,center=True)
    elif k=='models':
        d.photo('pv-apple-01',40,227,150,166)
        for y,name,color,feature in [(205,'Frozen',FROZEN,'FIXED'),(347,'Fine-tuned',FINE,'TRAINED')]:
            d.text(name,222,y-32,180,30,21,color,True)
            d.arrow(195,310,238,y+40,color)
            for shift in [10,5,0]:d.shape(250+shift,y+shift,245,81,'D8E2D9' if shift else color)
            d.text('ImageNet features',264,y+12,220,28,19,'FFFFFF',True)
            d.text(feature,264,y+44,220,23,13,'FFFFFF')
            d.arrow(505,y+40,613,y+40,color);d.text('576 values',510,y+9,103,23,13,MUTED,center=True)
            d.shape(620,y,180,81,FINE);d.text('Category layer',635,y+12,150,27,17,'FFFFFF',True)
            d.text('TRAINED',635,y+44,150,22,13,'FFFFFF')
            d.arrow(805,y+40,839,y+40,color);d.text('22\nlabels',847,y+10,72,65,20,INK,True)
        d.text('Fine-tuning starts after one warmup epoch. Both models are fixed during the demo.',40,471,880,27,16,MUTED)
    elif k=='process':
        for x,title in [(40,'1  Photograph'),(265,'2  Prepare'),(490,'3  Responses'),(715,'4  Predict')]:d.text(title,x,189,200,28,20,GREEN,True)
        d.photo('pd-apple-02',40,230,200,195);d.photo('prepared-apple',265,230,200,195)
        d.photo('late-frozen',490,246,90,140);d.photo('late-finetune',595,246,90,140)
        d.text('Frozen',490,391,90,23,13,FROZEN,True,True);d.text('Fine-tuned',590,391,100,23,13,FINE,True,True)
        for x in [243,468,693]:d.arrow(x,325,x+17,325)
        for y,mode,color,category in [(244,'Frozen',FROZEN,'Potato /\nlate blight'),(348,'Fine-tuned',FINE,'Tomato /\nearly blight')]:
            d.text(mode,723,y,190,25,16,color,True);d.text(category,723,y+28,190,59,19,WRONG,True)
        d.text('Label: apple scab',40,445,240,26,16,INK,True)
        d.text('Actual maps ≠ disease masks',318,445,320,26,16,MUTED)
        d.text('Both differ from label',700,445,220,26,15,WRONG,True)
        d.text('Selected example · seed 551 · open the live demo or recorded replay',40,478,880,22,13,MUTED)
    elif k=='results':
        x0,span=277,510
        for value in [0,25,50,75,100]:
            x=x0+value/100*span;d.line(x,215,x,438,LINE,1);d.text(str(value),x-20,190,40,22,12,MUTED,center=True)
        d.text('Accuracy (%)',x0,165,span,23,13,MUTED,center=True)
        for y,split,label in [(225,'source_test','PlantVillage'),(350,'target_test','PlantDoc')]:
            count=results['groups']['frozen'][split]['images']
            d.text(label,40,y+8,200,34,23,INK,True);d.text(f'{count:,} test images',40,y+47,210,27,16,MUTED)
            for offset,mode,color in [(0,'frozen',FROZEN),(51,'finetune',FINE)]:
                metric=results['groups'][mode][split]['accuracy'];value=metric['mean']*100;sd=metric['sample_standard_deviation']*100
                yy=y+offset;d.shape(x0,yy,span*value/100,26,color)
                a=x0+span*(value-sd)/100;b=x0+span*(value+sd)/100
                d.line(a,yy+13,b,yy+13,INK,1.4);d.line(a,yy+7,a,yy+19,INK,1.4);d.line(b,yy+7,b,yy+19,INK,1.4)
                d.text(f'{value:.2f}%',805,yy-1,114,35,23,color,True)
        d.shape(277,460,16,16,FROZEN);d.text('Frozen',302,456,130,25,16,FROZEN,True)
        d.shape(437,460,16,16,FINE);d.text('Fine-tuned',462,456,170,25,16,FINE,True)
        d.text('Whiskers: seed SD, not new-image uncertainty',40,486,850,18,11,MUTED)
    elif k=='examples':
        for i,name in enumerate(['pv-apple-01','pv-apple-02','pd-apple-02']):
            x=40+i*300;asset=d.manifest['assets'][name]
            d.photo(name,x,184,280,173)
            d.text('PlantVillage' if asset['dataset']=='plantvillage' else 'PlantDoc',x,365,280,27,18,INK,True)
            for y,mode in [(402,'frozen'),(447,'finetune')]:
                pred=asset['predictions'][mode];color=GREEN if pred['matches_label'] else WRONG
                d.text('Frozen' if mode=='frozen' else 'Fine-tuned',x,y,95,23,12,MUTED)
                d.text(pred['label'].replace(' / ',' /\n'),x+100,y-5,180,50,15,color,True)
                d.shape(x+80,y+5,9,9,color,True)
        d.text('Green = matches label · Rust = differs · illustrative cases, not an accuracy estimate',40,486,880,18,11,MUTED)
    elif k=='limits':
        for x in [40,340,640]:d.shape(x,195,280,275,'FFFFFF')
        d.text('183',60,217,240,73,58,GREEN,True,True)
        for row in range(3):
            for col in range(8):d.shape(93+col*22,310+row*22,10,10,LINE,True)
        d.text('Small external test',58,403,245,34,20,INK,True,True)
        d.text('Same images across seeds',58,438,245,23,13,MUTED,center=True)
        d.shape(441,280,78,78,GREEN,True);d.text('?',443,282,74,72,51,'FFFFFF',True,True)
        for x,y,label in [(355,222,'Lighting'),(498,222,'Framing'),(355,368,'Background'),(502,368,'Labels')]:
            d.text(label,x,y,110,25,14,MUTED,center=True)
            d.arrow(x+55,y+29 if y<300 else y-7,460 if x<480 else 500,280 if y<300 else 345,LINE)
        d.text('Causes not isolated',350,426,260,32,20,INK,True,True)
        d.photo('pd-apple-01',666,212,228,163);d.tag('DATASET LABEL ≠ DIAGNOSIS',650,386,260,WRONG)
        d.text('Annotations can be wrong',657,432,246,27,16,MUTED,center=True)
    elif k=='conclusion':
        d.text('THE TAKEAWAY',42,48,400,25,13,LIME,True)
        d.text('Evaluate beyond\nthe training\ncollection.',42,135,420,182,39,'FFFFFF',True)
        d.text(item['subtitle'],42,348,410,75,23,LIME)
        d.photo('pd-apple-01',500,85,420,340)
        d.text('PlantDoc · illustrative photograph',500,440,420,27,16,'FFFFFF')
    elif k=='reproduction':
        labels=['Audit','Split','Train','Verify','Evaluate']
        for i,label in enumerate(labels):
            x=40+i*178
            for shift in [8,4,0]:d.shape(x+12+shift,226+shift,110,115,LINE if shift else GREEN)
            d.text(str(i+1),x+22,244,90,65,43,'FFFFFF',True,True)
            d.text(label,x,366,156,35,23,INK,True,True)
            if i<4:d.arrow(x+143,287,x+174,287)
        d.text('docs/reproduction.md',40,430,460,32,23,GREEN,True)
        d.text('ONE-CLICK-LAUNCHERS/',510,430,410,32,23,GREEN,True)
        d.text('Repeat the original experiment',40,468,450,26,16,MUTED)
        d.text('Present the existing saved models',510,468,410,26,16,MUTED)
    elif k=='references':
        for i,ref in enumerate(item['refs']):d.text(ref,40,185+i*58,880,54,12,MUTED)
        d.line(40,423,920,423,LINE,1)
        d.text('REAL PHOTO CREDITS',40,437,250,25,13,GREEN,True)
        d.text('PlantVillage: S. P. Mohanty · CC BY-SA 3.0\nPlantDoc: D. Singh et al. / P. Kayal et al. · CC BY 4.0',280,434,640,49,14,INK)
        d.text('Photo IDs, pinned sources and licenses: PHOTO_CREDITS.md',40,482,880,20,12,MUTED)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--office', help='Optional LibreOffice/soffice executable to export the final PDF from the PPTX')
    args=parser.parse_args()
    content=json.loads((OUT/'slides.json').read_text(encoding='utf-8'));manifest=json.loads((OUT/'assets/manifest.json').read_text(encoding='utf-8'))
    results=json.loads((ROOT/'docs/evaluation/final-results.json').read_text(encoding='utf-8'))
    expected=json.loads((ROOT/'demo/examples.json').read_text(encoding='utf-8'))['results_sha256']
    if hashlib.sha256((ROOT/'docs/evaluation/final-results.json').read_bytes()).hexdigest()!=expected:
        raise ValueError('Original result evidence changed')
    deck=Deck(manifest);notes=['# Phase 4 speaker notes','',content['timing_note'],'',
        'Suggested portions: Jacob 1–3; Troy 4–5; Max 6–7; Shriyansh 8–10. Slides 11–12 are backup.',
        'Confirm the assigned duration and rehearse. For five minutes, shorten background and use one demo example.',
        '', 'Photo credits and original source paths are included below and in PHOTO_CREDITS.md.','']
    for number,item in enumerate(content['slides'],1):
        expanded=deck.new_slide(item,len(content['slides']));build_slide(deck,item,results);deck.finish()
        notes += [f"## {number}. {item['title'].replace(chr(10),' ')}",'',
                  f"Suggested speaker: **{item['presenter']}** · {item['seconds']} seconds (provisional).",'',expanded,'']
    deck.save();(OUT/'speaker-notes.md').write_text('\n'.join(notes).rstrip()+'\n', encoding='utf-8')
    if args.office:
        # Isolate the office profile and use its headless renderer only.
        with tempfile.TemporaryDirectory(prefix='cecs-slide-render-') as directory:
            folder=Path(directory)
            subprocess.run([args.office, f'-env:UserInstallation={(folder/"profile").as_uri()}',
                '--headless','--convert-to','pdf','--outdir',str(folder),
                str(OUT/'Phase-4-Presentation.pptx')],check=True,
                env=dict(os.environ,SAL_USE_VCLPLUGIN='svp'),timeout=90)
            shutil.copyfile(folder/'Phase-4-Presentation.pdf',OUT/'Phase-4-Presentation.pdf')
    print(f'Built {len(content["slides"])} visual slides with editable diagrams/charts and credited real images.')

if __name__=='__main__':main()
