"""Build editable slides, a matching PDF companion, and readable speaker notes.

Run: python scripts/build_presentation.py
Requires requirements-presentation.txt. Both exports use the same layout/content;
the PDF is drawn from that layout, not a screenshot of the PowerPoint. Text and
shapes in the PPTX remain editable; the two measured-result plots are images.
The submitted paper and experimental evidence are never modified.
"""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.util import Pt
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/presentation'
W, H = 960, 540
INK, MUTED, GREEN, BG, LINE, LIME = '17382D','52645C','176A4C','F3F6F2','D8E2D9','D8ED9D'
FONT_DIR = Path('/usr/share/fonts/truetype/dejavu')
for name, file in [('Deck','DejaVuSans.ttf'),('DeckBold','DejaVuSans-Bold.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONT_DIR/file)))


def wrap(text, width, size, bold=False):
    """Use explicit line breaks in both exports so text remains predictable."""
    lines = []
    font = 'DeckBold' if bold else 'Deck'
    for paragraph in text.split('\n'):
        line = ''
        for word in paragraph.split():
            proposed = f'{line} {word}'.strip()
            if line and pdfmetrics.stringWidth(proposed,font,size)>width:
                lines.append(line);line=word
            else:
                line=proposed
        lines.append(line)
    return lines


class Deck:
    def __init__(self):
        self.ppt = Presentation()
        self.ppt.slide_width, self.ppt.slide_height = Pt(W),Pt(H)
        self.pdf = canvas.Canvas(str(OUT/'Phase-4-Presentation.pdf'),pagesize=(W,H))
        self.pdf.setTitle('Phase 4 — Plant Disease Classification Across Image Datasets')
        self.pdf.setAuthor('coolTeamName.ipynb')
        self.bounds=[]

    def slide(self, number, item, total):
        self.current = self.ppt.slides.add_slide(self.ppt.slide_layouts[6])
        self.current.notes_slide.notes_text_frame.text = f"Suggested speaker: {item['presenter']}\nSuggested time: {item['seconds']} seconds (provisional)\n\n{item['notes']}"
        self.rect(0,0,W,H,BG)
        if item['kind']=='title':
            self.rect(0,0,W,H,INK)
            self.rect(54,54,92,6,LIME)
            self.text('PHASE 4 / RESEARCH FINDINGS',54,83,850,24,13,LIME,True)
            self.text(item['title'],54,142,850,150,42,'FFFFFF',True)
            self.text(item['subtitle'],54,320,830,75,23,'D0E2D5')
            self.text('Jacob Ayoub · Max Nanasy · Troy Waters · Shriyansh Singh',54,444,850,35,16,'FFFFFF')
            self.text('coolTeamName.ipynb',54,488,850,20,12,LIME)
            return
        self.rect(0,0,10,H,GREEN)
        self.text(f'PLANT DISEASE / PHASE 4    {"BACKUP" if number>10 else "FINDINGS"}',40,24,880,20,11,GREEN,True)
        self.text(item['title'],40,60,880,79,31,INK,True)
        self.text(item['subtitle'],40,143,880,49,17,MUTED)
        self.text(item.get('source',''),40,506,825,27,9,MUTED)
        self.text(f'{number:02} / {total}',885,512,65,16,9,MUTED)

    def rect(self,x,y,w,h,color):
        shape = self.current.shapes.add_shape(MSO_SHAPE.RECTANGLE,Pt(x),Pt(y),Pt(w),Pt(h))
        shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string(color)
        shape.line.fill.background()
        self.pdf.setFillColorRGB(*[int(color[i:i+2],16)/255 for i in (0,2,4)])
        self.pdf.rect(x,H-y-h,w,h,fill=1,stroke=0)

    def text(self,text,x,y,w,h,size=20,color=INK,bold=False):
        lines=wrap(text,w,size,bold)
        leading=size*1.25
        if len(lines)*leading > h+1:
            raise ValueError(f'Text does not fit ({len(lines)*leading:.1f}>{h}): {text}')
        box=self.current.shapes.add_textbox(Pt(x),Pt(y),Pt(w),Pt(h))
        frame=box.text_frame
        frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
        frame.word_wrap=False
        for i,line in enumerate(lines):
            paragraph=frame.paragraphs[0] if i==0 else frame.add_paragraph()
            paragraph.text=line
            paragraph.font.name='Arial';paragraph.font.size=Pt(size);paragraph.font.bold=bold
            paragraph.font.color.rgb=RGBColor.from_string(color)
            paragraph.line_spacing=Pt(leading);paragraph.space_before=Pt(0);paragraph.space_after=Pt(0)
        self.pdf.setFont('DeckBold' if bold else 'Deck',size)
        self.pdf.setFillColorRGB(*[int(color[i:i+2],16)/255 for i in (0,2,4)])
        for i,line in enumerate(lines): self.pdf.drawString(x,H-y-size-i*leading,line)
        self.bounds.append({'text':text,'x':x,'y':y,'w':w,'h':h})

    def image(self,path,x,y,w,h):
        with Image.open(path) as source:
            iw,ih=source.size
        scale=min(w/iw,h/ih);dw,dh=iw*scale,ih*scale
        x+=(w-dw)/2;y+=(h-dh)/2
        self.current.shapes.add_picture(str(path),Pt(x),Pt(y),width=Pt(dw),height=Pt(dh))
        self.pdf.drawImage(str(path),x,H-y-dh,dw,dh,mask='auto')

    def cards(self,cards,y=222,height=232):
        gap=18;width=(880-gap*(len(cards)-1))/len(cards)
        for i,(title,body) in enumerate(cards):
            x=40+i*(width+gap)
            self.rect(x,y,width,height,'FFFFFF')
            self.rect(x,y,5,height,GREEN)
            self.text(title,x+20,y+20,width-40,70,22,INK,True)
            self.text(body,x+20,y+102,width-40,height-112,18,MUTED)

    def end(self): self.pdf.showPage()

    def save(self):
        self.ppt.save(OUT/'Phase-4-Presentation.pptx');self.pdf.save()


def main():
    content=json.loads((OUT/'slides.json').read_text())
    deck=Deck();notes=['# Phase 4 speaker notes','',content['timing_note'],'',
        'These are suggested handoffs, not confirmed commitments. Agree on speakers and rehearse together.',
        '', 'For ~5 minutes: shorten slides 2–5, demonstrate examples 1 and 6 only, and summarize slides 8–9. For more time: discuss the full demo and questions. Confirm the actual time limit first.','']
    for number,item in enumerate(content['slides'],1):
        deck.slide(number,item,len(content['slides']))
        kind=item['kind']
        if kind=='pipeline':
            for i,(title,count,detail) in enumerate(item['steps']):
                x=40+i*225
                deck.rect(x,235,205,204,'FFFFFF')
                deck.text(title,x+16,255,173,40,20,INK,True)
                deck.text(count,x+16,313,173,55,35,GREEN,True)
                deck.text(detail,x+16,385,173,45,14,MUTED)
            deck.text('Development: train + validate',40,465,420,25,15,GREEN,True)
            deck.text('Final evaluation: two separate collections',490,465,430,25,15,GREEN,True)
        elif kind=='results':
            # The measured plot is reused byte-for-byte, not redrawn with invented data.
            deck.image(OUT/item['image'],40,194,880,305)
        elif kind=='references':
            for i,reference in enumerate(item['refs']):
                deck.text(reference,40,215+i*68,880,63,14,MUTED)
        elif kind!='title':
            deck.cards(item['cards'],y=210,height=227 if len(item['cards'])==3 else 212)
            if item.get('bottom'): deck.text(item['bottom'],40,450,880,48,16,GREEN,True)
        deck.end()
        notes += [f"## {number}. {item['title'].replace(chr(10),' ')}",'',
            f"Suggested speaker: **{item['presenter']}**. Suggested time: {item['seconds']} seconds." if item['seconds'] else 'Backup slide; use for questions.',
            '',item['notes'],'']
    deck.save()
    (OUT/'speaker-notes.md').write_text('\n'.join(notes),encoding='utf-8')
    print(f'Built {len(content["slides"])} editable slides, PDF companion, and notes.')


if __name__=='__main__': main()
