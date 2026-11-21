"""Check deck evidence and export consistency without opening a window.

Run after build_presentation.py. Uses the original saved prediction CSVs to check
example labels/outcomes, checks real image hashes, and checks PDF/PPTX text and
geometry. Visual inspection of rendered pages is still required. Writes its
report to ignored outputs/presentation-redesign/.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/presentation'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def words(text):return re.findall(r'[^\W_]+',text.casefold())
def label(c):return c['plantvillage'].replace('___',' / ').replace('_',' ')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path.home()/'datasets/cecs551-plant-disease')
    parser.add_argument('--predictions',type=Path,default=ROOT/'outputs/evaluation/final-01')
    parser.add_argument('--replay',type=Path,default=ROOT/'outputs/demo-visual-cleanup/replay/replay.json')
    args=parser.parse_args()
    manifest=json.loads((OUT/'assets/manifest.json').read_text())
    result=json.loads((ROOT/'docs/evaluation/final-results.json').read_text())
    classes=json.loads((ROOT/'configs/categories.json').read_text())['classes']
    tables={}
    for r in result['results']:
        if r['seed']!=551:continue
        path=args.predictions/r['predictions_file']
        assert sha(path)==r['predictions_sha256'],path
        with path.open(newline='') as f:tables[(r['mode'],r['split'])]={v['relative_path']:v for v in csv.DictReader(f)}
    replay=json.loads(args.replay.read_text())
    record=replay['records'][5]
    displays={'prepared-apple':record['prepared'],
              'late-frozen':record['features']['frozen'][2]['maps'][0],
              'late-finetune':record['features']['finetune'][2]['maps'][0]}
    for name,url in displays.items():
        asset=manifest['assets'][name]
        assert sha(args.replay)==asset['replay_sha256'],name
        assert hashlib.sha256(base64.b64decode(url.split(',',1)[1])).hexdigest()==asset['sha256'],name
    photo_count=0
    for name,a in manifest['assets'].items():
        assert sha(OUT/'assets'/a['file'])==a['sha256'],name
        if 'source_path' not in a:continue
        photo_count+=1
        assert sha(args.data_root/a['source_path'])==a['source_sha256'],name
        split='source_test' if a['dataset']=='plantvillage' else 'target_test'
        for mode,p in a['predictions'].items():
            row=tables[(mode,split)][a['source_path']]
            assert a['label']==label(classes[int(row['true_class_id'])]),name
            assert p['label']==label(classes[int(row['predicted_class_id'])]),name
            assert p['matches_label']==(row['correct']=='True'),name
    ppt=Presentation(OUT/'Phase-4-Presentation.pptx')
    pdf=subprocess.check_output(['pdftotext','-layout',str(OUT/'Phase-4-Presentation.pdf'),'-'],text=True)
    pages=pdf.split('\f');pages=pages[:-1] if not pages[-1].strip() else pages
    assert len(ppt.slides)==len(pages)==12
    counts=[];pictures=[]
    for i,(slide,page) in enumerate(zip(ppt.slides,pages),1):
        texts=[s.text for s in slide.shapes if s.has_text_frame]
        # Content equality catches missing text after actual office conversion.
        assert Counter(words(' '.join(texts)))==Counter(words(page)),f'PDF text differs on slide {i}'
        assert slide.has_notes_slide and len(slide.notes_slide.notes_text_frame.text)>200
        body=[s for s in slide.shapes if s.has_text_frame and s.top<500*12700]
        counts.append(len(' '.join(s.text for s in body).split()))
        assert counts[-1]<=105 if i<12 else counts[-1]<=170,(i,counts[-1])
        pics=[s for s in slide.shapes if s.shape_type==MSO_SHAPE_TYPE.PICTURE]
        pictures.append(len(pics))
        for s in slide.shapes:
            assert s.left>=0 and s.top>=0 and s.left+s.width<=ppt.slide_width+12700 and s.top+s.height<=ppt.slide_height+12700,(i,s.name)
        for s in pics:
            assert not any([s.crop_left,s.crop_right,s.crop_top,s.crop_bottom]),(i,s.name)
    assert sum(pictures)>=15 and sum(bool(n) for n in pictures)>=7
    # The full-results chart must reproduce all four saved means, with clear units.
    chart=pages[6]
    for mode in ['frozen','finetune']:
        for split in ['source_test','target_test']:
            assert f"{result['groups'][mode][split]['accuracy']['mean']*100:.2f}%" in chart
    assert 'Accuracy (%)' in chart and 'seed SD' in chart
    # Inspect actual PDF word boxes, not just intended PPTX text frames.
    bbox=subprocess.check_output(['pdftotext','-bbox',str(OUT/'Phase-4-Presentation.pdf'),'-'],text=True)
    root=ET.fromstring(bbox);ns={'x':'http://www.w3.org/1999/xhtml'}
    for page in root.findall('.//x:page',ns):
        for word in page.findall('x:word',ns):
            assert float(word.attrib['xMin'])>=0 and float(word.attrib['yMin'])>=0
            assert float(word.attrib['xMax'])<=float(page.attrib['width'])+.5
            assert float(word.attrib['yMax'])<=float(page.attrib['height'])+.5
    report={'slides':12,'real_photographs':photo_count,'derived_illustrations':len(manifest['assets'])-photo_count,
        'picture_objects_by_slide':pictures,'visible_body_words_by_slide':counts,
        'captured_processing_displays_verified':True,'original_photo_and_prediction_hashes_verified':True,'labels_match_saved_predictions':True,
        'all_slide_text_present_in_pdf':True,'pdf_words_inside_pages':True,'no_picture_cropping':True,
        'full_result_means_verified':True}
    dest=ROOT/'outputs/presentation-redesign';dest.mkdir(exist_ok=True,parents=True)
    (dest/'checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
