"""Prepare credited presentation illustrations from existing local evidence.

Run with --data-root and --replay pointing to the original downloaded images and
verified demo replay JSON. This only resizes/orients real photos and copies real
captured model displays; it never trains, downloads or synthesizes photographs.
The small presentation derivatives are distinct from the raw datasets.
"""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/presentation/assets'
SOURCES = {
    'plantvillage': {'creator':'S. P. Mohanty / PlantVillage', 'license':'CC BY-SA 3.0',
        'license_url':'https://creativecommons.org/licenses/by-sa/3.0/',
        'revision':'9e97599868962bd0079b8db4b7f1efa9185fa1e7',
        'url':'https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7'},
    'plantdoc': {'creator':'D. Singh et al. / cropped repository by P. Kayal et al.', 'license':'CC BY 4.0',
        'license_url':'https://creativecommons.org/licenses/by/4.0/',
        'revision':'5467f6012d78d1c446145d5f582da6096f852ae8',
        'url':'https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8'}
}

def digest(raw): return hashlib.sha256(raw).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path.home()/'datasets/cecs551-plant-disease')
    parser.add_argument('--replay',type=Path,default=ROOT/'outputs/demo-visual-cleanup/replay/replay.json')
    args=parser.parse_args()
    replay=json.loads(args.replay.read_text())
    examples=json.loads((ROOT/'demo/examples.json').read_text())
    # Use the same documented examples; the grape pair illustrates another class.
    choices=[('pv-apple-01',0),('pv-apple-02',1),('pd-apple-01',3),('pd-apple-02',5)]
    records={name:dict(replay['records'][i],sha256=examples['examples'][i]['sha256']) for name,i in choices}
    for dataset,prefix in [('plantvillage','pv'),('plantdoc','pd')]:
        r=replay['batch'][5]['records'][dataset]
        records[f'{prefix}-grape-01']=dict(r,sha256=r['image_sha256'])
    OUT.mkdir(exist_ok=True)
    manifest={'selection':'Illustrative examples selected after evaluation; not representative of accuracy.',
              'original_commit':examples['original_commit'],'seed':551,'sources':SOURCES,'assets':{}}
    for name,r in records.items():
        raw=(args.data_root/r['source_path']).read_bytes()
        if digest(raw)!=r['sha256']: raise ValueError(f'Changed photograph: {name}')
        photo=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
        original_size=list(photo.size);photo.thumbnail((900,700))
        path=OUT/f'{name}.jpg';photo.save(path,quality=94,subsampling=0)
        manifest['assets'][name]={'file':path.name,'dataset':r['dataset'],'label':r['label'],
            'source_path':r['source_path'],'source_sha256':r['sha256'],
            'original_size':original_size,'display_size':list(photo.size),
            'sha256':digest(path.read_bytes()),'changes':'EXIF orientation, RGB, fit without cropping, JPEG encoding.',
            'predictions':{mode:{'label':p['label'],'matches_label':p['matches_label']} for mode,p in r['predictions'].items()}}
    # The classroom process diagram uses an actual padded input and actual maps.
    r=replay['records'][5]
    derived={'prepared-apple':r['prepared'],
             'late-frozen':r['features']['frozen'][2]['maps'][0],
             'late-finetune':r['features']['finetune'][2]['maps'][0]}
    for name,url in derived.items():
        raw=base64.b64decode(url.split(',',1)[1]);path=OUT/f'{name}.png';path.write_bytes(raw)
        manifest['assets'][name]={'file':path.name,'dataset':'plantdoc','source_photo':'pd-apple-02',
            'sha256':digest(raw),'changes':'Original preprocessing display' if name.startswith('prepared') else
            'Actual seed-551 final-block channel 0; shared color scale; not an attention map.',
            'replay_sha256':digest(args.replay.read_bytes())}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    lines=['# Photograph and visual credits','',
        'These nine small presentation illustrations use real dataset photos and recorded model outputs. No AI-generated photographs are used.',
        'Photos retain the full frame: orientation correction, size reduction and JPEG encoding only. The model-input and feature displays are identified separately.',
        '','## Dataset credits','']
    for dataset,s in SOURCES.items():
        lines += [f"- **{dataset}:** {s['creator']}. [{s['license']}]({s['license_url']}). [Pinned source]({s['url']})."]
    lines += ['','The photo derivatives retain their source licenses. The dataset declarations do not independently establish the original provenance of every internet-sourced PlantDoc photograph. Visible image credits are preserved.',
              '','## Illustrations','', '| ID | Dataset label / display | File |', '| --- | --- | --- |']
    for name,a in manifest['assets'].items():lines.append(f"| {name} | {a.get('label',a['changes'])} | [{a['file']}](assets/{a['file']}) |")
    lines += ['', 'Exact relative source paths, original checksums, derivative checksums, dimensions and dataset revisions are in [assets/manifest.json](assets/manifest.json). Photo IDs appear in slide footers; notes give full source paths. The process diagram uses the original seed-551 CPU replay. Predicted category names agree with the original saved evaluation. Examples were selected after testing and are not an accuracy estimate.',
              '', 'To recreate these derivatives from the existing local evidence:', '', '```bash', 'python scripts/prepare_presentation_assets.py --data-root /path/to/datasets --replay /path/to/replay.json', '```',
              '', 'The deck builder uses these checked-in presentation derivatives; it does not require raw datasets or model weights.']
    (OUT.parent/'PHOTO_CREDITS.md').write_text('\n'.join(lines)+'\n')
    print(f'Prepared {len(manifest["assets"])} credited illustrations.')

if __name__=='__main__': main()
