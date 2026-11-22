"""Run the same practical checks locally and on GitHub (no research data needed).

From the repo root: python scripts/check_project.py
Install requirements-checks*.txt, Node.js and Playwright Chromium first; see
CONTRIBUTING.md. This never commits, downloads data, trains research models or
opens a visible browser. A failing step prints its name and exits unsuccessfully.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_files(root=ROOT):
    # Include new, unignored local files so checks are useful before committing.
    result = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=root)
    return sorted(set(result.decode('utf-8').split('\0')) - {''})


def private_file_reason(name, size):
    path = PurePosixPath(name.lower())
    parts = set(path.parts)
    if parts & {'data', 'datasets', 'raw', 'weights', 'checkpoints', 'outputs', '.venv', 'venv', '__pycache__'}:
        return 'Local data, environments and generated outputs belong outside Git'
    if path.suffix in {'.pt', '.pth', '.ckpt', '.onnx', '.safetensors', '.h5', '.hdf5', '.pkl', '.pickle', '.bin', '.npy', '.npz', '.zip', '.tar', '.gz', '.7z'}:
        return 'Model weights, serialized objects and data archives belong outside Git'
    if path.name in {'agents.local.md', 'walkthrough.md', 'demo-settings.local.json', 'dataset-paths.local.txt'}:
        return 'Personal settings/notes belong outside Git'
    if path.name == '.env' or (path.name.startswith('.env.') and path.name != '.env.example'):
        return 'Environment secrets belong outside Git'
    compact = re.sub(r'[^a-z]', '', name.lower())
    if any(term in compact for term in ('teamevaluation', 'teammemberevaluation', 'teammemberchecklist', 'teammemberassessment', 'personalratings')):
        return 'Personal assessment documents belong outside Git'
    if size > 10 * 1024 * 1024:
        return 'Files over 10 MiB need a reviewed distribution plan, not an accidental commit'
    return None


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    def invalid(value):
        raise ValueError(f'Invalid JSON number: {value}')
    return json.loads(path.read_text(encoding='utf-8-sig'),
                      object_pairs_hook=reject_duplicate_keys, parse_constant=invalid)


def same_copy(root, source, destination, text=False):
    # Windows may use CRLF for Markdown; compare decoded text in that case.
    a, b = root / source, root / destination
    left = a.read_text(encoding='utf-8') if text else a.read_bytes()
    right = b.read_text(encoding='utf-8') if text else b.read_bytes()
    require(left == right, f'{destination} is out of date. Refresh it from {source}.')


def check_files(root=ROOT):
    names = project_files(root)
    assets = read_json(root / 'docs/presentation/assets/manifest.json')['assets']
    allowed_photos = {'docs/presentation/assets/' + a['file'] for a in assets.values()}
    for name in names:
        path = root / name
        require(path.is_file() and not path.is_symlink(), f'Missing or symlinked tracked file: {name}')
        reason = private_file_reason(name, path.stat().st_size)
        require(reason is None, f'{name}: {reason}')
        # Existing analysis charts are allowed. JPEG photographs need provenance.
        if path.suffix.lower() in {'.jpg', '.jpeg', '.webp', '.bmp', '.tiff'}:
            require(name in allowed_photos, f'{name}: dataset photos need a reviewed presentation asset/credit entry; keep raw photos local.')
        if path.suffix.lower() == '.png':
            require(name.startswith('docs/') or name in allowed_photos,
                    f'{name}: put documented report figures under docs/; keep raw images local.')
        if path.suffix == '.py':
            # Compile without executing: catches invalid syntax and contexts such
            # as a return outside a function, without writing bytecode caches.
            compile(path.read_bytes(), name, "exec", dont_inherit=True)
        if path.suffix in {'.json', '.ipynb'}:
            value = read_json(path)
            if path.suffix == '.ipynb':
                require(value.get('nbformat') == 4 and isinstance(value.get('cells'), list),
                        f'{name}: invalid notebook structure')
                for cell in value['cells']:
                    require(cell.get('cell_type') in {'code', 'markdown', 'raw'} and
                            isinstance(cell.get('source'), (str, list)), f'{name}: invalid notebook cell')
    for asset in assets.values():
        path = root / 'docs/presentation/assets' / asset['file']
        require(path.resolve().is_relative_to((root/'docs/presentation/assets').resolve()), 'Asset escaped its folder')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256'], f'Changed presentation asset: {path.name}')
    same_copy(root, 'docs/phase4-report/Phase-4-Plant-Disease-Classification.pdf',
              'Phase-4-Submission/Phase-4-Plant-Disease-Classification.pdf')
    for name in ['Phase-4-Presentation.pptx', 'Phase-4-Presentation.pdf', 'speaker-notes.md']:
        same_copy(root, 'docs/presentation/' + name, 'Phase-4-Submission/presentation/' + name, text=name.endswith('.md'))
    credits = (root/'docs/presentation/PHOTO_CREDITS.md').read_text(encoding='utf-8').replace(
        '](assets/', '](https://github.com/EnragedWaters/CECS-551-Plant-Disease-Classification/blob/main/docs/presentation/assets/')
    require(credits == (root/'Phase-4-Submission/presentation/PHOTO_CREDITS.md').read_text(encoding='utf-8'),
            'Submission PHOTO_CREDITS.md differs from its source (use repository links for assets).')
    require((root/'Phase-4-Submission/repository-link.txt').read_text().strip() ==
            'https://github.com/EnragedWaters/CECS-551-Plant-Disease-Classification', 'Submission TXT must contain only the repository URL.')
    return names


def check_presentation_sources(root=ROOT):
    """Catch changed slide wording/notes that have not reached the exported deck.

    This is a text consistency check, not a replacement for rendered-page review.
    It uses only committed files and does not require Office or original photos.
    """
    from pptx import Presentation
    folder = root / 'docs/presentation'
    source = read_json(folder / 'slides.json')
    slides = source['slides']
    deck = Presentation(folder / 'Phase-4-Presentation.pptx')
    notes = (folder / 'speaker-notes.md').read_text(encoding='utf-8')
    normalize = lambda text: ' '.join(text.split())
    require(slides and len(deck.slides) == len(slides),
            'Presentation source and PPTX slide counts differ; rebuild and review the exports.')
    require(normalize(source['timing_note']) in normalize(notes), 'Speaker notes have an outdated timing note.')
    for index, (item, slide) in enumerate(zip(slides, deck.slides), 1):
        require(all(item.get(key) for key in ('title', 'notes', 'presenter')),
                f'Slide {index} needs a title, notes and speaking portion.')
        visible = normalize(' '.join(shape.text for shape in slide.shapes if shape.has_text_frame))
        for text in [item['title'], item.get('subtitle', ''), *item.get('refs', [])]:
            require(normalize(text) in visible, f'Slide {index} wording changed; rebuild and review the exports.')
        recorded = normalize(slide.notes_slide.notes_text_frame.text) if slide.has_notes_slide else ''
        for text in [item['notes'], f"Suggested speaker: {item['presenter']}",
                     f"Suggested time: {item['seconds']} seconds (provisional)"]:
            require(normalize(text) in recorded, f'Slide {index} PPTX notes changed; rebuild the exports.')
        # Compare the matching Markdown section, so moving a note to another
        # speaker's slide cannot accidentally satisfy this check.
        heading = f"## {index}. {item['title'].replace(chr(10), ' ')}"
        section = notes.partition(heading)[2].split('\n## ', 1)[0]
        require(section and normalize(item['notes']) in normalize(section)
                and f"**{item['presenter']}** · {item['seconds']} seconds" in section,
                f'Slide {index} Markdown speaker notes are outdated; rebuild the exports.')


def check_configuration():
    import yaml
    from scripts.build_splits import category_lookup
    categories = read_json(ROOT/'configs/categories.json')
    category_lookup(categories)
    selection = read_json(ROOT/'docs/training-runs/selected-checkpoints.json')
    evidence = read_json(ROOT/'docs/evaluation/final-results.json')
    demo = read_json(ROOT/'demo/examples.json')
    for path, expected in [
        ('configs/categories.json', selection['categories_sha256']),
        ('docs/training-runs/selected-checkpoints.json', evidence['metadata']['selection_sha256']),
        ('docs/evaluation/final-results.json', demo['results_sha256'])]:
        require(hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected,
                f'{path} no longer matches the saved demo evidence. Preserve originals; review new experiments separately.')
    for example in demo['examples']:
        require(0 <= int(example['class_id']) < len(categories['classes']), 'Invalid demo class ID')
        require(example['dataset'] in {'plantvillage', 'plantdoc'}, 'Invalid demo dataset')
    # BaseLoader preserves the YAML key "on" instead of treating it as a boolean.
    for path in (ROOT/'.github/workflows').glob('*.y*ml'):
        flow = yaml.load(path.read_text(encoding='utf-8'), Loader=yaml.BaseLoader)
        require(isinstance(flow, dict) and flow.get('on') and flow.get('jobs'), f'Invalid workflow: {path.name}')


def run(label, command):
    print(f'\n--- {label} ---', flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-only', action='store_true', help='Quick file/syntax/config checks only; not sufficient for merging')
    args = parser.parse_args()
    os.environ.update(PYTHONUTF8='1', CUDA_VISIBLE_DEVICES='', MPLBACKEND='Agg', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    try:
        print('Checking files, Python syntax and submission copies...', flush=True)
        names = check_files()
        check_configuration()
        check_presentation_sources()
        require(shutil.which('node'), 'Install Node.js 22 (or compatible LTS) to check the demo JavaScript.')
        for name in names:
            if name.endswith('.js'):
                run(f'JavaScript syntax: {name}', ['node', '--check', name])
        if args.static_only:
            print('Static checks passed. Run without --static-only before merging.')
            return 0
        print('\nChecking runtime imports...', flush=True)
        for name in names:
            if name.startswith('plant_disease/') and name.endswith('.py'):
                importlib.import_module(name[:-3].replace('/', '.'))
        for module in ['scripts.launch_demo', 'scripts.analyze_data', 'scripts.build_splits',
                       'scripts.prepare_data', 'scripts.inspect_images']:
            importlib.import_module(module)
        run('Automated tests (including OS-specific launchers)', [sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
        run('Headless live and offline demo using invented fixtures', [sys.executable, 'scripts/check_demo_smoke.py'])
        print('\nPASS: project checks completed. No research datasets, checkpoints or training were used.')
        return 0
    except (ValueError, SyntaxError, ImportError, OSError, subprocess.CalledProcessError) as error:
        print(f'\nFAIL: {error}\nFix the reported problem and rerun python scripts/check_project.py.\nDo not disable a check to make a pull request pass.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
