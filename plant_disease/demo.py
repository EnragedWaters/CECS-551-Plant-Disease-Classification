"""Visual, local-only presentation of the original saved model pair.

Run: python -m plant_disease.demo
No training, downloads, or changes to the paper/results. The original model and
image-preparation code is read from Git, verified against checkpoint metadata,
and loaded in a temporary package. Later experiment code is never substituted.
Use --export DIR to make a portable, clearly labeled recorded replay.
"""

import argparse
import base64
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_COMMIT = '9619975a2f64954bc25d056708962eea1895d84b'
CORE_FILES = ('__init__.py', 'data.py', 'model.py', 'train.py')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def checked_path(root, relative):
    """Only allow manifest-relative paths inside the chosen local data folder."""
    name = PurePosixPath(relative)
    root = Path(root).resolve()
    if name.is_absolute() or '..' in name.parts or '\\' in relative:
        raise ValueError('Unsafe relative path in example list')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Example resolves outside the dataset folder')
    return path


def original_package(expected, repository=ROOT):
    """Read and verify historical code without switching or editing the checkout."""
    folder = tempfile.TemporaryDirectory(prefix='plant-demo-')
    package = Path(folder.name) / '_phase4_demo_original'
    package.mkdir()
    try:
        for name in CORE_FILES:
            result = subprocess.run(['git', '-C', str(repository), 'show',
                f'{ORIGINAL_COMMIT}:plant_disease/{name}'], capture_output=True, check=True)
            if hashlib.sha256(result.stdout).hexdigest() != expected[name]:
                raise ValueError(f'Original code does not match the saved run: {name}')
            (package / name).write_bytes(result.stdout)
        spec = importlib.util.spec_from_file_location('_phase4_demo_original', package / '__init__.py')
        # Replace only our own isolated namespace if another engine ran previously.
        for key in list(sys.modules):
            if key == '_phase4_demo_original' or key.startswith('_phase4_demo_original.'):
                del sys.modules[key]
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        data = importlib.import_module(spec.name + '.data')
        model = importlib.import_module(spec.name + '.model')
        return folder, data, model
    except Exception:
        folder.cleanup()
        raise


def label_name(category):
    return category['plantvillage'].replace('___', ' / ').replace('_', ' ')


def image_url(image):
    """Embed a display copy; the original dataset image is never overwritten."""
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode('ascii')


class DemoEngine:
    def __init__(self, data_root, runs_root, splits_dir, device='cpu'):
        import torch
        self.torch = torch
        # A single image is cheap on CPU; this avoids requiring a GPU in class.
        torch.set_num_threads(2)
        self.device = torch.device(device)
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; use --device cpu')
        self.data_root = Path(data_root)
        self.config = read_json(ROOT / 'demo/examples.json')
        evidence_path = ROOT / 'docs/evaluation/final-results.json'
        if digest(evidence_path) != self.config['results_sha256']:
            raise ValueError('Original results changed; review the demonstration evidence')
        evidence = read_json(evidence_path)
        selection_path = ROOT / 'docs/training-runs/selected-checkpoints.json'
        if digest(selection_path) != evidence['metadata']['selection_sha256']:
            raise ValueError('Selection record does not match the published experiment')
        selection = read_json(selection_path)
        categories_path = ROOT / 'configs/categories.json'
        if digest(categories_path) != selection['categories_sha256']:
            raise ValueError('Category mapping differs from the original experiment')
        self.categories = read_json(categories_path)['classes']
        if self.config['seed'] != 551 or self.config['original_commit'] != ORIGINAL_COMMIT:
            raise ValueError('This demonstration is fixed to the original seed-551 pair')
        self.examples = self.config['examples']
        self.validate_examples(splits_dir, selection)
        chosen = [r for r in selection['checkpoints'] if r['seed'] == 551]
        self.models = {}
        self.hashes = {}
        self.epochs = {}
        self.original_folder = None
        try:
            for record in chosen:
                path = checked_path(runs_root, record['run_name'] + '/' + record['checkpoint'])
                if digest(path) != record['sha256']:
                    raise ValueError(f"Saved model changed: {record['mode']}")
                saved = torch.load(path, map_location='cpu', weights_only=True)
                if (saved['trial'] or saved['mode'] != record['mode']
                        or saved['epoch'] != record['epoch'] or saved['class_count'] != len(self.categories)
                        or saved['metadata']['arguments']['seed'] != 551
                        or saved['metadata']['categories'] != self.categories):
                    raise ValueError('Saved model metadata does not match the selected model')
                if self.original_folder is None:
                    self.code_hashes = saved['metadata']['code_sha256']
                    self.original_folder, self.data, self.model_code = original_package(self.code_hashes)
                if any(saved['metadata']['code_sha256'][n] != self.code_hashes[n] for n in CORE_FILES):
                    raise ValueError('The two models used different original code')
                model = self.model_code.build_model(len(self.categories),
                    self.data_root / 'weights/mobilenet_v3_small-047dcff4.pth', 551)
                model.load_state_dict(saved['model_state'], strict=True)
                model.to(self.device).eval()
                model.requires_grad_(False)
                self.models[record['mode']] = model
                self.hashes[record['mode']] = record['sha256']
                self.epochs[record['mode']] = record['epoch']
            if set(self.models) != {'frozen', 'finetune'}:
                raise ValueError('Both selected models are required')
        except Exception:
            self.close()
            raise

    def close(self):
        if self.original_folder is not None:
            self.original_folder.cleanup()

    def validate_examples(self, splits_dir, selection):
        rows = {}
        for split in ('source_test', 'target_test'):
            path = Path(splits_dir) / f'{split}.csv'
            if digest(path) != selection['split_manifest_sha256'][path.name]:
                raise ValueError(f'Changed test image list: {split}')
            with path.open(encoding='utf-8', newline='') as stream:
                rows[split] = {r['relative_path']: r for r in csv.DictReader(stream)}
        for example in self.examples:
            original = rows.get(example['split'], {}).get(example['relative_path'])
            if not original or any(example[k] != original[k] for k in
                    ('sha256', 'class_id', 'dataset', 'split')):
                raise ValueError('Demo image does not match the original test manifest')
            if not 0 <= int(example['class_id']) < len(self.categories):
                raise ValueError('Unknown example category')

    def info(self):
        return {'mode': 'live', 'seed': 551, 'device': str(self.device),
            'original_commit': ORIGINAL_COMMIT, 'count': len(self.examples),
            'selection_policy': self.config['selection_policy'], 'checkpoint_sha256': self.hashes,
            'examples': [{'dataset': e['dataset'], 'purpose': e['purpose']} for e in self.examples]}

    def load_image(self, index):
        from PIL import Image
        if not 0 <= index < len(self.examples):
            raise IndexError('Unknown demonstration image')
        example = self.examples[index]
        path = checked_path(self.data_root, example['relative_path'])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != example['sha256']:
            raise ValueError('Image contents changed since the experiment')
        with Image.open(io.BytesIO(raw)) as image:
            return image.copy()

    def example(self, index):
        image = self.load_image(index)
        item = self.examples[index]
        from PIL import ImageOps
        original = ImageOps.exif_transpose(image).convert("RGB")
        original.thumbnail((850, 650))
        return {'index': index, 'dataset': item['dataset'], 'purpose': item['purpose'],
            'label': label_name(self.categories[int(item['class_id'])]),
            'source_path': item['relative_path'], 'original_size': list(image.size),
            'original': image_url(original), 'prepared': image_url(self.data.resize_with_padding(image))}

    def predict(self, index):
        image = self.load_image(index)
        tensor = self.data.prepare_image(image).unsqueeze(0).to(self.device)
        predictions = {}
        for mode, model in self.models.items():
            before = self.model_code.state_fingerprint(model.state_dict())
            start = time.perf_counter()
            with self.torch.inference_mode():
                logits = model(tensor)
                if logits.shape != (1, len(self.categories)) or not self.torch.isfinite(logits).all():
                    raise ValueError('Invalid model output')
                scores = logits.softmax(1)[0]
                guess = int(scores.argmax().item())
                score = float(scores[guess].item())
            if self.model_code.state_fingerprint(model.state_dict()) != before:
                raise RuntimeError('Inference unexpectedly changed the saved model state')
            item = self.examples[index]
            recorded = item['recorded'][mode]
            predictions[mode] = {'label': label_name(self.categories[guess]), 'class_id': guess,
                'score': score, 'matches_label': guess == int(item['class_id']),
                'matches_recorded_prediction': guess == int(recorded['predicted_class_id']),
                'recorded_score_difference': abs(score - float(recorded['confidence'])),
                'epoch': self.epochs[mode], 'milliseconds': round((time.perf_counter() - start) * 1000, 1)}
        return {'index': index, 'predictions': predictions, 'mode': 'live'}


def export_replay(engine, destination):
    """Save local images and genuine model outputs for an explicitly offline replay."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Choose a new replay folder; existing backups are preserved')
    records = [dict(engine.example(i), **engine.predict(i)) for i in range(len(engine.examples))]
    destination.mkdir(parents=True, exist_ok=False)
    payload = {'info': dict(engine.info(), mode='recorded'), 'records': records}
    # A self-contained HTML file works from a flash drive with no server/network.
    page = (ROOT / 'demo/index.html').read_text()
    page = page.replace('<link rel="stylesheet" href="/assets/demo.css">',
                        '<style>' + (ROOT / 'demo/demo.css').read_text() + '</style>')
    data = json.dumps(payload).replace('<', '\\u003c')
    page = page.replace('<script src="/assets/demo.js" defer></script>',
        '<script>window.RECORDED_DEMO=' + data + ';</script><script defer>'
        + (ROOT / 'demo/demo.js').read_text() + '</script>')
    (destination / 'offline-demo.html').write_text(page, encoding='utf-8')
    (destination / 'replay.json').write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    shutil.copyfile(ROOT / 'demo/ATTRIBUTION.md', destination / 'ATTRIBUTION.md')
    return records


def make_handler(engine):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            # Only exact routes are exposed; no arbitrary local file serving.
            route = urlparse(self.path).path
            try:
                if self.headers.get('Host') not in (f'127.0.0.1:{self.server.server_port}',
                                                   f'localhost:{self.server.server_port}'):
                    self.send_error(403, 'Open the local URL printed in the terminal')
                    return
                if route in ('/', '/assets/demo.css', '/assets/demo.js'):
                    name = 'index.html' if route == '/' else route.rsplit('/', 1)[1]
                    content = (ROOT / 'demo' / name).read_bytes()
                    mime = {'html': 'text/html', 'css': 'text/css', 'js': 'text/javascript'}[name.split('.')[-1]]
                elif route == '/api/info':
                    content, mime = json.dumps(engine.info()).encode(), 'application/json'
                elif route.startswith('/api/example/') or route.startswith('/api/predict/'):
                    index = int(route.rsplit('/', 1)[1])
                    if not 0 <= index < len(engine.examples):
                        raise IndexError('Unknown demonstration image')
                    value = engine.example(index) if route.startswith('/api/example/') else engine.predict(index)
                    content, mime = json.dumps(value).encode(), 'application/json'
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', mime + '; charset=utf-8')
                self.send_header('Content-Length', str(len(content)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.end_headers()
                self.wfile.write(content)
            except (ValueError, IndexError):
                self.send_error(400, 'Invalid example or changed experiment file; see the terminal')
            except Exception as error:
                print(f'Demo request failed: {error}', file=sys.stderr)
                self.send_error(500, 'Demo could not run; use the offline backup and check the terminal')

        def log_message(self, format, *args):
            pass
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path.home() / 'datasets/cecs551-plant-disease')
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'outputs/runs')
    parser.add_argument('--splits-dir', type=Path, default=ROOT / 'outputs/splits')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--export', type=Path, help='Create a NEW local replay folder instead of serving')
    args = parser.parse_args()
    engine = None
    try:
        engine = DemoEngine(args.data_root, args.runs_root, args.splits_dir, args.device)
        if args.export:
            records = export_replay(engine, args.export)
            print(f'Saved {len(records)} illustrative examples to {args.export / "offline-demo.html"}')
        else:
            server = HTTPServer(('127.0.0.1', args.port), make_handler(engine))
            print(f'Open http://localhost:{args.port} (Ctrl+C stops the demo). CPU works; no training.', flush=True)
            try:
                server.serve_forever()
            finally:
                server.server_close()
    except KeyboardInterrupt:
        print('\nDemo stopped.')
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'Demo could not start: {error}\nSee docs/presentation/demo-guide.md. '
                    'Saved models, original splits, data and Git history must exist locally.\n')
    finally:
        if engine is not None:
            engine.close()


if __name__ == '__main__':
    main()
