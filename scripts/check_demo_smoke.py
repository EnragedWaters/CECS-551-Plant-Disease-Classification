"""Exercise the real demo UI with invented images and tiny, untrained CPU models.

Only the data/model setup is replaced. The production preparation, inference,
feature capture, HTTP routes, replay exporter and browser UI run unchanged.
Fixtures and screenshots live in a temporary directory and are removed afterward.
These outputs are software checks, never project research results.
"""
import hashlib
from http.server import HTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from PIL import Image, ImageDraw
from plant_disease import data, model
from plant_disease.demo import DemoEngine, export_replay, make_handler


class FixtureModel(torch.nn.Module):
    """Small fixed network with the shapes expected by the teaching displays."""
    def __init__(self, winner):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Conv2d(3, 4, 1), torch.nn.AdaptiveAvgPool2d((7, 7))),
            *[torch.nn.ReLU() for _ in range(11)], torch.nn.Conv2d(4, 576, 1))
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.classifier = torch.nn.Linear(576, 22)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.fill_(0.01)
            self.classifier.weight.zero_()
            self.classifier.bias.zero_()
            self.classifier.bias[winner] = 2
        self.eval().requires_grad_(False)

    def forward(self, image):
        return self.classifier(self.avgpool(self.features(image)).flatten(1))


def picture(item):
    size = (80, 50) if item['dataset'] == 'plantvillage' else (40, 70)
    image = Image.new('RGB', size, (35, 90, 50))
    ImageDraw.Draw(image).rectangle((10, 10, 30, 40), fill=(180, 60 + int(item['class_id']), 20))
    return image


def fixture_engine():
    engine = DemoEngine.__new__(DemoEngine)
    engine.torch = torch
    engine.device = torch.device('cpu')
    engine.data = data
    engine.model_code = model
    engine.original_folder = None
    engine.categories = [{'plantvillage': f'Fixture___category_{i}'} for i in range(22)]
    engine.models = {m: FixtureModel(i) for i, m in enumerate(['frozen', 'finetune'])}
    engine.hashes = {m: model.state_fingerprint(v.state_dict()) for m, v in engine.models.items()}
    engine.epochs = {'frozen': 0, 'finetune': 0}
    engine.config = {'selection_policy': 'Invented software-test fixtures, not plant research evidence.'}
    def item(dataset, category):
        name = f'{dataset}/fixture-{category}.png'
        return {'dataset': dataset, 'class_id': str(category), 'relative_path': name,
                'purpose': f'Invented fixture {category}', 'sha256': hashlib.sha256(name.encode()).hexdigest()}
    engine.examples = [item(d, i) for d in ['plantvillage', 'plantdoc'] for i in range(3)]
    engine.batch = [{d: item(d, i) for d in ['plantvillage', 'plantdoc']} for i in range(22)]
    engine.load_record = picture
    return engine


def main():
    torch.set_num_threads(1)
    engine = fixture_engine()
    server = HTTPServer(('127.0.0.1', 0), make_handler(engine))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory(prefix='cecs demo checks ') as directory:
            folder = Path(directory)
            export_replay(engine, folder/'replay')
            subprocess.run([sys.executable, str(ROOT/'scripts/check_demo_browser.py'),
                '--url', f'http://127.0.0.1:{server.server_port}',
                '--replay', str(folder/'replay/offline-demo.html'),
                '--output-dir', str(folder/'browser')], cwd=ROOT, check=True, timeout=180)
        for mode, network in engine.models.items():
            if model.state_fingerprint(network.state_dict()) != engine.hashes[mode]:
                raise RuntimeError('Browser checks changed fixture model weights')
        print('PASS: both datasets, seven stages, learning controls, playback, error recovery,\n'
              'sample counts, narrow screens and offline replay; invented CPU fixtures only.')
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        engine.close()


if __name__ == '__main__':
    main()
