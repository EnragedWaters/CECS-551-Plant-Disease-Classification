"""Check demo boundaries and inference without real datasets or saved models."""
import hashlib
import json
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import torch
from PIL import Image
from plant_disease.demo import (DemoEngine, checked_path, export_replay,
                                make_handler, original_package)
from plant_disease.model import state_fingerprint


class DemoTests(unittest.TestCase):
    def test_paths_cannot_escape_dataset_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in ('../secret', '/etc/passwd', 'a/../../secret', 'a\\b'):
                with self.assertRaises(ValueError):
                    checked_path(root, path)
            (root / 'escape').symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                checked_path(root, 'escape/secret')
            self.assertEqual(checked_path(root, 'plantdoc/test/photo.jpg'), root / 'plantdoc/test/photo.jpg')

    def test_changed_historical_code_is_rejected_before_import(self):
        with patch('plant_disease.demo.subprocess.run', return_value=SimpleNamespace(stdout=b'changed')):
            with self.assertRaisesRegex(ValueError, 'Original code'):
                original_package({'__init__.py': 'not-the-hash'})

    def test_changed_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'leaf.png'
            Image.new('RGB', (4, 6)).save(path)
            engine = DemoEngine.__new__(DemoEngine)
            engine.data_root = Path(directory)
            engine.examples = [{'relative_path':'leaf.png', 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}]
            self.assertEqual(engine.load_image(0).size, (4, 6))
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'Image contents changed'):
                engine.load_image(0)
            with self.assertRaises(IndexError):
                engine.load_image(-1)

    def test_predictions_are_real_and_do_not_update_model_state(self):
        engine = DemoEngine.__new__(DemoEngine)
        engine.torch = torch
        engine.device = torch.device('cpu')
        engine.load_image = lambda index: None
        engine.data = SimpleNamespace(prepare_image=lambda image: torch.tensor([1., 2.]))
        engine.model_code = SimpleNamespace(state_fingerprint=state_fingerprint)
        engine.categories = [{'plantvillage':'A___healthy'}, {'plantvillage':'B___spot'}]
        engine.examples = [{'class_id':'1', 'recorded': {k:{'predicted_class_id':'1','confidence':'0.9'} for k in ('frozen','finetune')}}]
        engine.models = {}
        engine.epochs = {'frozen':9,'finetune':5}
        for mode in engine.epochs:
            model = torch.nn.Linear(2,2)
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[0.,0.],[1.,1.]]))
                model.bias.zero_()
            engine.models[mode] = model.eval()
        before = {k:state_fingerprint(m.state_dict()) for k,m in engine.models.items()}
        result = engine.predict(0)
        for mode,prediction in result['predictions'].items():
            self.assertEqual(prediction['class_id'], 1)
            self.assertTrue(prediction['matches_label'])
            self.assertAlmostEqual(prediction['score'], float(torch.tensor([0.,3.]).softmax(0)[1]))
            self.assertEqual(before[mode],state_fingerprint(engine.models[mode].state_dict()))
            self.assertTrue(all(p.grad is None for p in engine.models[mode].parameters()))

    def test_replay_is_labeled_embedded_and_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = SimpleNamespace(examples=[{}], info=lambda:{'mode':'live'},
                example=lambda i:{'index':i,'label':'</script>'},
                predict=lambda i:{'index':i,'predictions':{}})
            destination = Path(directory)/'replay'
            export_replay(engine, destination)
            text = (destination/'offline-demo.html').read_text()
            self.assertIn('window.RECORDED_DEMO=', text)
            self.assertIn('\\u003c/script>', text)
            self.assertNotIn('src="/assets/demo.js"', text)
            self.assertEqual(json.loads((destination/'replay.json').read_text())['info']['mode'],'recorded')
            with self.assertRaises(FileExistsError):
                export_replay(engine, destination)

    def test_http_serves_only_known_routes_and_rejects_bad_indexes(self):
        engine = SimpleNamespace(examples=[{}], info=lambda:{'count':1}, example=lambda i:{'index':i})
        server = HTTPServer(('127.0.0.1',0),make_handler(engine))
        thread = threading.Thread(target=server.serve_forever,daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/api/info') as response:
                self.assertEqual(json.load(response), {'count':1})
            for route,code in [('/api/example/-1',400),('/api/example/2',400),('/api/example/no',400),('/.env',404),('/../AGENTS.local.md',404)]:
                with self.assertRaises(HTTPError) as raised:
                    urlopen(base+route)
                self.assertEqual(raised.exception.code,code)
            with self.assertRaises(HTTPError) as raised:
                urlopen(Request(base+'/api/info',headers={'Host':'external.example'}))
            self.assertEqual(raised.exception.code,403)
        finally:
            server.shutdown();server.server_close();thread.join()


if __name__ == '__main__':
    unittest.main()
