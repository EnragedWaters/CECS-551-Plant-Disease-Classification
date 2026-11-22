"""Regression checks for useful failures, without modifying tracked project files."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.check_project import private_file_reason, read_json, same_copy


class ProjectCheckTests(unittest.TestCase):
    def test_rejects_local_data_weights_and_private_documents(self):
        for path in ['data/leaf.jpg', 'other/checkpoints/best.bin', 'weights/file.bin',
                     'saved/best.pt', 'saved/model.safetensors', 'dataset.zip',
                     'WALKTHROUGH.md', 'nested/AGENTS.local.md', '.env.production',
                     'demo-settings.local.json', 'TeamMember_Checklist.docx',
                     'team-evaluation.pdf', 'Phase_4_TeamMember_Evaluation.docx', 'personal-ratings.csv']:
            with self.subTest(path=path):
                self.assertIsNotNone(private_file_reason(path, 10))
        for path in ['docs/presentation/assets/pv-apple-01.jpg', '.env.example',
                     'AGENTS.md', 'tests/test_training.py', 'docs/model-plan.md']:
            self.assertIsNone(private_file_reason(path, 10))
        self.assertIsNotNone(private_file_reason('big.bin', 11 * 1024 * 1024))

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            for content in ['{"a":1,"a":2}', '{"score":NaN}', '{"score":Infinity}']:
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_json(path)
            path.write_text('{"a":1}')
            self.assertEqual(read_json(path), {'a': 1})

    def test_stale_submission_copy_reports_the_file_to_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'original.pdf').write_bytes(b'original')
            (root/'submission.pdf').write_bytes(b'outdated')
            with self.assertRaisesRegex(ValueError, 'submission.pdf is out of date'):
                same_copy(root, 'original.pdf', 'submission.pdf')
            (root/'submission.pdf').write_bytes(b'original')
            same_copy(root, 'original.pdf', 'submission.pdf')
            (root/'original.md').write_bytes(b'line\n')
            (root/'submission.md').write_bytes(b'line\r\n')
            same_copy(root, 'original.md', 'submission.md', text=True)


if __name__ == '__main__':
    unittest.main()
