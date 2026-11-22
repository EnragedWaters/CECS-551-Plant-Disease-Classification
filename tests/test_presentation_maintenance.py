"""Check portable font discovery and catch stale decks using committed examples."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
from reportlab.pdfbase import pdfmetrics

from scripts import build_presentation
from scripts.check_project import check_presentation_sources

ROOT = Path(__file__).resolve().parents[1]


class PresentationMaintenanceTests(unittest.TestCase):
    def test_fonts_work_without_a_linux_system_font_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(build_presentation, 'FONT_DIR', Path(directory)):
                chosen = build_presentation.register_fonts()
        self.assertEqual(chosen, Path(matplotlib.get_data_path())/'fonts/ttf')
        self.assertGreater(pdfmetrics.stringWidth('Plant disease', 'Deck', 20), 0)
        self.assertGreater(pdfmetrics.stringWidth('Plant disease', 'DeckBold', 20), 0)

    def test_existing_linux_fonts_remain_first_choice(self):
        # Use bundled fonts as a stand-in system directory on either OS.
        directory = Path(matplotlib.get_data_path())/'fonts/ttf'
        with patch.object(build_presentation, 'FONT_DIR', directory):
            self.assertEqual(build_presentation.register_fonts(), directory)

    def test_complete_deck_builds_with_bundled_fonts_in_a_separate_folder(self):
        # This checks authoring on native Windows too; it never replaces the
        # published deck, paper, or local research outputs.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'docs/presentation'
            shutil.copytree(ROOT/'docs/presentation/assets', folder/'assets')
            for name in ['docs/presentation/slides.json', 'docs/evaluation/final-results.json',
                         'demo/examples.json']:
                target = root/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT/name, target)
            with patch.object(build_presentation, 'ROOT', root), \
                    patch.object(build_presentation, 'OUT', folder), \
                    patch.object(build_presentation, 'FONT_DIR', root/'no-system-fonts'), \
                    patch('sys.argv', ['build_presentation.py']):
                build_presentation.main()
            check_presentation_sources(root)
            self.assertTrue((folder/'Phase-4-Presentation.pdf').read_bytes().startswith(b'%PDF'))

    def test_source_edits_require_updated_pptx_and_speaker_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'docs/presentation'
            folder.mkdir(parents=True)
            for filename in ('slides.json', 'Phase-4-Presentation.pptx', 'speaker-notes.md'):
                shutil.copyfile(ROOT/'docs/presentation'/filename, folder/filename)
            check_presentation_sources(root)
            original = (folder/'slides.json').read_text(encoding='utf-8')
            for key, value in [('title', 'A new title absent from the export'),
                               ('subtitle', 'Changed subtitle'), ('notes', 'Changed explanation'),
                               ('presenter', 'Someone else'), ('seconds', 999)]:
                source = json.loads(original)
                source['slides'][0][key] = value
                (folder/'slides.json').write_text(json.dumps(source), encoding='utf-8')
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Slide 1'):
                    check_presentation_sources(root)
            source = json.loads(original)
            source['slides'].pop()
            (folder/'slides.json').write_text(json.dumps(source), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'slide counts differ'):
                check_presentation_sources(root)
            (folder/'slides.json').write_text(original, encoding='utf-8')
            (folder/'speaker-notes.md').write_text('outdated', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'timing note'):
                check_presentation_sources(root)


if __name__ == '__main__':
    unittest.main()
