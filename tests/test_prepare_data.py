"""Exercise archive failures with tiny invented files, never the real datasets."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.prepare_data import extract


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'data.zip'
        self.destination = self.root / 'unpacked'

    def archive_with(self, names):
        with zipfile.ZipFile(self.archive, 'w', compression=zipfile.ZIP_STORED) as stream:
            for name in names:
                stream.writestr(name, b'original archive payload')

    def extract(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return extract(self.archive, self.destination, **kwargs)

    def test_original_folder_layout_and_color_selection_are_preserved(self):
        self.archive_with(['raw/color/apple/a.jpg', 'raw/grayscale/apple/a.jpg'])
        self.assertEqual(self.extract(color_only=True), 1)
        self.assertEqual((self.destination/'raw/color/apple/a.jpg').read_bytes(), b'original archive payload')
        self.assertFalse((self.destination/'raw/grayscale').exists())
        self.archive_with(['repository-revision/train/apple/b.jpg', 'repository-revision/LICENSE'])
        self.assertEqual(self.extract(), 2)
        self.assertTrue((self.destination/'train/apple/b.jpg').is_file())
        self.assertTrue((self.destination/'LICENSE').is_file())

    def test_bad_checksum_preserves_existing_file_and_removes_temporary_file(self):
        self.archive_with(['repository/file.jpg'])
        # Change stored bytes without updating the archive's CRC checksum.
        data = self.archive.read_bytes().replace(b'original archive payload', b'corrupted archive bytes!')
        self.archive.write_bytes(data)
        self.destination.mkdir()
        target = self.destination/'file.jpg'
        target.write_bytes(b'previous good copy')
        with self.assertRaisesRegex(zipfile.BadZipFile, 'Bad CRC'):
            self.extract()
        self.assertEqual(target.read_bytes(), b'previous good copy')
        self.assertEqual(list(self.destination.iterdir()), [target])

    def test_rejects_paths_that_are_unsafe_on_either_operating_system(self):
        for name in ['../escape', '/outside', 'repository/../../escape',
                     'C:/outside', 'repository/C:/outside', 'repository/..\\escape']:
            with self.subTest(name=name):
                self.archive_with([name])
                with self.assertRaisesRegex(RuntimeError, 'Unsafe archive path'):
                    self.extract()

    def test_existing_symlink_cannot_redirect_extraction(self):
        self.destination.mkdir()
        outside = self.root/'outside'
        outside.mkdir()
        try:
            (self.destination/'train').symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f'This account cannot create symlinks: {error}')
        self.archive_with(['repository/train/file.jpg'])
        with self.assertRaisesRegex(RuntimeError, 'Unsafe extraction target'):
            self.extract()
        self.assertEqual(list(outside.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
