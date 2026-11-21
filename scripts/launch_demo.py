"""Shared launcher checks for native Windows, Linux, macOS, and WSL.

The shell launchers try local Python environments with --check, then start the
first ready one. This script never installs packages, downloads data, or trains.
Personal data paths can go in the ignored demo-settings.local.json file.
"""
import argparse
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PATH_OPTIONS = ('data_root', 'runs_root', 'splits_dir')


def settings_arguments(root=ROOT):
    """Convert optional local JSON settings to ordinary demo command arguments."""
    path = root / 'demo-settings.local.json'
    if not path.exists():
        return []
    values = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(values, dict) or set(values) - {*PATH_OPTIONS, 'port', 'device'}:
        raise ValueError('Settings must contain only data_root, runs_root, splits_dir, port, or device')
    arguments = []
    for key, value in values.items():
        if key in PATH_OPTIONS:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{key} must be a nonempty path')
            value = Path(value).expanduser()
            value = str(value if value.is_absolute() else root / value)
        elif key == 'port':
            if type(value) is not int or not 1 <= value <= 65535:
                raise ValueError('port must be an integer from 1 to 65535')
        elif value not in ('cpu', 'cuda'):
            raise ValueError('device must be cpu or cuda')
        arguments.extend(['--' + key.replace('_', '-'), str(value)])
    return arguments


def check_ready(arguments, root=ROOT):
    """Explain missing prerequisites before starting a server or opening a browser."""
    if sys.version_info < (3, 10):
        raise ValueError('Python 3.10 or newer is needed; the original environment uses 3.12')
    # Help should remain available on a fresh clone without model packages/data.
    if '--help' in arguments or '-h' in arguments:
        return
    for module in ('torch', 'torchvision', 'numpy', 'PIL'):
        try:
            importlib.import_module(module)
        except Exception as error:
            raise ValueError(f'Cannot load {module}: {error}') from error
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--data-root', type=Path, default=Path.home()/'datasets/cecs551-plant-disease')
    parser.add_argument('--runs-root', type=Path, default=root/'outputs/runs')
    parser.add_argument('--splits-dir', type=Path, default=root/'outputs/splits')
    paths, _ = parser.parse_known_args(arguments)
    required = [paths.data_root/'plantvillage/raw/color', paths.data_root/'plantdoc/test',
        paths.data_root/'weights/mobilenet_v3_small-047dcff4.pth',
        *(paths.runs_root/f'source-seed-551/{mode}/best.pt' for mode in ('frozen','finetune')),
        *(paths.splits_dir/f'{split}.csv' for split in ('source_test','target_test'))]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError('Missing local files:\n  ' + '\n  '.join(missing)
                         + '\nSet their locations in demo-settings.local.json; see the demo guide.')
    try:
        subprocess.run(['git', '-C', str(root), 'cat-file', '-e',
                        '9619975a2f64954bc25d056708962eea1895d84b:plant_disease/model.py'],
                       capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError('Git and the original experiment history are required; use a full Git clone') from error


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    check_only = '--check' in arguments
    if check_only:
        arguments.remove('--check')
    try:
        # Explicit command-line arguments follow settings, so they take priority.
        arguments = settings_arguments() + arguments
        os.chdir(ROOT)
        check_ready(arguments)
        if check_only:
            return 0
        print(f'Starting the demo with {sys.executable}', flush=True)
        return subprocess.call([sys.executable, '-m', 'plant_disease.demo', '--open-browser', *arguments])
    except (OSError, ValueError) as error:
        print(f'Demo setup is not ready: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
