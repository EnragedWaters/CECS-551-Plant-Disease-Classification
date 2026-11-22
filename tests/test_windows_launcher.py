"""Run real Windows wrappers against a stand-in launcher, without WSL or data."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.name == 'nt', 'Native Windows wrapper checks run on the Windows CI runner')
class WindowsLauncherTests(unittest.TestCase):
    def test_native_launch_arguments_cwd_and_failure_handling(self):
        with tempfile.TemporaryDirectory(prefix='launcher checks ') as directory:
            root = Path(directory).resolve()/'project with spaces'
            (root/'scripts').mkdir(parents=True)
            (root/'ONE-CLICK-LAUNCHERS').mkdir()
            for name in ['scripts/start_demo.ps1', 'ONE-CLICK-LAUNCHERS/Windows - Run Demo.cmd']:
                shutil.copyfile(ROOT/name, root/name)
            # The real PowerShell/CMD files run; only their Python entry is a fixture.
            (root/'scripts/launch_demo.py').write_text(
                'import json,os,sys\n'
                'with open(os.environ["DEMO_TEST_LOG"],"a") as f:\n'
                ' f.write(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd()})+"\\n")\n'
                'sys.exit(int(os.environ["CHECK_EXIT"] if "--check" in sys.argv else os.environ["RUN_EXIT"]))\n')
            log = Path(directory)/'calls.jsonl'
            env = dict(os.environ, PLANT_DISEASE_PYTHON=sys.executable, DEMO_TEST_LOG=str(log))
            ps = ['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(root/'scripts/start_demo.ps1')]
            cmd = ['cmd.exe', '/d', '/c', str(root/'ONE-CLICK-LAUNCHERS/Windows - Run Demo.cmd')]
            args = ['--no-open-browser', '--data-root', str(Path(directory)/'images with spaces')]
            for prefix in [ps, cmd]:
                for check_exit, run_exit in [(0, 0), (0, 7), (3, 0)]:
                    with self.subTest(wrapper=prefix[0], check=check_exit, run=run_exit):
                        log.write_text('')
                        # cmd /s /c strips its outer quotation pair. Preserve a
                        # second pair around the batch path when it contains spaces.
                        command = ('cmd.exe /d /s /c "' +
                                   subprocess.list2cmdline(prefix[3:] + args) + '"'
                                   if prefix[0] == 'cmd.exe' else prefix + args)
                        result = subprocess.run(command, cwd=directory,
                            env=dict(env, CHECK_EXIT=str(check_exit), RUN_EXIT=str(run_exit)),
                            input='\n', text=True, capture_output=True, timeout=30)
                        self.assertEqual(result.returncode == 0, check_exit == 0 and run_exit == 0,
                                         result.stdout + result.stderr)
                        calls = [json.loads(line) for line in log.read_text().splitlines()]
                        self.assertEqual(len(calls), 2 if check_exit == 0 else 1, result.stdout + result.stderr)
                        self.assertEqual(calls[0]['args'], ['--check', *args])
                        if check_exit == 0:
                            self.assertEqual(calls[1]['args'], args)
                        self.assertTrue(all(Path(c['cwd']).resolve() == root for c in calls))
                        self.assertNotIn('Checking the default WSL', result.stdout)
