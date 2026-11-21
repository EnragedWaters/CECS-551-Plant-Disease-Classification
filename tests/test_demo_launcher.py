"""Launcher checks use fake prerequisites; they do not download or train models."""
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import launch_demo


class DemoLauncherTests(unittest.TestCase):
    def test_personal_settings_accept_bom_and_resolve_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'demo-settings.local.json').write_text(
                json.dumps({'data_root':'data with spaces','port':8768}),encoding='utf-8-sig')
            self.assertEqual(launch_demo.settings_arguments(root),
                ['--data-root',str(root/'data with spaces'),'--port','8768'])

    def test_invalid_settings_fail_without_starting_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for value in [{'command':'anything'},{'port':True},{'port':0},{'device':'other'},
                          {'data_root':12},[],{'data_root':''}]:
                (root/'demo-settings.local.json').write_text(json.dumps(value))
                with self.assertRaises(ValueError):launch_demo.settings_arguments(root)

    def test_missing_files_report_paths_before_git_or_server(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(launch_demo.importlib,'import_module'), \
                patch.object(launch_demo.subprocess,'run') as git:
            with self.assertRaisesRegex(ValueError,'Missing local files'):
                launch_demo.check_ready(['--data-root',directory],Path(directory))
            git.assert_not_called()

    def test_ready_checks_require_original_history(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(launch_demo.importlib,'import_module'):
            root=Path(directory)
            for name in ['plantvillage/raw/color','plantdoc/test',
                'weights/mobilenet_v3_small-047dcff4.pth',
                'outputs/runs/source-seed-551/frozen/best.pt',
                'outputs/runs/source-seed-551/finetune/best.pt',
                'outputs/splits/source_test.csv','outputs/splits/target_test.csv']:
                path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.touch()
            with patch.object(launch_demo.subprocess,'run') as git:
                launch_demo.check_ready(['--data-root',str(root)],root)
                self.assertIn('9619975',git.call_args.args[0][-1])
            with patch.object(launch_demo.subprocess,'run',side_effect=FileNotFoundError):
                with self.assertRaisesRegex(ValueError,'original experiment history'):
                    launch_demo.check_ready(['--data-root',str(root)],root)

    def test_check_only_does_not_launch_and_cli_overrides_settings(self):
        with patch.object(launch_demo,'settings_arguments',return_value=['--port','8766']), \
                patch.object(launch_demo,'check_ready') as check, \
                patch.object(launch_demo.os,'chdir'), \
                patch.object(launch_demo.subprocess,'call',return_value=7) as run:
            self.assertEqual(launch_demo.main(['--check','--port','8780']),0)
            check.assert_called_with(['--port','8766','--port','8780'])
            run.assert_not_called()
            self.assertEqual(launch_demo.main(['--port','8780']),7)
            self.assertEqual(run.call_args.args[0],
                [sys.executable,'-m','plant_disease.demo','--open-browser','--port','8766','--port','8780'])

    @unittest.skipUnless(os.name=='posix','Checks the POSIX wrapper using a stand-in interpreter')
    def test_shell_preserves_spaces_and_does_not_retry_after_launch_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fake=root/'python with spaces';log=root/'arguments.jsonl'
            fake.write_text('#!'+sys.executable+'\nimport json,os,sys\n'
                'with open(os.environ["DEMO_TEST_LOG"],"a") as f:f.write(json.dumps({"args":sys.argv[1:],"cwd":os.getcwd()})+"\\n")\n'
                'sys.exit(0 if "--check" in sys.argv else 7)\n')
            fake.chmod(0o755)
            env=dict(os.environ,PLANT_DISEASE_PYTHON=str(fake),DEMO_TEST_LOG=str(log))
            project=root/'project with spaces'
            (project/'ONE-CLICK-LAUNCHERS').mkdir(parents=True)
            (project/'scripts').mkdir()
            shutil.copyfile(launch_demo.ROOT/'scripts/start_demo.sh',project/'scripts/start_demo.sh')
            for name in ['Linux or WSL - Run Demo.sh','macOS - Run Demo.command']:
                launcher=project/'ONE-CLICK-LAUNCHERS'/name
                shutil.copyfile(launch_demo.ROOT/'ONE-CLICK-LAUNCHERS'/name,launcher)
                log.write_text('')
                result=subprocess.run(['bash',str(launcher),
                    '--data-root',str(root/'data with spaces')],cwd=directory,env=env,
                    input='\n',text=True,capture_output=True)
                self.assertEqual(result.returncode,7,result.stderr)
                calls=[json.loads(line) for line in log.read_text().splitlines()]
                self.assertEqual(len(calls),2)
                self.assertEqual(calls[0]['args'][-1],str(root/'data with spaces'))
                self.assertIn('--check',calls[0]['args']);self.assertNotIn('--check',calls[1]['args'])
                self.assertTrue(all(Path(call['cwd'])==project for call in calls))



if __name__=='__main__':unittest.main()
