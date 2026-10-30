"""Launcher argument and failure propagation checks without starting training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_pointpillars_turbo.sh'


def make_launch_env(tmp_path):
    repo = tmp_path / 'model repository'
    (repo / 'tools').mkdir(parents=True)
    (repo / 'tools' / 'train.py').touch()
    config = tmp_path / 'config.py'
    config.touch()
    optimization = tmp_path / 'optimization.yaml'
    optimization.touch()
    runtime = tmp_path / 'runtime.yaml'
    runtime.touch()
    data = tmp_path / 'data root'
    data.mkdir()
    stub = tmp_path / 'turbo-physai'
    stub.write_text('#!/usr/bin/env python3\nimport json,os,sys\n'
                    'open(os.environ["CAPTURE"],"w").write(json.dumps('
                    '{"args":sys.argv[1:],"hip":os.environ.get("HIP_VISIBLE_DEVICES"),'
                    '"cuda":os.environ.get("CUDA_VISIBLE_DEVICES"),'
                    '"cwd":os.getcwd()}))\n'
                    'sys.exit(int(os.environ.get("STUB_EXIT", "0")))\n')
    stub.chmod(0o755)
    env = dict(os.environ, CONDA_DEFAULT_ENV='mmcv2',
               POINTPILLARS_REPO_ROOT=str(repo), POINTPILLARS_CONFIG=str(config),
               POINTPILLARS_OPTIMIZATION_CONFIG=str(optimization),
               POINTPILLARS_RUNTIME_CONFIG=str(runtime), DATA_ROOT=str(data),
               WORK_DIR=str(tmp_path / 'work dir'), LOG_DIR=str(tmp_path / 'logs'),
               PATH=str(tmp_path) + os.pathsep + os.environ['PATH'],
               CAPTURE=str(tmp_path / 'capture.json'),
               HIP_VISIBLE_DEVICES='2,6')
    env.pop('CUDA_VISIBLE_DEVICES', None)
    return env


def check_launch_forwards_runtime_and_train_overrides(launch_env):
    result = subprocess.run(['bash', str(SCRIPT), 'randomness.seed=42'], env=launch_env,
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    capture = json.loads(Path(launch_env['CAPTURE']).read_text())
    args = capture['args']
    assert args[:1] == ['run']
    base = '/workspace/PointPillars_TurboPhysAI/TurboPhysAI/turbo_physai/optimizations/models/PointPillars/configs/'
    assert args[args.index('--runtime-config') + 1] == base + 'runtime.yaml'
    assert args[args.index('--optimization-config') + 1] == base + 'optimization.yaml'
    assert args[args.index('--nproc_per_node') + 1] == '2'
    pythonpath = next(arg for arg in args if arg.startswith('PYTHONPATH='))
    assert Path(pythonpath.split('=', 1)[1].split(':', 1)[0]).resolve() == Path(capture['cwd']).resolve()
    assert 'train_dataloader.batch_size=4' in args
    assert 'train_dataloader.num_workers=10' in args
    assert 'randomness.seed=42' in args
    assert 'HIP_VISIBLE_DEVICES=2,6' in args
    assert capture['cuda'] is None
    assert Path(capture['cwd']).resolve() == Path(launch_env['POINTPILLARS_REPO_ROOT']).resolve()


def check_card_ids_do_not_require_environment(launch_env):
    launch_env.pop('HIP_VISIBLE_DEVICES')
    result = subprocess.run(['bash', str(SCRIPT)], env=launch_env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    capture = json.loads(Path(launch_env['CAPTURE']).read_text())
    assert 'HIP_VISIBLE_DEVICES=2,6' in capture['args']


def check_training_exit_code_is_preserved(launch_env):
    launch_env['STUB_EXIT'] = '7'
    result = subprocess.run(['bash', str(SCRIPT)], env=launch_env, text=True, capture_output=True)
    assert result.returncode == 7


class LaunchTests(unittest.TestCase):
    def setUp(self):
        self.env = make_launch_env(Path(tempfile.mkdtemp(prefix='pp-launch-test-')))

    def test_arguments(self):
        check_launch_forwards_runtime_and_train_overrides(self.env)

    def test_card_ids_are_explicit(self):
        check_card_ids_do_not_require_environment(self.env)

    def test_exit_code(self):
        check_training_exit_code_is_preserved(self.env)


if __name__ == '__main__':
    unittest.main()
