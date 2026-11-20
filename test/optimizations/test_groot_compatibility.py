"""Official-baseline compatibility without starting GR00T training."""

import importlib
import inspect
import os
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest


def compat():
    return importlib.import_module("turbo_physai.optimizations.models.groot.compat")


@pytest.mark.parametrize("requested,available,expected", [
    (True, False, False), (True, True, True), (False, False, False), (False, True, False),
])
def test_training_config_checks_tf32_only_when_requested(monkeypatch, requested, available, expected):
    import transformers.utils

    def capability():
        assert requested, "Disabled TF32 must not probe device capability"
        return available

    monkeypatch.setattr(transformers.utils, "is_torch_tf32_available", capability)
    monkeypatch.delenv("VLM_MODEL_PATH", raising=False)
    config = SimpleNamespace(training=SimpleNamespace(tf32=requested),
                             model=SimpleNamespace(model_name="nvidia/Cosmos-Reason2-2B"))

    def run(config):
        return config.training.tf32, config.model.model_name

    wrapped = compat().training_config_wrapper(run, {})
    assert inspect.signature(wrapped) == inspect.signature(run)
    assert wrapped(config=config) == (expected, "nvidia/Cosmos-Reason2-2B")


@pytest.mark.parametrize("path,expected", [
    (None, "nvidia/Cosmos-Reason2-2B"), ("", "nvidia/Cosmos-Reason2-2B"),
    ("/local/cosmos", "/local/cosmos"),
])
def test_training_config_applies_only_nonempty_local_model(monkeypatch, path, expected):
    if path is None:
        monkeypatch.delenv("VLM_MODEL_PATH", raising=False)
    else:
        monkeypatch.setenv("VLM_MODEL_PATH", path)
    config = SimpleNamespace(training=SimpleNamespace(tf32=False),
                             model=SimpleNamespace(model_name="nvidia/Cosmos-Reason2-2B"))
    assert compat().training_config_wrapper(lambda config: config.model.model_name, {})(config) == expected


def test_training_config_preserves_original_exception(monkeypatch):
    monkeypatch.delenv("VLM_MODEL_PATH", raising=False)
    config = SimpleNamespace(training=SimpleNamespace(tf32=False), model=SimpleNamespace())

    def run(config):
        raise RuntimeError("original run failed")

    with pytest.raises(RuntimeError, match="original run failed"):
        compat().training_config_wrapper(run, {})(config)


def test_video_preserves_official_backend_arguments_and_result(monkeypatch):
    monkeypatch.delenv("VIDEO_DECODER_BACKEND", raising=False)
    indices, options, result = [2, 0], {"device": "cpu"}, object()

    def decode(video_path, indices, decoder_kwargs=None):
        assert video_path == "video.mp4"
        assert indices == [2, 0]
        assert decoder_kwargs is options
        return result

    assert compat().opencv_video_wrapper(decode, {})("video.mp4", indices, options) is result


def test_opencv_decodes_requested_rgb_frames_including_duplicates(monkeypatch, tmp_path):
    import cv2

    monkeypatch.setenv("VIDEO_DECODER_BACKEND", "opencv")
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 5, (16, 16))
    assert writer.isOpened(), "Test requires OpenCV's MJPEG encoder"
    try:
        for color in [(10, 30, 80), (20, 70, 140), (90, 120, 200)]:
            writer.write(np.full((16, 16, 3), color, dtype=np.uint8))
    finally:
        writer.release()

    def unexpected(*args, **kwargs):
        raise AssertionError("OpenCV selection must not call torchcodec")

    frames = compat().opencv_video_wrapper(unexpected, {})(str(path), np.array([2, 0, 2]))
    assert frames.shape == (3, 16, 16, 3)
    assert frames.dtype == np.uint8
    np.testing.assert_array_equal(frames[0], frames[2])
    np.testing.assert_allclose(frames[0, 0, 0], [200, 120, 90], atol=3)
    np.testing.assert_allclose(frames[1, 0, 0], [80, 30, 10], atol=3)


def test_opencv_empty_and_negative_indices(monkeypatch):
    monkeypatch.setenv("VIDEO_DECODER_BACKEND", "opencv")
    wrapped = compat().opencv_video_wrapper(lambda *a, **k: pytest.fail("wrong backend"), {})
    assert wrapped("missing.mp4", []).shape == (0, 0, 0, 3)
    assert wrapped("missing.mp4", []).dtype == np.uint8
    with pytest.raises(ValueError, match="non-negative"):
        wrapped("missing.mp4", [-1])


@pytest.mark.parametrize("opened", [False, True])
def test_opencv_releases_capture_on_open_or_read_failure(monkeypatch, opened):
    import cv2

    monkeypatch.setenv("VIDEO_DECODER_BACKEND", "opencv")

    class Capture:
        released = False

        def isOpened(self):
            return opened

        def read(self):
            return False, None

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda path: capture)
    wrapped = compat().opencv_video_wrapper(lambda *a, **k: None, {})
    with pytest.raises(IndexError if opened else RuntimeError):
        wrapped("missing.mp4", [0])
    assert capture.released


@pytest.mark.model_deps
@pytest.mark.parametrize("early_import", [False, True])
def test_real_entrypoint_aliases_apply_and_restore_in_subprocess(early_import):
    # The replacement engine imports targets during preparation. Exercise a consumer
    # imported before preparation as well as a consumer imported after application.
    source = textwrap.dedent("""
        import importlib
        import os
        import runpy
        import sys
        from types import SimpleNamespace
        from turbo_physai.optimizations.models.groot import catalog
        from turbo_physai.engine.execution.replacements import default_handlers

        early = EARLY_IMPORT
        experiment = importlib.import_module('gr00t.experiment.experiment')
        video = importlib.import_module('gr00t.utils.video_utils')
        consumers = ('gr00t.experiment.launch_finetune',
                     'gr00t.data.dataset.lerobot_episode_loader')
        if early:
            for name in consumers:
                importlib.import_module(name)

        captured = []
        def run(config):
            captured.append(config)
            return config.model.model_name, config.training.tf32
        def decode(video_path, indices, decoder_kwargs=None):
            return 'official decoder'
        experiment.run = run
        video.get_frames_by_indices = decode
        # experiment itself can indirectly import the video consumer. Keep the
        # test doubles coherent for any consumer already loaded by upstream.
        if consumers[0] in sys.modules:
            sys.modules[consumers[0]].run = run
        if consumers[1] in sys.modules:
            sys.modules[consumers[1]].get_frames_by_indices = decode

        os.environ['VLM_MODEL_PATH'] = '/local/cosmos'
        os.environ.pop('VIDEO_DECODER_BACKEND', None)
        prepared = []
        for declaration in (catalog.TRAINING_CONFIG, catalog.OPENCV_VIDEO):
            spec, = declaration.specs
            handler = default_handlers()[spec.mechanism]
            item = handler.prepare(spec, {})
            snapshot = handler.snapshot(item)
            handler.apply(item)
            prepared.append((handler, snapshot))
        launcher = importlib.import_module(consumers[0])
        loader = importlib.import_module(consumers[1])
        cfg = SimpleNamespace(model=SimpleNamespace(model_name='official'),
                              training=SimpleNamespace(tf32=False))
        assert launcher.run(cfg) == ('/local/cosmos', False)
        assert loader.get_frames_by_indices('video', [0]) == 'official decoder'
        assert launcher.run is experiment.run
        assert loader.get_frames_by_indices is video.get_frames_by_indices
        # Execute the actual official CLI's configuration construction. The
        # replacement's original is the stub above, so training cannot start.
        captured.clear()
        sys.argv = [launcher.__file__, '--base-model-path', '/unused/checkpoint',
                    '--dataset-path', '/unused/data', '--embodiment-tag', 'LIBERO_PANDA',
                    '--global-batch-size', '32', '--max-steps', '10']
        runpy.run_path(launcher.__file__, run_name='__main__')
        assert len(captured) == 1
        assert captured[0].model.model_name == '/local/cosmos'
        assert captured[0].training.tf32 is False
        assert captured[0].training.global_batch_size == 32
        assert captured[0].training.max_steps == 10
        import torch.distributed
        assert not torch.distributed.is_initialized()
        for handler, snapshot in reversed(prepared):
            handler.restore(snapshot)
        assert launcher.run is experiment.run is run
        assert loader.get_frames_by_indices is video.get_frames_by_indices is decode
        cfg.model.model_name = 'official'
        assert launcher.run(cfg) == ('official', False)
    """).replace("EARLY_IMPORT", repr(early_import))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
               ROCR_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    result = subprocess.run([sys.executable, "-c", source], env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
