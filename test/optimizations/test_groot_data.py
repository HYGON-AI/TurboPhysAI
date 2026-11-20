"""BF16 pixel collation and sequence padding behavior."""

import copy
import pickle
import socket
from types import SimpleNamespace

import pytest
import torch

from turbo_physai.optimizations.models.groot.data import (
    Bf16PixelValuesCollator,
    trainer_init_bf16_wrapper,
)
from turbo_physai.optimizations.models.groot.fixed_padding import (
    fixed_padding_wrapper,
    training_padding_wrapper,
)


# Pixel values bf16

class Collator:
    marker = 3
    def __call__(self, features):
        return {"inputs": {"pixel_values": torch.tensor(features, dtype=torch.float32)}}


def test_collator_conversion_pickle_and_forwarded_attributes():
    wrapped = Bf16PixelValuesCollator(Collator())
    for obj in (wrapped, copy.copy(wrapped), pickle.loads(pickle.dumps(wrapped))):
        assert obj.marker == 3
        assert torch.equal(obj([1.25])["inputs"]["pixel_values"], torch.tensor([1.25], dtype=torch.bfloat16))


def test_trainer_wraps_only_deepspeed_bf16():
    def init(self, args=None, data_collator=None):
        self.data_collator = data_collator
    init = trainer_init_bf16_wrapper(init, {})
    trainer = SimpleNamespace()
    collator = Collator()
    init(trainer, SimpleNamespace(deepspeed="ds.json", bf16=True, fp16=False), collator)
    assert isinstance(trainer.data_collator, Bf16PixelValuesCollator)
    init(trainer, SimpleNamespace(deepspeed=None, bf16=True, fp16=False), collator)
    assert trainer.data_collator is collator


def test_custom_gr00t_trainer_keywords_are_preserved():
    def init(self, *args, **kwargs):
        self.kwargs = kwargs
    trainer = SimpleNamespace()
    args = SimpleNamespace(deepspeed="ds", bf16=True, fp16=False)
    trainer_init_bf16_wrapper(init, {})(trainer, args=args, data_collator=Collator(),
                                       multiprocessing_context="fork", action_offset=5)
    assert trainer.kwargs["multiprocessing_context"] == "fork"
    assert trainer.kwargs["action_offset"] == 5
    assert isinstance(trainer.kwargs["data_collator"], Bf16PixelValuesCollator)


def test_collator_runs_in_dataloader_worker():
    try:
        with socket.socket(socket.AF_UNIX) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    except PermissionError:
        pytest.skip("sandbox forbids the Unix socket needed for tensor IPC")
    loader = torch.utils.data.DataLoader([1.25, 2.5], batch_size=2, num_workers=1,
                                        multiprocessing_context="spawn", timeout=30,
                                        collate_fn=Bf16PixelValuesCollator(Collator()))
    assert next(iter(loader))["inputs"]["pixel_values"].dtype == torch.bfloat16


# Fixed padding

def trainer(pad=7):
    return SimpleNamespace(data_collator=SimpleNamespace(
        processor=SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=pad))))


def prepare_training_inputs(prepare, instance, batch):
    def training_step(self, model, inputs):
        return prepare(self, inputs)

    return training_padding_wrapper(training_step, {})(instance, None, batch)


def test_pad_156_left_pads_before_original_without_mutating_input():
    seen = []
    def original(self, inputs):
        seen.append(inputs)
        return inputs
    wrapped = fixed_padding_wrapper(original, {})
    ids = torch.tensor([[7, 4, 5], [6, 8, 9]])
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    batch = {"inputs": {"input_ids": ids, "attention_mask": mask, "state": torch.ones(2, 3)}}
    result = prepare_training_inputs(wrapped, trainer(), batch)
    values = result["inputs"]
    assert values["input_ids"].shape == (2, 156)
    assert torch.all(values["input_ids"][:, :153] == 7)
    assert torch.equal(values["input_ids"][:, -3:], ids)
    assert torch.all(values["attention_mask"][:, :153] == 0)
    assert torch.equal(values["attention_mask"][:, -3:], mask)
    assert values["state"] is batch["inputs"]["state"]
    assert batch["inputs"]["input_ids"] is ids
    assert seen == [result]


@pytest.mark.parametrize("width", [156, 157])
def test_exact_limit_and_oversized_input(width):
    fn = fixed_padding_wrapper(lambda self, x: x, {})
    instance = trainer()
    batch = {"inputs": {"input_ids": torch.ones((1, width), dtype=torch.long),
                        "attention_mask": torch.ones((1, width), dtype=torch.long)}}
    if width == 156:
        assert prepare_training_inputs(fn, instance, batch)["inputs"]["input_ids"].shape == (1, 156)
    else:
        with pytest.raises(ValueError, match="156"):
            prepare_training_inputs(fn, instance, batch)
    assert fn(instance, batch) is batch


@pytest.mark.parametrize("mask", [[[1, 0, 1]], [[0, 2, 1]]])
def test_reject_invalid_attention_mask(mask):
    batch = {"inputs": {"input_ids": torch.ones((1, 3), dtype=torch.long),
                        "attention_mask": torch.tensor(mask)}}
    fn = fixed_padding_wrapper(lambda self, x: x, {})
    with pytest.raises(ValueError, match="left-padded"):
        prepare_training_inputs(fn, trainer(), batch)


def test_reject_unknown_fields_and_missing_tokenizer():
    batch = {"inputs": {"input_ids": torch.ones((1, 2), dtype=torch.long),
                        "attention_mask": torch.ones((1, 2), dtype=torch.long)}}
    fn = fixed_padding_wrapper(lambda self, x: x, {})
    with pytest.raises(ValueError, match="pad token"):
        prepare_training_inputs(fn, trainer(None), batch)
    batch["inputs"]["unknown"] = 1
    with pytest.raises(ValueError, match="fields"):
        prepare_training_inputs(fn, trainer(), batch)


def test_pad_token_resolution_through_bf16_collator():
    instance = trainer()
    instance.data_collator = Bf16PixelValuesCollator(instance.data_collator)
    batch = {"inputs": {"input_ids": torch.tensor([[4]]), "attention_mask": torch.tensor([[1]])}}
    fn = fixed_padding_wrapper(lambda self, x: x, {})
    result = prepare_training_inputs(fn, instance, batch)
    assert result["inputs"]["input_ids"][0, 0].item() == 7
