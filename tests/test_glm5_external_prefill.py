"""State-machine tests for GLM5Generator external-prefill (cache injection) support.

These tests exercise the pure-Python sequence state machine with a mocked
decode layer; kernel-level correctness is covered by
scripts/verify_external_prefill.py on an 8-GPU node.
"""

from unittest.mock import MagicMock

import pytest
import torch

from tilert.models.glm_5.generator import GLM5Generator
from tilert.models.glm_5.model_args import ModelArgsGLM5

N_LAYERS = ModelArgsGLM5().n_layers


def make_generator(with_mtp: bool) -> GLM5Generator:
    gen = GLM5Generator.__new__(GLM5Generator)
    gen.config = ModelArgsGLM5()
    gen.decode_layer = MagicMock()
    gen.with_mtp = with_mtp
    gen.max_new_tokens = 32
    gen.temperature = 1.0
    gen.top_p = 0.9
    gen.top_k = 1
    gen.use_topp = False
    gen.sampling_seed = 42
    gen.batch_size = 1
    gen.mtp_seq_len = 4
    gen.stop_token_ids = {2}
    gen.default_device = torch.device("cpu")
    gen._reset_sequence_state()

    # Injection primitives touch CUDA; record calls instead.
    gen.inject_cache = MagicMock()
    gen.set_cur_pos = MagicMock()
    gen.inject_last_hidden_state = MagicMock()
    return gen


def make_layer_caches(rows: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.zeros(rows, 128, dtype=torch.bfloat16),
            torch.zeros(rows, 512, dtype=torch.bfloat16),
            torch.zeros(rows, 64, dtype=torch.bfloat16),
        )
        for _ in range(N_LAYERS)
    ]


def test_prompt_to_tokens_accepts_batch_encoding_shape():
    class BatchEncodingLike:
        input_ids = [[11, 12, 13]]

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return BatchEncodingLike()

    gen = GLM5Generator.__new__(GLM5Generator)
    gen.tokenizer = Tokenizer()
    gen.enable_thinking = True

    assert gen._prompt_to_tokens("hello", None) == [11, 12, 13]


def test_full_cache_mtp_goes_straight_to_decode():
    gen = make_generator(with_mtp=True)
    prompt = list(range(100, 110))  # L = 10
    caches = make_layer_caches(10)
    hidden = torch.zeros(6144, dtype=torch.bfloat16)

    gen.start_sequence_from_cache(prompt, caches, last_hidden_state=hidden)

    assert gen._sequence_active
    assert gen._seq_prefill_done
    assert gen._seq_cur_pos == 9
    assert gen._seq_prefill_pos == 10
    gen.set_cur_pos.assert_called_once_with(9)
    gen.inject_last_hidden_state.assert_called_once()
    gen.decode_layer.set_prefill_valid_tokens.assert_called_once_with(0)
    (injected,), kwargs = gen.inject_cache.call_args
    assert kwargs == {"start_pos": 0}
    assert len(injected) == N_LAYERS
    assert injected[0][0].shape[0] == 10  # trimmed to cached_len rows


def test_full_cache_non_mtp_state():
    gen = make_generator(with_mtp=False)
    prompt = list(range(100, 110))  # L = 10
    caches = make_layer_caches(10)

    gen.start_sequence_from_cache(prompt, caches)

    assert gen._seq_prefill_done
    assert gen._seq_prev_pos == 9
    assert gen._seq_cur_pos == 10
    assert gen._seq_prefill_pos == 10
    gen.set_cur_pos.assert_called_once_with(9)
    gen.inject_last_hidden_state.assert_not_called()
    gen.decode_layer.set_prefill_valid_tokens.assert_not_called()


def test_partial_cache_mtp_resumes_internal_prefill():
    gen = make_generator(with_mtp=True)
    prompt = list(range(100))  # L = 100
    caches = make_layer_caches(64)  # page-aligned external cache
    hidden = torch.zeros(6144, dtype=torch.bfloat16)

    gen.start_sequence_from_cache(prompt, caches, last_hidden_state=hidden)

    assert not gen._seq_prefill_done
    assert gen._seq_prefill_pos == 64
    gen.set_cur_pos.assert_called_once_with(64)
    gen.inject_last_hidden_state.assert_called_once()
    # Decode transition (valid_tokens=0) must be left to the internal
    # prefill steps, not done eagerly.
    gen.decode_layer.set_prefill_valid_tokens.assert_not_called()


def test_partial_cache_non_mtp_resumes_internal_prefill():
    gen = make_generator(with_mtp=False)
    prompt = list(range(100))
    caches = make_layer_caches(64)

    gen.start_sequence_from_cache(prompt, caches)

    assert not gen._seq_prefill_done
    assert gen._seq_prefill_pos == 65  # next forward processes token 64
    gen.set_cur_pos.assert_called_once_with(64)


def test_cached_len_capped_at_prompt_len_minus_one():
    gen = make_generator(with_mtp=False)
    prompt = list(range(100, 110))  # L = 10
    caches = make_layer_caches(11)  # more rows than needed

    gen.start_sequence_from_cache(prompt, caches, cached_len=11)

    assert gen._seq_prefill_done
    gen.set_cur_pos.assert_called_once_with(9)
    (injected,), _ = gen.inject_cache.call_args
    assert injected[0][0].shape[0] == 10


def test_wrong_layer_count_rejected():
    gen = make_generator(with_mtp=False)
    with pytest.raises(ValueError, match="layer_caches"):
        gen.start_sequence_from_cache([1, 2, 3], make_layer_caches(2)[: N_LAYERS - 1])


def test_mtp_hidden_state_optional():
    gen = make_generator(with_mtp=True)
    prompt = list(range(100))
    # Partial cache without hidden state: internal prefill tail rebuilds the
    # hidden chain, so no injection should happen and no error raised.
    gen.start_sequence_from_cache(prompt, make_layer_caches(64))
    gen.inject_last_hidden_state.assert_not_called()
    assert not gen._seq_prefill_done
    assert gen._seq_prefill_pos == 64


def test_mixed_mtp_mode_rejected():
    gen = make_generator(with_mtp=True)
    with pytest.raises(ValueError, match="MTP mode"):
        gen.start_sequence_from_cache([1, 2, 3], make_layer_caches(2), with_mtp=False)


def test_cached_len_exceeding_rows_rejected():
    gen = make_generator(with_mtp=False)
    with pytest.raises(ValueError, match="positions"):
        gen.start_sequence_from_cache(list(range(100)), make_layer_caches(8), cached_len=16)


def test_short_prompt_falls_back_to_internal_prefill():
    gen = make_generator(with_mtp=False)
    gen.start_sequence = MagicMock()
    gen.start_sequence_from_cache([1], make_layer_caches(1))
    gen.start_sequence.assert_called_once()
    gen.inject_cache.assert_not_called()


def test_next_tokens_routes_to_decode_after_full_injection():
    gen = make_generator(with_mtp=True)
    prompt = list(range(100, 110))
    hidden = torch.zeros(6144, dtype=torch.bfloat16)
    gen.start_sequence_from_cache(prompt, make_layer_caches(10), last_hidden_state=hidden)

    gen._next_tokens_with_mtp = MagicMock(return_value=[7])
    assert gen.next_tokens() == [7]
    gen._next_tokens_with_mtp.assert_called_once()


def test_extract_cache_layout():
    gen = make_generator(with_mtp=False)
    caches = []
    for layer_id in range(N_LAYERS):
        for dim in (128, 512, 64):
            t = torch.full((1, 16, dim), float(layer_id), dtype=torch.bfloat16)
            caches.append(t)
    gen.decode_layer._get_device_result.return_value = (None, caches, None, None)
    gen.decode_layer.num_devices = 8

    out = gen.extract_cache(end_pos=5)
    assert len(out) == N_LAYERS
    ki, kv, pe = out[3]
    assert ki.shape == (5, 128) and kv.shape == (5, 512) and pe.shape == (5, 64)
    assert torch.all(kv == 3.0)
