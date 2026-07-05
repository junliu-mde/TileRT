"""State-machine tests for GLM5Generator external-prefill (cache injection) support.

These tests exercise the pure-Python sequence state machine with a mocked
decode layer; kernel-level correctness is covered by
scripts/verify_external_prefill.py on an 8-GPU node.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tilert.models.glm_5._dsa_v32.modules.mla_v2 import PureMlaV2, SparseSelectMlaV2
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
    gen._default_top_k = 1
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


def test_sequence_decode_stats_reports_mtp_acceptance():
    gen = make_generator(with_mtp=True)
    gen._seq_active_mtp = True
    gen._seq_output_tokens = [101, 102, 103, 104, 105, 106]
    gen._seq_accepted_counts = [1, 4, 1]

    stats = gen.sequence_decode_stats(since_step=1)

    assert stats["with_mtp"] is True
    assert stats["output_tokens"] == 6
    assert stats["decode_steps"] == 3
    assert stats["accepted_total"] == 6
    assert stats["accepted_avg"] == 2.0
    assert stats["accepted_min"] == 1
    assert stats["accepted_max"] == 4
    assert stats["recent_steps"] == 2
    assert stats["recent_accepted_total"] == 5
    assert stats["recent_accepted_avg"] == 2.5


def test_sequence_decode_stats_reports_non_mtp_as_single_acceptance():
    gen = make_generator(with_mtp=False)
    gen._seq_active_mtp = False
    gen._seq_output_tokens = [101, 102, 103]

    stats = gen.sequence_decode_stats(since_step=99)

    assert stats["with_mtp"] is False
    assert stats["output_tokens"] == 3
    assert stats["decode_steps"] == 3
    assert stats["accepted_total"] == 3
    assert stats["accepted_avg"] == 1.0
    assert stats["accepted_min"] == 1
    assert stats["accepted_max"] == 1
    assert stats["recent_steps"] == 0
    assert stats["recent_accepted_total"] == 0
    assert stats["recent_accepted_avg"] == 0.0


def test_update_sampling_from_request_updates_decode_layer_config():
    gen = make_generator(with_mtp=True)
    params = SimpleNamespace(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_new_tokens=7,
        sampling_seed=1234,
    )

    gen._update_sampling_from_request(params)

    assert gen.temperature == 0.0
    assert gen.top_p == 1.0
    assert gen.top_k == gen._default_top_k
    assert gen.use_topp is False
    assert gen.max_new_tokens == 7
    assert gen.sampling_seed == 1234
    gen.decode_layer.update_sampling_config.assert_called_once_with(
        temperature=0.0,
        top_p=1.0,
        top_k=gen._default_top_k,
        use_topp=False,
    )


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


def test_next_tokens_with_mtp_commits_predicted_tokens_until_stop():
    gen = make_generator(with_mtp=True)
    gen._seq_active_mtp = True
    gen._sequence_active = True
    gen._seq_prompt_len = 3
    gen._seq_total_len = 10
    gen._seq_cur_pos = 3
    gen._seq_prefill_done = True
    gen._seq_tokens = torch.full((1, 10), -1, dtype=torch.long)
    gen._seq_output_tokens = [100]
    gen.decode_layer.get_next_draft_tokens.return_value = torch.tensor(
        [[11, 12, 13, 14]], dtype=torch.int32
    )
    gen.decode_layer.get_num_accepted.return_value = 4
    gen.decode_layer.get_predicted_tokens.return_value = torch.tensor(
        [[201, 202, 2, 204]], dtype=torch.int32
    )

    assert gen.next_tokens() == [201, 202, 2]
    assert gen._seq_output_tokens == [100, 201, 202, 2]
    assert gen._seq_cur_pos == 6
    assert gen._seq_finished
    assert gen._seq_accepted_counts == [4]


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


def test_inject_cache_copies_host_tensors_directly(monkeypatch):
    gen = make_generator(with_mtp=False)
    gen.inject_cache = GLM5Generator.inject_cache.__get__(gen, GLM5Generator)
    gen.decode_layer.num_devices = 3
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device_id: None)

    device_caches = [_make_device_caches(rows=8) for _ in range(gen.decode_layer.num_devices)]
    gen.decode_layer._get_device_result.side_effect = [
        (None, device_caches[0], None, None),
        (None, device_caches[1], None, None),
        (None, device_caches[2], None, None),
    ]
    layer_caches = []
    for layer_id in range(N_LAYERS):
        layer_caches.append(
            (
                torch.full((4, 128), 10 + layer_id, dtype=torch.bfloat16),
                torch.full((4, 512), 20 + layer_id, dtype=torch.bfloat16),
                torch.full((4, 64), 30 + layer_id, dtype=torch.bfloat16),
            )
        )

    gen.inject_cache(layer_caches, start_pos=2, end_pos=6)

    assert torch.all(device_caches[0][0][0, 2:6, :] == 10)
    assert torch.all(device_caches[0][1] == 0)
    assert torch.all(device_caches[0][2] == 0)
    for caches in device_caches[1:]:
        assert torch.all(caches[0] == 0)
        assert torch.all(caches[1][0, 2:6, :] == 20)
        assert torch.all(caches[2][0, 2:6, :] == 30)


def test_sparse_select_mla_allocates_only_index_cache(monkeypatch):
    monkeypatch.setattr(torch, "zeros", _cpu_tensor_factory(torch.zeros))
    monkeypatch.setattr(torch, "empty", _cpu_tensor_factory(torch.empty))
    args = ModelArgsGLM5(max_seq_len=17, kv_cache_pad=3, max_batch_size=2)

    caches = SparseSelectMlaV2(args, device_id=0, num_devices=8).get_cache_vars()[-3:]

    assert caches[0].shape == (2, 20, args.index_head_dim)
    assert caches[1].numel() == 1
    assert caches[2].numel() == 1


def test_pure_mla_allocates_only_kvpe_cache(monkeypatch):
    monkeypatch.setattr(torch, "zeros", _cpu_tensor_factory(torch.zeros))
    monkeypatch.setattr(torch, "empty", _cpu_tensor_factory(torch.empty))
    args = ModelArgsGLM5(max_seq_len=17, kv_cache_pad=3, max_batch_size=2)

    caches = PureMlaV2(args, device_id=1, num_devices=7).get_cache_vars()[-3:]

    assert caches[0].numel() == 1
    assert caches[1].shape == (2, 20, args.kv_lora_rank)
    assert caches[2].shape == (2, 20, args.qk_rope_head_dim)


def _cpu_tensor_factory(factory):
    def make_cpu_tensor(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["device"] = "cpu"
        return factory(*args, **kwargs)

    return make_cpu_tensor


def _make_device_caches(rows: int) -> list[torch.Tensor]:
    caches = []
    for _ in range(N_LAYERS):
        caches.extend(
            (
                torch.zeros(1, rows, 128, dtype=torch.bfloat16),
                torch.zeros(1, rows, 512, dtype=torch.bfloat16),
                torch.zeros(1, rows, 64, dtype=torch.bfloat16),
            )
        )
    return caches
