"""DSA show hands for GLM5."""

import os
import time
from numbers import Integral

import torch
from transformers import AutoTokenizer

from tilert import logger
from tilert.models.glm_5._dsa_v32.generator import stats_time
from tilert.models.glm_5._dsa_v32.model_args import ModelArgs
from tilert.models.glm_5._dsa_v32.modules.end2end import ShowHandsDSALayer
from tilert.models.glm_5._dsa_v32.temp_var_indices import Idx
from tilert.tilert_init import tilert_init

__all__ = [
    "GLM5Generator",
]


class GLM5Generator:
    """Show hands generator for GLM5."""

    def __init__(
        self,
        model_args: ModelArgs,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        model_weights_dir: str = "",
        with_mtp: bool = False,
        top_p: float = 0.9,
        top_k: int = 256,
        use_topp: bool = False,
        enable_thinking: bool = False,
        sampling_seed: int = 42,
        mtp_cache_mode: int | None = None,
    ):
        """Initialize the ShowHandsGeneratorGlm5.

        Args:
            max_new_tokens: Maximum number of new tokens to generate. Defaults to 100.
            temperature: Temperature for sampling. Defaults to 1.0.
            model_weights_dir: Path of the model weights directory.
            with_mtp: Whether to use MTP (Multi-Token Prediction) for speculative decoding.
            top_p: Top-p (nucleus) sampling threshold. Defaults to 0.9.
            top_k: Top-k sampling threshold. Defaults to 1.
            use_topp: Whether to use top-p sampling. Defaults to False (top-1 argmax).
            enable_thinking: Whether to enable thinking mode in chat template.
        """
        torch.set_num_threads(64)
        self.model_weights_dir = model_weights_dir

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.with_mtp = with_mtp
        self.top_p = top_p
        self.top_k = top_k
        self._default_top_k = top_k
        self.use_topp = use_topp
        self.enable_thinking = enable_thinking
        self.sampling_seed = sampling_seed
        self.mtp_cache_mode = mtp_cache_mode

        self.config = model_args
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_weights_dir, trust_remote_code=True
        )  # nosec B615
        jinja_file_path = os.path.join(self.model_weights_dir, "chat_template.jinja")
        with open(jinja_file_path, encoding="utf-8") as f:
            chat_template = f.read()
        self.tokenizer.chat_template = chat_template
        self.eos_id = self.tokenizer.eos_token_id
        self.batch_size = 1
        self.mtp_seq_len = 4

        self.stop_tokens = [
            "<|user|>",
            "<|endoftext|>",
            "<|observation|>",
            "<|assistant|>",
        ]
        self.stop_token_ids: set[int] = set()
        for token in self.stop_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 1:
                self.stop_token_ids.add(token_ids[0])
            else:
                if (
                    hasattr(self.tokenizer, "added_tokens_encoder")
                    and token in self.tokenizer.added_tokens_encoder
                ):
                    self.stop_token_ids.add(self.tokenizer.added_tokens_encoder[token])
        if self.eos_id is not None:
            self.stop_token_ids.add(self.eos_id)
        logger.info(f"Stop token IDs: {self.stop_token_ids}")

        self.default_device = torch.device("cuda:0")

        self.decode_layer = ShowHandsDSALayer(
            model_args=self.config,
            model_path=self.model_weights_dir,
            with_mtp=with_mtp,
            top_p=top_p,
            top_k=top_k,
            use_topp=use_topp,
        )
        self._sequence_active = False
        self._seq_active_mtp = False
        self._seq_tokens: torch.Tensor | None = None
        self._seq_prompt_mask: torch.Tensor | None = None
        self._seq_prompt_len = 0
        self._seq_total_len = 0
        self._seq_prev_pos = 0
        self._seq_cur_pos = 0
        self._seq_prefill_done = False
        self._seq_prefill_pos = 0
        self._seq_finished = False
        self._seq_time_list: list[float] = []
        self._seq_accepted_counts: list[int] = []
        self._seq_output_tokens: list[int] = []
        self._seq_next_draft_tokens_cpu: torch.Tensor | None = None
        self._seq_decode_forward_seconds = 0.0
        self._seq_decode_post_seconds = 0.0

    def init(self) -> None:
        """Initialize the ShowHandsGeneratorGlm5."""
        tilert_init()

    def cleanup(self) -> None:
        """Cleanup the ShowHandsGeneratorGlm5."""
        self.decode_layer.cleanup()

    def init_random_weights(self) -> None:
        """Random initialize the weights."""
        self.decode_layer.init_random_weights()

    def from_pretrained(self) -> None:
        """Load the model weights from the given path."""
        self.decode_layer.from_pretrained(self.model_weights_dir)
        if self.mtp_cache_mode is not None:
            if hasattr(self.decode_layer, "set_mtp_cache_mode"):
                self.decode_layer.set_mtp_cache_mode(self.mtp_cache_mode)
            else:
                torch.ops.tilert.dsa_mtp_e2e_show_hands_set_cache_mode_glm5(self.mtp_cache_mode)
            logger.info("Set TileRT MTP cache mode to %s", self.mtp_cache_mode)

    def extract_ffn_cache(self) -> tuple[dict[int, list], dict[int, set[str]]]:
        """Extract MOE/MLP op objects and skip keys from current loaded weights.

        Returns:
            Tuple of (cached_ffn_ops_per_device, skip_keys_per_device).
        """
        from tilert.models.glm_5._dsa_v32.modules.end2end import (
            _extract_ffn_ops,
            _get_moe_weight_keys,
        )

        cached_ffn_ops: dict[int, list] = {}
        skip_keys: dict[int, set[str]] = {}
        for device_id in range(self.decode_layer.num_devices):
            dsa = self.decode_layer._dsa_objects[device_id]
            if dsa is None:
                raise RuntimeError(f"Device {device_id} Dsa not available for cache extraction")
            cached_ffn_ops[device_id] = _extract_ffn_ops(dsa)
            skip_keys[device_id] = _get_moe_weight_keys(dsa)
        return cached_ffn_ops, skip_keys

    def from_pretrained_with_cache(
        self,
        cached_ffn_ops_per_device: dict[int, list],
        skip_keys_per_device: dict[int, set[str]],
    ) -> None:
        """Load weights reusing cached MOE/MLP ops."""
        self.decode_layer.from_pretrained_with_cache(
            self.model_weights_dir, cached_ffn_ops_per_device, skip_keys_per_device
        )

    def update_sampling_params(
        self,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 256,
        use_topp: bool = True,
    ) -> None:
        """Update sampling parameters for the next generation."""
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.use_topp = use_topp
        self.decode_layer.update_sampling_config(
            temperature=temperature, top_p=top_p, top_k=top_k, use_topp=use_topp
        )

    def _normalize_token_ids(self, token_ids: object) -> list[int]:
        if hasattr(token_ids, "input_ids"):
            token_ids = token_ids.input_ids
        elif isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]

        if isinstance(token_ids, torch.Tensor):
            if token_ids.ndim == 2:
                if token_ids.shape[0] != 1:
                    raise ValueError("Expected exactly one prompt")
                token_ids = token_ids[0]
            if token_ids.ndim != 1:
                raise ValueError("Expected a 1D token id tensor")
            return [int(token_id) for token_id in token_ids.tolist()]

        if isinstance(token_ids, str):
            return self.tokenizer.encode(token_ids, add_special_tokens=False)

        if not isinstance(token_ids, (list, tuple)):
            raise TypeError(f"Unsupported token container: {type(token_ids)!r}")
        if not token_ids:
            return []

        first = token_ids[0]
        if isinstance(first, Integral):
            return [int(token_id) for token_id in token_ids]
        if isinstance(first, torch.Tensor):
            if len(token_ids) != 1:
                raise ValueError("Expected exactly one prompt")
            return self._normalize_token_ids(first)
        if isinstance(first, (list, tuple)):
            if len(token_ids) != 1:
                raise ValueError("Expected exactly one prompt")
            return self._normalize_token_ids(first)
        if isinstance(first, str):
            return self.tokenizer.encode("".join(token_ids), add_special_tokens=False)
        if hasattr(first, "ids"):
            if len(token_ids) != 1:
                raise ValueError("Expected exactly one prompt")
            return list(first.ids)

        raise TypeError(f"Unsupported token id element: {type(first)!r}")

    def _prompt_to_tokens(self, prompt: str, prompt_tokens: list[int] | None) -> list[int]:
        if prompt_tokens is not None:
            return self._normalize_token_ids(prompt_tokens)

        messages = [{"role": "user", "content": prompt}]
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        return self._normalize_token_ids(token_ids)

    def _update_sampling_from_request(self, sampling_params: object | None) -> None:
        if sampling_params is None:
            return

        temperature = getattr(sampling_params, "temperature", self.temperature)
        top_p = getattr(sampling_params, "top_p", self.top_p)
        top_k = getattr(sampling_params, "top_k", self.top_k)
        if top_k is None or int(top_k) < 0:
            top_k = self._default_top_k
        self.update_sampling_params(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            use_topp=bool(float(top_p) < 1.0),
        )

        max_new_tokens = getattr(sampling_params, "max_new_tokens", None)
        if max_new_tokens is not None:
            self.max_new_tokens = int(max_new_tokens)

        sampling_seed = getattr(sampling_params, "sampling_seed", None)
        if sampling_seed is not None:
            self.sampling_seed = int(sampling_seed)

    def _reset_sequence_state(self) -> None:
        self._sequence_active = False
        self._seq_active_mtp = False
        self._seq_tokens = None
        self._seq_prompt_mask = None
        self._seq_prompt_len = 0
        self._seq_total_len = 0
        self._seq_prev_pos = 0
        self._seq_cur_pos = 0
        self._seq_last_prompt_token = 0
        self._seq_prefill_done = False
        self._seq_prefill_pos = 0
        self._seq_finished = False
        self._seq_time_list = []
        self._seq_accepted_counts = []
        self._seq_output_tokens = []
        self._seq_next_draft_tokens_cpu = None
        self._seq_decode_forward_seconds = 0.0
        self._seq_decode_post_seconds = 0.0

    def _init_sequence_tensors(self, prompt_tokens: list[int]) -> None:
        if not prompt_tokens:
            raise ValueError("prompt_tokens must contain at least one token")

        max_seq_len = self.config.max_seq_len
        prompt_len = len(prompt_tokens)
        total_len = min(max_seq_len, self.max_new_tokens + prompt_len)

        tokens = torch.full(
            (self.batch_size, total_len),
            -1,
            dtype=torch.long,
            device=self.default_device,
        )
        tokens[0, :prompt_len] = torch.tensor(
            prompt_tokens, dtype=torch.long, device=self.default_device
        )

        self._seq_tokens = tokens
        self._seq_prompt_mask = tokens != -1
        self._seq_prompt_len = prompt_len
        self._seq_total_len = total_len
        self._seq_last_prompt_token = int(prompt_tokens[-1])
        self._seq_finished = total_len <= prompt_len
        self._seq_prefill_done = prompt_len <= 1
        self._seq_prefill_pos = 0 if self._seq_active_mtp else 1
        if self._seq_prefill_done:
            self._seq_prev_pos = prompt_len - 1
            self._seq_cur_pos = prompt_len - 1 if self._seq_active_mtp else prompt_len

    def _step_prefill_without_mtp(self) -> None:
        assert self._seq_tokens is not None
        assert self._seq_prompt_mask is not None

        if self._seq_prefill_done:
            return

        cur_pos_val = self._seq_prefill_pos
        prev_pos = cur_pos_val - 1
        start_time = time.time()
        multi_devices_results = self.decode_layer.forward(
            self._seq_tokens[0, prev_pos], with_mtp=self._seq_active_mtp
        )
        end_time = time.time()
        self._seq_time_list.append(end_time - start_time)

        intermediates, *_ = multi_devices_results[0]
        next_token = intermediates[Idx.TOKEN_OUT][0][0]
        next_token = torch.where(
            self._seq_prompt_mask[0, cur_pos_val],
            self._seq_tokens[0, cur_pos_val],
            next_token,
        )
        self._seq_tokens[0, cur_pos_val] = next_token
        self._seq_prefill_pos += 1

        if self._seq_prefill_pos >= self._seq_prompt_len:
            self._seq_prev_pos = self._seq_prompt_len - 1
            self._seq_cur_pos = self._seq_prompt_len
            self._seq_prefill_done = True

    def _step_prefill_with_mtp(self) -> None:
        assert self._seq_tokens is not None

        if self._seq_prefill_done:
            return

        cur_pos = self._seq_prefill_pos
        draft_end = min(cur_pos + self.mtp_seq_len, self._seq_prompt_len)
        draft_tokens = self._seq_tokens[0, cur_pos:draft_end].clone()
        actual_token_count = draft_tokens.shape[0]

        if actual_token_count < self.mtp_seq_len:
            pad_token = draft_tokens[-1].item()
            padding = torch.full(
                (self.mtp_seq_len - actual_token_count,),
                pad_token,
                dtype=torch.long,
                device=self.default_device,
            )
            draft_tokens = torch.cat([draft_tokens, padding])

        draft_tokens = draft_tokens.reshape(1, self.mtp_seq_len).to(torch.int32)

        mtp_extra_pos = cur_pos + self.mtp_seq_len
        if mtp_extra_pos < self._seq_prompt_len:
            mtp_extra_token = int(self._seq_tokens[0, mtp_extra_pos].item())
        else:
            mtp_extra_token = int(self._seq_tokens[0, draft_end - 1].item())
        self.decode_layer.set_prefill_mtp_extra_token(mtp_extra_token)
        self.decode_layer.set_prefill_valid_tokens(actual_token_count)
        self.decode_layer.forward(draft_tokens, with_mtp=True)

        self._seq_prefill_pos += actual_token_count
        if self._seq_prefill_pos >= self._seq_prompt_len - 1:
            self._seq_cur_pos = self._seq_prompt_len - 1
            self.set_cur_pos(self._seq_prompt_len - 1)
            self.decode_layer.set_prefill_valid_tokens(0)
            self._seq_prefill_done = True

    @torch.inference_mode()
    def start_sequence(
        self,
        prompt_tokens: list[int],
        sampling_params: object | None = None,
        with_mtp: bool | None = None,
    ) -> None:
        """Start an incremental single-sequence generation."""
        if self._sequence_active:
            self.finish_sequence()

        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        if active_mtp and not self.with_mtp:
            raise ValueError("Cannot use MTP mode: MTP weights were not loaded")

        self._reset_sequence_state()
        self._seq_active_mtp = active_mtp
        self._update_sampling_from_request(sampling_params)
        self.decode_layer.set_sampling_seed(self.sampling_seed, with_mtp=active_mtp)
        self._init_sequence_tensors(prompt_tokens)
        self._sequence_active = True

    @torch.inference_mode()
    def start_sequence_from_cache(
        self,
        prompt_tokens: list[int],
        layer_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        last_hidden_state: torch.Tensor | None = None,
        cached_len: int | None = None,
        sampling_params: object | None = None,
        with_mtp: bool | None = None,
    ) -> None:
        """Start a sequence whose prompt KV cache was computed by an external engine.

        This is the decode half of prefill-decode disaggregation: an external
        prefill system (e.g. SGLang) computes the per-layer (ki, kv, pe) caches
        for a prefix of the prompt, TileRT finishes any uncached tail with its
        internal chunked prefill, then decodes through the normal next_tokens()
        path.

        Args:
            prompt_tokens: Full prompt token ids (length L >= 2 for injection;
                shorter prompts fall back to internal prefill).
            layer_caches: One (ki, kv, pe) tuple per model layer (n_layers
                total), covering prompt positions [0, cached_len). Extra
                trailing rows are ignored. See inject_cache() for shapes and
                dtypes.
            last_hidden_state: Main-model hidden state of the last cached token
                (position cached_len - 1), shape [hidden_size] or
                [1, hidden_size] BF16. Only used in MTP mode, and optional even
                there: with cached_len <= L-2 the internal prefill tail rebuilds
                the hidden chain from the main model, so callers that cannot
                produce this tensor should simply cache one chunk less. Without
                it, a full hand-off (cached_len == L-1) only degrades early
                draft quality, never correctness.
            cached_len: Number of prompt positions covered by layer_caches.
                Defaults to min(available rows, L) and is capped at L. May be
                smaller than L (e.g. a page-aligned external cache); the
                remaining positions run through internal prefill.
            sampling_params: Optional per-request overrides (max_new_tokens,
                sampling_seed), same contract as start_sequence().
            with_mtp: Must match the loaded mode (self.with_mtp); mixed modes
                are rejected because cur_pos handling differs between kernels.

        Note:
            In MTP mode the TileRT draft layer's own KV cache for the prompt
            region cannot be provided by a main-model-only prefill, so early
            draft acceptance is degraded until decode has produced enough
            context. Output correctness is unaffected (the main model verifies
            every draft).
        """
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        if active_mtp != self.with_mtp:
            raise ValueError(
                "start_sequence_from_cache requires the sequence MTP mode to match "
                f"the loaded mode (with_mtp={self.with_mtp})"
            )

        if len(prompt_tokens) < 2:
            self.start_sequence(prompt_tokens, sampling_params=sampling_params, with_mtp=with_mtp)
            return

        n_layers = self.config.n_layers
        if len(layer_caches) != n_layers:
            raise ValueError(f"layer_caches must have {n_layers} entries, got {len(layer_caches)}")

        prompt_len = len(prompt_tokens)
        max_cached_len = prompt_len
        available_rows = layer_caches[0][0].size(0)
        if cached_len is None:
            cached_len = min(available_rows, max_cached_len)
        cached_len = min(cached_len, max_cached_len)
        if cached_len < 1:
            raise ValueError(f"cached_len must be >= 1, got {cached_len}")
        if available_rows < cached_len:
            raise ValueError(f"layer_caches cover {available_rows} positions, need {cached_len}")
        if active_mtp and last_hidden_state is None and cached_len >= max_cached_len:
            logger.warning(
                "Full cache hand-off in MTP mode without last_hidden_state; "
                "early draft acceptance will be degraded"
            )

        if self._sequence_active:
            self.finish_sequence()

        self._reset_sequence_state()
        self._seq_active_mtp = active_mtp
        self._update_sampling_from_request(sampling_params)
        self.decode_layer.set_sampling_seed(self.sampling_seed, with_mtp=active_mtp)
        self._init_sequence_tensors(prompt_tokens)

        trimmed = [
            (ki[:cached_len], kv[:cached_len], pe[:cached_len]) for ki, kv, pe in layer_caches
        ]
        self.inject_cache(trimmed, start_pos=0)
        self.set_cur_pos(cached_len if cached_len < prompt_len else prompt_len - 1)
        if active_mtp and last_hidden_state is not None:
            self.inject_last_hidden_state(last_hidden_state)

        # Leave the sequence state exactly where internal prefill would be
        # after covering [0, cached_len).
        if cached_len >= max_cached_len:
            self._seq_prefill_done = True
            if active_mtp:
                self.decode_layer.set_prefill_valid_tokens(0)
                self._seq_prefill_pos = prompt_len
                self._seq_cur_pos = prompt_len - 1
            else:
                self._seq_prefill_pos = prompt_len
                self._seq_prev_pos = prompt_len - 1
                self._seq_cur_pos = prompt_len
        else:
            self._seq_prefill_done = False
            self._seq_prefill_pos = cached_len if active_mtp else cached_len + 1
        self._sequence_active = True

    @torch.inference_mode()
    def next_tokens(self) -> list[int]:
        """Advance decoding once and return newly accepted token ids."""
        if not self._sequence_active:
            raise RuntimeError("start_sequence must be called before next_tokens")
        if self._seq_finished:
            return []
        if not self._seq_prefill_done:
            if self._seq_active_mtp:
                self._step_prefill_with_mtp()
            else:
                self._step_prefill_without_mtp()
            return []
        if self._seq_active_mtp:
            return self._next_tokens_with_mtp()
        return self._next_token_without_mtp()

    def _next_token_without_mtp(self) -> list[int]:
        assert self._seq_tokens is not None
        assert self._seq_cur_pos >= self._seq_prompt_len

        if self._seq_cur_pos >= self._seq_total_len:
            self._seq_finished = True
            return []

        start_time = time.time()
        multi_devices_results = self.decode_layer.forward(
            self._seq_tokens[0, self._seq_prev_pos], with_mtp=self._seq_active_mtp
        )
        end_time = time.time()
        self._seq_time_list.append(end_time - start_time)

        intermediates, *_ = multi_devices_results[0]
        next_token = intermediates[Idx.TOKEN_OUT][0][0]
        self._seq_tokens[0, self._seq_cur_pos] = next_token

        token_id = int(next_token.item())
        self._seq_output_tokens.append(token_id)
        self._seq_prev_pos = self._seq_cur_pos
        self._seq_cur_pos += 1
        self._seq_finished = (
            token_id in self.stop_token_ids or self._seq_cur_pos >= self._seq_total_len
        )
        return [token_id]

    def _next_tokens_with_mtp(self) -> list[int]:
        assert self._seq_tokens is not None

        if self._seq_cur_pos >= self._seq_total_len - 1:
            self._seq_finished = True
            return []

        if self._seq_cur_pos == self._seq_prompt_len - 1:
            draft_tokens = torch.full(
                (1, self.mtp_seq_len),
                self._seq_last_prompt_token,
                dtype=torch.int32,
            )
        else:
            draft_tokens = self._seq_next_draft_tokens_cpu
            if draft_tokens is None:
                draft_tokens = (
                    self.decode_layer.get_next_draft_tokens(0)
                    .reshape(1, self.mtp_seq_len)
                    .detach()
                    .cpu()
                )

        start_time = time.time()
        self.decode_layer.forward(draft_tokens, with_mtp=True)
        end_time = time.time()
        forward_seconds = end_time - start_time
        self._seq_time_list.append(forward_seconds)
        self._seq_decode_forward_seconds += forward_seconds

        post_start_time = time.time()
        num_accepted = int(self.decode_layer.get_num_accepted(0))
        if num_accepted <= 0:
            raise RuntimeError("TileRT MTP returned no accepted tokens")

        predicted_tokens = self.decode_layer.get_predicted_tokens(0).flatten()
        self._seq_next_draft_tokens_cpu = (
            self.decode_layer.get_next_draft_tokens(0).reshape(1, self.mtp_seq_len).detach().cpu()
        )
        self._seq_accepted_counts.append(num_accepted)

        remaining_slots = self._seq_total_len - self._seq_cur_pos - 1
        if remaining_slots <= 0:
            self._seq_finished = True
            self._seq_decode_post_seconds += time.time() - post_start_time
            return []

        num_output_tokens = min(num_accepted, remaining_slots)
        output_tokens = []
        for new_token in predicted_tokens[:num_output_tokens].detach().cpu().tolist():
            new_token = int(new_token)
            output_tokens.append(new_token)
            self._seq_output_tokens.append(new_token)

            if new_token in self.stop_token_ids:
                self._seq_finished = True
                break

        self._seq_cur_pos += len(output_tokens)
        if self._seq_cur_pos >= self._seq_total_len - 1:
            self._seq_finished = True

        self._seq_decode_post_seconds += time.time() - post_start_time
        return output_tokens

    def is_sequence_finished(self) -> bool:
        return self._seq_finished

    def sequence_progress(self) -> tuple[int, int, int, bool]:
        prefilled = min(self._seq_prefill_pos, self._seq_prompt_len)
        return (
            self._seq_prompt_len,
            prefilled,
            len(self._seq_output_tokens),
            self._seq_prefill_done,
        )

    def sequence_decode_stats(self, since_step: int = 0) -> dict[str, int | float | bool]:
        """Return decode-only progress counters for the active sequence."""
        output_tokens = len(self._seq_output_tokens)
        if self._seq_active_mtp:
            accepted_counts = self._seq_accepted_counts
            decode_steps = len(accepted_counts)
            accepted_total = sum(accepted_counts)
            if decode_steps:
                accepted_min = min(accepted_counts)
                accepted_max = max(accepted_counts)
                accepted_avg = accepted_total / decode_steps
            else:
                accepted_min = 0
                accepted_max = 0
                accepted_avg = 0.0

            if since_step < 0:
                since_step = 0
            if since_step > decode_steps:
                since_step = decode_steps
            recent_counts = accepted_counts[since_step:]
            recent_steps = len(recent_counts)
            recent_total = sum(recent_counts)
            recent_avg = recent_total / recent_steps if recent_steps else 0.0
        else:
            decode_steps = output_tokens
            accepted_total = output_tokens
            accepted_min = 1 if output_tokens else 0
            accepted_max = 1 if output_tokens else 0
            accepted_avg = 1.0 if output_tokens else 0.0
            if since_step < 0:
                since_step = 0
            if since_step > decode_steps:
                since_step = decode_steps
            recent_steps = decode_steps - since_step
            recent_total = recent_steps
            recent_avg = 1.0 if recent_steps else 0.0

        return {
            "with_mtp": self._seq_active_mtp,
            "output_tokens": output_tokens,
            "decode_steps": decode_steps,
            "accepted_total": accepted_total,
            "accepted_avg": accepted_avg,
            "accepted_min": accepted_min,
            "accepted_max": accepted_max,
            "recent_steps": recent_steps,
            "recent_accepted_total": recent_total,
            "recent_accepted_avg": recent_avg,
            "forward_seconds": self._seq_decode_forward_seconds,
            "post_seconds": self._seq_decode_post_seconds,
        }

    def _completion_tokens_for_decode(self) -> list[int]:
        stop_idx = len(self._seq_output_tokens)
        for i, tok in enumerate(self._seq_output_tokens):
            if tok in self.stop_token_ids:
                stop_idx = i
                break
        return self._seq_output_tokens[:stop_idx]

    def finish_sequence(self) -> None:
        """Reset runtime state for the active sequence."""
        if self._sequence_active:
            self.decode_layer.reset_sequence()
        self._reset_sequence_state()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        print_log: bool = True,
        with_mtp: bool | None = None,
        prompt_tokens: list[int] | None = None,
    ) -> tuple[str, list[float], list[int], int]:
        """Main function to load the model and perform single sequence generation.

        Args:
            prompt: The input prompt string.
            print_log: Whether to print generation logs.
            with_mtp: Override MTP mode for this call. None uses self.with_mtp.
                Requires MTP weights to have been loaded (self.with_mtp=True).
            prompt_tokens: Pre-tokenized prompt tokens. If provided, skip tokenization
                and use these tokens directly (useful for exact-length benchmarking).

        Returns:
            Tuple of (result_text, time_list, accepted_counts, prompt_len).
            accepted_counts is empty for non-MTP mode.
        """
        active_mtp = with_mtp if with_mtp is not None else self.with_mtp
        prompt_tokens = self._prompt_to_tokens(prompt, prompt_tokens)

        self.start_sequence(
            prompt_tokens,
            sampling_params=None,
            with_mtp=active_mtp,
        )
        try:
            while not self.is_sequence_finished():
                token_ids = self.next_tokens()
                if not token_ids:
                    continue
                if print_log:
                    for token_id in token_ids:
                        decoded_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                        print(decoded_text, end="", flush=True)

            time_list = list(self._seq_time_list)
            accepted_counts = list(self._seq_accepted_counts)
            prompt_len = self._seq_prompt_len
            completion_tokens = self._completion_tokens_for_decode()

            if print_log:
                print("\n")
                if active_mtp:
                    total_tokens = sum(accepted_counts)
                    logger.info(f"--Number of forward calls (decode): {len(accepted_counts)}")
                    logger.info(f"--Total tokens generated: {total_tokens}")
                    if len(accepted_counts) > 0:
                        avg_accepted = sum(accepted_counts) / len(accepted_counts)
                        min_accepted = min(accepted_counts)
                        max_accepted = max(accepted_counts)
                        logger.info(
                            f"--Accepted tokens per call: mean={avg_accepted:.2f}, "
                            f"min={min_accepted}, max={max_accepted}"
                        )

                    if time_list:
                        total_decode_time = sum(time_list)
                        effective_tps = (
                            total_tokens / total_decode_time if total_decode_time > 0 else 0
                        )
                        avg_time_ms = total_decode_time / len(time_list) * 1000
                        logger.info(
                            f"--Avg forward time: {avg_time_ms:.2f}ms, "
                            + f"({1000 / avg_time_ms:.2f} forwards/s)"
                        )
                        logger.info(f"--Effective TPS (with MTP): {effective_tps:.2f} tokens/s")
                else:
                    logger.info(f"--Number of tokens generated: {len(time_list)}")
                    stats_time(time_list, "==== Performance ====")
                print("\n")

            decoded_tokens = self.tokenizer.batch_decode(
                [completion_tokens], skip_special_tokens=True
            )
            result = f"{decoded_tokens[0]}\n" if decoded_tokens else ""
            return result, time_list, accepted_counts if active_mtp else [], prompt_len
        finally:
            self.finish_sequence()

    def inject_cache(
        self,
        layer_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        start_pos: int = 0,
        end_pos: int | None = None,
    ) -> None:
        """Inject external cache data into TileRT for P/D separation.

        This API allows injecting pre-computed KI/KV/PE cache data from an external
        prefill system (e.g., SGLang), enabling prefill-decode disaggregation.

        Args:
            layer_caches: List of (ki, kv, pe) tuples for each layer (0 to NUM_LAYERS-1).
                Each tensor should be BF16 with shape [seqlen, dim] where:
                - ki: [seqlen, 128] - compressed key (index_head_dim)
                - kv: [seqlen, 512] - compressed key-value (kv_lora_rank)
                - pe: [seqlen, 64] - position encoding cache (qk_rope_head_dim)
            start_pos: Start position in cache to write (0-indexed). Defaults to 0.
            end_pos: End position in cache (exclusive). If None, uses seqlen from tensors.

        Example:
            >>> # Load cache from external prefill system
            >>> layer_caches = []  # List of 78 (ki, kv, pe) tuples for GLM-5
            >>> for layer_id in range(78):
            ...     ki = load_ki_for_layer(layer_id)  # [seqlen, 128] bf16
            ...     kv = load_kv_for_layer(layer_id)  # [seqlen, 512] bf16
            ...     pe = load_pe_for_layer(layer_id)  # [seqlen, 64] bf16
            ...     layer_caches.append((ki, kv, pe))
            >>> generator.inject_cache(layer_caches, start_pos=0)
            >>> generator.set_cur_pos(seqlen)  # Set RoPE position
            >>> # Continue generation from cache
        """
        num_layers = len(layer_caches)
        if num_layers == 0:
            logger.warning("inject_cache called with empty layer_caches")
            return

        first_ki, _, _ = layer_caches[0]
        seqlen = first_ki.size(0)
        if end_pos is None:
            end_pos = start_pos + seqlen

        cache_len = end_pos - start_pos
        logger.info(f"Injecting cache: {num_layers} layers, positions [{start_pos}, {end_pos})")

        num_devices = self.decode_layer.num_devices
        device_caches = [
            self.decode_layer._get_device_result(device_id)[1] for device_id in range(num_devices)
        ]

        # TileRT's megakernel does not keep a full cache replica on every GPU:
        # the DSA indexer key (ki) is consumed on device 0, while the MLA
        # latent/rope caches (kv/pe) are consumed on devices 1..N-1. Avoid
        # populating unused tensors; for long prompts this removes most of the
        # host/device copy traffic from external-prefill injection.
        for layer_id, (ki, kv, pe) in enumerate(layer_caches):
            base_idx = layer_id * 3
            device_caches[0][base_idx + 0][0, start_pos:end_pos, :].copy_(
                ki[:cache_len],
                non_blocking=True,
            )

            if num_devices == 1:
                kv_device_ids = (0,)
            else:
                kv_device_ids = range(1, num_devices)
            for device_id in kv_device_ids:
                caches = device_caches[device_id]
                caches[base_idx + 1][0, start_pos:end_pos, :].copy_(
                    kv[:cache_len],
                    non_blocking=True,
                )
                caches[base_idx + 2][0, start_pos:end_pos, :].copy_(
                    pe[:cache_len],
                    non_blocking=True,
                )

        for device_id in range(num_devices):
            torch.cuda.synchronize(device_id)

        logger.info(f"Cache injection completed for {num_devices} devices")

    def set_cur_pos(self, cur_pos: int) -> None:
        """Set the current position for RoPE.

        This should be called after inject_cache() to ensure the runtime position
        matches the injected cache length, for correct RoPE position encoding
        during continued generation.

        Args:
            cur_pos: The current sequence position (typically the length of prefilled tokens).

        Example:
            >>> generator.inject_cache(layer_caches, start_pos=0)
            >>> generator.set_cur_pos(prefill_len)  # Set position to prefill length
            >>> # Now generate continues from the correct position
        """
        if self.with_mtp:
            num_devices = self.decode_layer.num_devices
            for device_id in range(num_devices):
                intermediates, _, _, _ = self.decode_layer._get_device_result(device_id)
                cur_pos_tensor = intermediates[Idx.CUR_POS]
                cur_pos_tensor.fill_(cur_pos)
        else:
            torch.ops.tilert.dsa_show_hands_set_cur_pos_glm5(cur_pos)
            logger.info(f"Set cur_pos to {cur_pos}")

    def inject_last_hidden_state(self, last_hidden_state: torch.Tensor) -> None:
        """Inject the last hidden state for MTP mode.

        For MTP (Multi-Token Prediction), the MTP preprocess layer needs the
        last hidden state from the main model's last token.

        Args:
            last_hidden_state: [hidden_size] or [1, hidden_size] BF16 tensor.
                The hidden state of the last token from prefill.

        Example:
            >>> # After inject_cache, inject the last hidden state for MTP
            >>> generator.inject_last_hidden_state(last_hidden_state)
            >>> generator.set_cur_pos(prefill_len)
            >>> # Then start generation
        """
        if not self.with_mtp:
            logger.warning("inject_last_hidden_state called but with_mtp is False, skipping")
            return

        if last_hidden_state.dim() == 1:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        num_devices = self.decode_layer.num_devices
        for device_id in range(num_devices):
            intermediates, _, _, _ = self.decode_layer._get_device_result(device_id)
            lhs_tensor = intermediates[Idx.LAST_HIDDEN_STATES]
            lhs_src = last_hidden_state.to(f"cuda:{device_id}")
            lhs_tensor[0, 0, :].copy_(lhs_src.squeeze(0))

        logger.info(f"Injected last_hidden_state to {num_devices} devices")

    def extract_cache(
        self,
        end_pos: int,
        start_pos: int = 0,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Extract per-layer (ki, kv, pe) cache rows for positions [start_pos, end_pos).

        The caches are NOT replicated across devices: the megakernel maintains
        ki only on device 0 (the DSA indexer / sparse-select device) and kv/pe
        only on the pure-MLA devices (1..num_devices-1); the unused tensors on
        each device stay zero. Reading everything from one device would return
        zero kv/pe. The returned tensors are CPU BF16 clones in the exact
        format inject_cache() accepts, which makes internal-prefill ->
        extract -> inject round-trip verification possible.

        Args:
            end_pos: End position (exclusive).
            start_pos: Start position. Defaults to 0.

        Returns:
            List of n_layers (ki, kv, pe) tuples on CPU.
        """
        if end_pos <= start_pos:
            raise ValueError(f"empty extraction range [{start_pos}, {end_pos})")

        _, ki_caches, _, _ = self.decode_layer._get_device_result(0)
        kvpe_device = 1 if self.decode_layer.num_devices > 1 else 0
        _, kvpe_caches, _, _ = self.decode_layer._get_device_result(kvpe_device)
        layer_caches = []
        for layer_id in range(self.config.n_layers):
            base_idx = layer_id * 3
            layer_caches.append(
                (
                    ki_caches[base_idx + 0][0, start_pos:end_pos, :].to("cpu", torch.bfloat16),
                    kvpe_caches[base_idx + 1][0, start_pos:end_pos, :].to("cpu", torch.bfloat16),
                    kvpe_caches[base_idx + 2][0, start_pos:end_pos, :].to("cpu", torch.bfloat16),
                )
            )
        return layer_caches

    def extract_last_hidden_state(self, device_id: int = 0) -> torch.Tensor:
        """Extract the runtime's current last-token hidden state (MTP mode).

        After an internal MTP prefill this is the hidden state of the last
        prompt token — exactly the tensor inject_last_hidden_state() expects.
        Useful for verifying an external prefill engine produces an equivalent
        hidden state.

        Returns:
            [hidden_size] BF16 tensor on CPU.
        """
        intermediates, _, _, _ = self.decode_layer._get_device_result(device_id)
        return intermediates[Idx.LAST_HIDDEN_STATES][0, 0, :].to("cpu", torch.bfloat16)
