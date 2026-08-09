import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict, Dict
from accelerate.utils import is_xpu_available

from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    LlamaForCausalLM_Conditional_Generation,
    LlamaConfig,
)
from llama_recipes.datasets.music_tokenizer import MusicTokenizer

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]
    logprobs: List[float]

_FORCE_FIELD_MAP = {
    "octave_dict_decode": "octave_tok",
    "pitch_dict_decode": "pitch_tok",
    "instrument_dict_decode": "instrument_tok",
}
# TIMING FORCING (added): timeshift and duration deliberately AREN'T in
# _FORCE_FIELD_MAP, because they can't use the same "look up a precomputed
# value" mechanism the fields above do. Octave/pitch/instrument are direct
# substitutions — the queued value IS the token to force. Timeshift is a
# DELTA relative to wherever the model's own running onset actually is at
# generation time, which isn't known until we're inside the loop; duration
# is more independent but still needs its own clamping against the
# tokenizer's reserved sos/eos boundary. Both are handled by dedicated
# blocks inside the generate() loop below (search "TIMING FORCING") rather
# than folded into the generic field-map mechanism.

class MusicLlama:
    @staticmethod
    def build(
        ckpt_dir: str, model_config_path: str, tokenizer_path: str,
        max_seq_len: int, max_batch_size: int, model_parallel_size: Optional[int] = None, seed: int = 1,
    ) -> "MusicLlama":
        if is_xpu_available(): torch.xpu.manual_seed(seed)
        else: torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        llama_config = LlamaConfig.from_pretrained(model_config_path)
        model = LlamaForCausalLM(llama_config)
        start_time = time.time()
        checkpoint = torch.load(ckpt_dir)['model_state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict)
        model.to("xpu" if is_xpu_available() else "cuda").eval()

        tokenizer = MusicTokenizer(timeshift_vocab_size=llama_config.onset_vocab_size, dur_vocab_size=llama_config.dur_vocab_size, octave_vocab_size=llama_config.octave_vocab_size, pitch_class_vocab_size=llama_config.pitch_class_vocab_size, instrument_vocab_size=llama_config.instrument_vocab_size, velocity_vocab_size=llama_config.velocity_vocab_size)

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            model = model.to(torch.bfloat16)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        return MusicLlama(model, tokenizer, llama_config)

    @staticmethod
    def build_commu_con_gen(
        ckpt_dir: str, model_config_path: str, tokenizer_path: str, max_seq_len: int, max_batch_size: int,
        model_parallel_size: Optional[int] = None, seed: int = 1, finetuned_PEFT_weight_path: Optional[str] = None,
        additional_token_dict: Optional[Dict] = None,
    ) -> "MusicLlama":
        if is_xpu_available(): torch.xpu.manual_seed(seed)
        else: torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        llama_config = LlamaConfig.from_pretrained(model_config_path)
        model = LlamaForCausalLM_Conditional_Generation(llama_config)
        checkpoint = torch.load(ckpt_dir)['model_state_dict']
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint.items()}
        model.load_state_dict(new_state_dict, strict=False)

        if finetuned_PEFT_weight_path is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, finetuned_PEFT_weight_path)

        model.to("xpu" if is_xpu_available() else "cuda").eval()
        tokenizer = MusicTokenizer(timeshift_vocab_size=llama_config.onset_vocab_size, dur_vocab_size=llama_config.dur_vocab_size, octave_vocab_size=llama_config.octave_vocab_size, pitch_class_vocab_size=llama_config.pitch_class_vocab_size, instrument_vocab_size=llama_config.instrument_vocab_size, velocity_vocab_size=llama_config.velocity_vocab_size)

        if additional_token_dict:
            for key, value in additional_token_dict.items():
                tokenizer.add_new_tokens(token_name=key, token_val=value)

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            model = model.to(torch.bfloat16)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        return MusicLlama(model, tokenizer, llama_config)

    @staticmethod
    def build_multi_task(
        ckpt_dir: str, lora_dir: str, active_adapter: str, model_config_path: str,
        max_seq_len: int, max_batch_size: int, seed: int = 1, additional_token_dict: Optional[Dict] = None,
    ) -> "MusicLlama":
        import os
        from peft import PeftModel

        if is_xpu_available(): torch.xpu.manual_seed(seed)
        else: torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

        llama_config = LlamaConfig.from_pretrained(model_config_path)
        model = LlamaForCausalLM_Conditional_Generation(llama_config)
        checkpoint = torch.load(ckpt_dir)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict, strict=False)

        model = PeftModel.from_pretrained(model, os.path.join(lora_dir, "commu_lora"), adapter_name="commu_lora")
        model.load_adapter(os.path.join(lora_dir, "emopia_lora"), adapter_name="emopia_lora")
        model.load_adapter(os.path.join(lora_dir, "slakh_lora"), adapter_name="slakh_lora")
        model.set_adapter(active_adapter)

        model.to("xpu" if is_xpu_available() else "cuda").eval()
        tokenizer = MusicTokenizer(
            timeshift_vocab_size=llama_config.onset_vocab_size, dur_vocab_size=llama_config.dur_vocab_size,
            octave_vocab_size=llama_config.octave_vocab_size, pitch_class_vocab_size=llama_config.pitch_class_vocab_size,
            instrument_vocab_size=llama_config.instrument_vocab_size, velocity_vocab_size=llama_config.velocity_vocab_size
        )

        if additional_token_dict:
            for key, value in additional_token_dict.items():
                tokenizer.add_new_tokens(token_name=key, token_val=value)

        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
            model = model.to(torch.bfloat16)

        return MusicLlama(model, tokenizer, llama_config)

    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        # 🚀 DYNAMIC DEVICE DETECTION (Fixes CPU/CUDA mismatch permanently)
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.inference_mode()
    def generate(
        self, prompt_tokens: List[List[List[int]]], bpm_condition: List[int], time_signature_condition: List[str],
        num_measures_condition: List[int], max_gen_len: int, temperature: float = 0.6, top_p: float = 0.9,
        logprobs: bool = False, echo: bool = False, metadata_condition: List = None, chord_condition: List = None,
        condition_token_lengths: List[int] = None, chord_dict: str = None, forced_token_streams: Optional[List[deque]] = None,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        
        bsz = len(prompt_tokens)
        if forced_token_streams is not None and len(forced_token_streams) != bsz:
            raise ValueError(f"forced_token_streams length ({len(forced_token_streams)}) must match batch size ({bsz})")

        if metadata_condition is not None:
            metadata_tokens = torch.tensor(metadata_condition, device=self.device, dtype=torch.long)
            
        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.config.max_len
        total_len = min(self.config.max_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_token_compound
        pad_tensor = torch.tensor(pad_id, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)
        tokens = pad_tensor.expand(bsz, total_len, -1).clone()

        for k, t in enumerate(prompt_tokens):
            t_tensor = torch.tensor(t, dtype=torch.long, device=self.device)
            tokens[k, :len(t)] = t_tensor

        if chord_condition is not None:
            bar_beat_chord_pad_id = [0, 0, 61]
            bar_beat_chord_pad_tensor = torch.tensor(bar_beat_chord_pad_id, dtype=torch.long, device=self.device).unsqueeze(0).unsqueeze(0)
            bar_beat_chord_condition = bar_beat_chord_pad_tensor.expand(bsz, total_len, -1).clone()
        else:
            bar_beat_chord_condition = None
            
        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        input_mask = torch.all(tokens != pad_tensor, dim=-1).unsqueeze(-1)

        past_key_values = None
        for cur_pos in range(min_prompt_len, total_len):
            output = self.model.forward(input_ids=tokens[:, prev_pos:cur_pos], past_key_values=past_key_values, use_cache=True, attention_mask=None)
            next_decoder_token = torch.tensor(self.tokenizer.sos_out, device=self.device).to(tokens).expand(tokens.shape[0]*(cur_pos - prev_pos), 1)
            next_decoder_token_out = next_decoder_token
            hidden_state = output.logits.view(output.logits.shape[0]*output.logits.shape[1], output.logits.shape[2]).unsqueeze(0).expand(self.model.decoder.num_hidden_layers, -1, -1).contiguous()

            if metadata_condition is not None:
                metadata_tokens_expanded = metadata_tokens.unsqueeze(1).expand(-1, cur_pos - prev_pos, -1).reshape(tokens.shape[0]*(cur_pos - prev_pos), -1)
            else:
                metadata_tokens_expanded = None
                
            if chord_condition is not None:
                bar_beat_chord_condition_expanded = bar_beat_chord_condition[:, prev_pos:cur_pos].reshape(-1, 3)

            # Ensure any condition tensors are on the same device as the decoder input
            # to avoid embedding/index-select device mismatch inside the model.
            decoder_device = None
            if 'next_decoder_token' in locals():
                decoder_device = next_decoder_token.device
            else:
                # Fallback to model/device stored on this wrapper
                decoder_device = self.device
            if metadata_condition is not None:
                metadata_tokens_expanded = metadata_tokens_expanded.to(decoder_device)
            if chord_condition is not None and bar_beat_chord_condition_expanded is not None:
                bar_beat_chord_condition_expanded = bar_beat_chord_condition_expanded.to(decoder_device)
            else:
                bar_beat_chord_condition_expanded = None

            # BUGFIX (row_offset): next_decoder_token has bsz*(cur_pos-prev_pos)
            # rows, not just bsz. On every iteration after the first,
            # cur_pos-prev_pos == 1, so this is a no-op (row_offset=0). But on
            # the VERY FIRST outer iteration, prev_pos=0 and cur_pos=
            # min_prompt_len — a prefill pass batching every primer position
            # together for KV-cache efficiency — so next_decoder_token has
            # bsz*min_prompt_len rows, and only the LAST bsz of them
            # correspond to the actual next new token; the rest are
            # re-decodings of already-known primer positions whose results
            # get discarded. Every forcing block below (this one and the
            # three added for timing) previously always indexed row `b`
            # (0 to bsz-1) directly — on this first iteration that's the
            # FIRST primer position being re-decoded, not the row that
            # actually determines tokens[:, cur_pos]. That meant the very
            # first genuinely-new note in every primer-continued section was
            # silently never forced at all (content OR timing), even though
            # forced_token_streams[b].popleft() still fired and "consumed"
            # it from the queue — confirmed directly: [ForceQueue] logs
            # showed full queue consumption on sections that still failed
            # completely, because the one row that mattered was untouched.
            # row_offset (below) assumes bsz=1, which is the only way this
            # pipeline ever calls generate() (AgenticComposer/HarmonyRouter
            # always pass a single-sequence prompt_tokens=[primer]). Output
            # is batch-major (batch_idx*seq_len + seq_idx) per the
            # `hidden_state = output.logits.view(...)` flattening above, so
            # for bsz>1 the last position PER BATCH ITEM is strided
            # (spaced seq_len apart), not contiguous — this simple
            # `row_offset + b` scheme would need reworking (e.g.
            # `b*num_positions + (num_positions-1)`) if this were ever used
            # with true multi-sequence batching. Flagging so this isn't
            # silently wrong if that assumption changes later.
            row_offset = (cur_pos - prev_pos - 1) * bsz

            for attribute in ["timeshift_dict_decode", "duration_dict_decode", "octave_dict_decode", "pitch_dict_decode", "instrument_dict_decode", "velocity_dict_decode"]:
                output_decoder = self.model.forward(decoded_hidden_state=hidden_state, decoded_language_tokens=next_decoder_token, attention_mask=None, metadata_condition=metadata_tokens_expanded, bar_beat_chord_condition=bar_beat_chord_condition_expanded)
                generation_logits = output_decoder.generation_logits
                hidden_state = output_decoder.generation_hidden_state

                sample_indices = list(getattr(self.tokenizer, attribute).keys())
                sample_indices_set = set(sample_indices)
                if temperature > 0:
                    probs = torch.softmax(generation_logits[:, -1, :] / temperature, dim=-1)
                    next_decoder_token = sample_top_p(probs, top_p)
                    for i in range(next_decoder_token.size(0)):
                        start_time = time.time()
                        while next_decoder_token[i, 0].item() not in sample_indices_set:
                            if time.time() - start_time > 15:
                                mask = torch.full_like(probs, float('-inf'))
                                mask[:, sample_indices] = probs[:, sample_indices]
                                probs = torch.softmax(mask, dim=-1)
                            next_decoder_token[i, 0] = sample_top_p(probs, top_p)[i, 0]
                else:
                    probs = torch.softmax(generation_logits[:, -1, :], dim=-1)
                    sample_indices_tensor = torch.tensor(sample_indices, device=self.device)
                    probs_at_sample_indices = probs[:, sample_indices_tensor]
                    next_token_index_in_subset = probs_at_sample_indices.argmax(dim=-1, keepdim=True)
                    next_decoder_token = sample_indices_tensor[next_token_index_in_subset.squeeze(-1)].unsqueeze(-1)

                # ============================================================
                # TIMING FORCING (added) — timeshift
                # ============================================================
                # Unlike octave/pitch/instrument below, timeshift is a DELTA
                # relative to wherever the model's own running onset
                # actually is right now, so it can't be a precomputed static
                # value in the queue — it has to be derived HERE, against
                # tokens[:, cur_pos-1, 0] (the previous position's actual
                # onset, already finalized from a prior loop iteration or
                # the prompt — always safe to read since cur_pos >= 1).
                # Forced whenever the queued item carries "target_tick",
                # with the delta CLAMPED into [0, sos_timeshift-1] rather
                # than skipped when out of range.
                #
                # BUGFIX (clamp instead of skip): the original version
                # skipped forcing entirely when the delta was invalid,
                # leaving the model's own free-sampled delta in place for
                # that one note. But the model's free sample has no
                # ceiling — it can jump by up to sos_timeshift-1 ticks in a
                # single step — while this section's own target range is
                # bounded (roughly primer_offset_ticks + max_section_ticks).
                # Confirmed against a real run: once the model's free onset
                # overshoots that ceiling even ONCE, every subsequent
                # desired_delta for the rest of the section is
                # mathematically stuck negative forever (our targets can
                # never exceed the section's bound; the model's onset
                # already has), so forcing kept skipping for the entire
                # remainder — a single bad free sample early on silently
                # broke timing for the whole rest of the section, even
                # though content forcing (octave/pitch/instrument) kept
                # succeeding the whole time. Clamping bounds the failure
                # mode instead: a negative delta forces 0 (stay put, let
                # the queue's own advancing targets catch back up instead
                # of drifting further away), and a too-large delta forces
                # the maximum representable step (bounded progress toward
                # the target) instead of an uncontrolled free jump.
                # Deliberately still excludes the top 2 reserved indices
                # (sos_timeshift, eos_timeshift) from the clamp range,
                # since accidentally forcing the EOS raw value would
                # trigger a false early stop via the eos_conditions check
                # further down this loop.
                if forced_token_streams is not None and attribute == "timeshift_dict_decode":
                    prev_onset_now = tokens[:, cur_pos - 1, 0].clone()
                    prev_onset_now = torch.where(prev_onset_now < 0, torch.zeros_like(prev_onset_now), prev_onset_now)
                    max_ts_val = self.tokenizer.sos_timeshift - 1
                    ts_override_vals = torch.zeros_like(next_decoder_token)
                    ts_override_mask = torch.zeros(next_decoder_token.shape[0], dtype=torch.bool, device=self.device)
                    for b in range(bsz):
                        row = row_offset + b  # BUGFIX: was just `b` — wrong row whenever this iteration is a multi-position prefill batch (see row_offset definition above)
                        dq = forced_token_streams[b]
                        if len(dq) > 0 and "target_tick" in dq[0]:
                            desired_delta = int(dq[0]["target_tick"]) - int(prev_onset_now[b].item())
                            clamped_delta = max(0, min(desired_delta, max_ts_val))
                            if clamped_delta in self.tokenizer.timeshift_dict:
                                ts_override_vals[row, 0] = self.tokenizer.timeshift_dict[clamped_delta]
                                ts_override_mask[row] = True
                    if ts_override_mask.any():
                        next_decoder_token = torch.where(ts_override_mask.unsqueeze(-1), ts_override_vals, next_decoder_token)

                # ============================================================
                # TIMING FORCING (added) — duration
                # ============================================================
                # Simpler than timeshift: this note's own intended length,
                # already converted to ticks by the caller, just needs
                # clamping into the valid (non-reserved) duration range.
                if forced_token_streams is not None and attribute == "duration_dict_decode":
                    max_dur_val = self.tokenizer.sos_dur - 1
                    dur_override_vals = torch.zeros_like(next_decoder_token)
                    dur_override_mask = torch.zeros(next_decoder_token.shape[0], dtype=torch.bool, device=self.device)
                    for b in range(bsz):
                        row = row_offset + b  # BUGFIX: was just `b` — see row_offset definition above
                        dq = forced_token_streams[b]
                        if len(dq) > 0 and "target_duration_ticks" in dq[0]:
                            dur_ticks = max(1, min(int(dq[0]["target_duration_ticks"]), max_dur_val))
                            if dur_ticks in self.tokenizer.duration_dict:
                                dur_override_vals[row, 0] = self.tokenizer.duration_dict[dur_ticks]
                                dur_override_mask[row] = True
                    if dur_override_mask.any():
                        next_decoder_token = torch.where(dur_override_mask.unsqueeze(-1), dur_override_vals, next_decoder_token)

                if forced_token_streams is not None and attribute in _FORCE_FIELD_MAP:
                    # BUGFIX: this pre-existing block had the exact same
                    # row-indexing bug as the timing-forcing blocks above —
                    # always writing to row `b` (0 to bsz-1) regardless of
                    # whether this outer iteration was a multi-position
                    # prefill batch. On the very first iteration of any
                    # primer-continued generation, that meant
                    # octave/pitch/instrument forcing ALSO silently failed
                    # for the first genuinely-new note, even before timing
                    # forcing was ever added — content forcing was never
                    # fully reliable on primer continuations.
                    field = _FORCE_FIELD_MAP[attribute]
                    override_vals = torch.zeros_like(next_decoder_token)
                    override_mask = torch.zeros(next_decoder_token.shape[0], dtype=torch.bool, device=self.device)
                    for b in range(bsz):
                        row = row_offset + b
                        dq = forced_token_streams[b]
                        if len(dq) > 0:
                            override_vals[row, 0] = dq[0][field]
                            override_mask[row] = True
                    next_decoder_token = torch.where(override_mask.unsqueeze(-1), override_vals, next_decoder_token)
                    if attribute == "instrument_dict_decode":
                        for b in range(bsz):
                            if len(forced_token_streams[b]) > 0:
                                forced_token_streams[b].popleft()

                next_decoder_token_out = torch.cat([next_decoder_token_out, next_decoder_token], dim=-1)

            next_decoder_token_out_reshaped = next_decoder_token_out[:, 1:].view(tokens.shape[0], -1, 6)
            next_decoder_token_lang = self.tokenizer.convert_from_language_tokens(next_decoder_token_out_reshaped)
            if next_decoder_token_lang.device != self.device:
                next_decoder_token_lang = next_decoder_token_lang.to(self.device)

            previous_onset = tokens[:, cur_pos-1, 0]
            if any(previous_onset < 0):
                previous_onset = torch.where(previous_onset < 0, torch.zeros_like(previous_onset), previous_onset)
            new_onset = previous_onset + next_decoder_token_lang.clone().detach()[:, -1, 0]
            next_decoder_token_onset = torch.cat([new_onset.unsqueeze(-1), next_decoder_token_lang.clone().detach()[:, -1, 1:]], dim=-1).to(tokens)
            next_token = torch.where(input_mask[:, cur_pos], tokens[:, cur_pos], next_decoder_token_onset)
            tokens[:, cur_pos] = next_token

            if chord_condition is not None:
                bar_beat_chord_new_onset = onset2bar_beat_chord(next_token[:, 0], chord_condition, time_signature_condition, bpm_condition, num_measures_condition, chord_dict)
                bar_beat_chord_new_onset_skip_pad = torch.where(input_mask[:, cur_pos], bar_beat_chord_condition[:, cur_pos], bar_beat_chord_new_onset.to(self.device))
                bar_beat_chord_condition[:, cur_pos] = bar_beat_chord_new_onset_skip_pad

            eos_conditions_all_attr = torch.stack([
                next_decoder_token_lang.clone().detach()[:, -1, 0] == self.tokenizer.eos_timeshift,
                next_decoder_token_lang.clone().detach()[:, -1, 1] == self.tokenizer.eos_dur,
                next_decoder_token_lang.clone().detach()[:, -1, 2] == self.tokenizer.eos_octave,
                next_decoder_token_lang.clone().detach()[:, -1, 3] == self.tokenizer.eos_pitch_class,
                next_decoder_token_lang.clone().detach()[:, -1, 4] == self.tokenizer.eos_instrument,
                next_decoder_token_lang.clone().detach()[:, -1, 5] == self.tokenizer.eos_velocity
            ], dim=-1)
            eos_conditions = torch.any(eos_conditions_all_attr, dim=-1).to(input_mask)
            eos_reached |= (~input_mask[:, cur_pos].squeeze(-1)) & eos_conditions
            
            prev_pos = cur_pos
            past_key_values = output.past_key_values
            if all(eos_reached): break

        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
            probs = None
            # BUGFIX: `if j == 0: continue` skipped checking timeshift's own
            # EOS value entirely. If generation actually stopped because
            # the model sampled eos_timeshift specifically — a completely
            # normal, expected way for generation to end — this loop never
            # found that boundary, so trailing PAD sentinels (self.tokenizer
            # .pad_token = -3, filled into `tokens` at init and never
            # overwritten past wherever generation actually stopped) were
            # never trimmed and passed straight through to compound_to_midi,
            # which then crashes trying to look up a "-3" instrument channel
            # that was never registered. Every attribute's own EOS is now
            # checked, not just 5 of the 6.
            for j, stop_token in enumerate([self.tokenizer.eos_timeshift, self.tokenizer.eos_dur, self.tokenizer.eos_octave, self.tokenizer.eos_pitch_class, self.tokenizer.eos_instrument, self.tokenizer.eos_velocity]):
                try:
                    eos_idx = [row[j] for row in toks].index(stop_token)
                    toks = toks[:eos_idx]
                    probs = probs[:eos_idx] if logprobs else None
                except ValueError: pass
            # Safety net, independent of which EOS check above would have
            # caught it: a row containing the raw PAD sentinel in ANY
            # column should never reach compound_to_midi. Catches any
            # future case where generation stops via a path none of the
            # six explicit EOS checks anticipated.
            pad_value = self.tokenizer.pad_token
            for idx, row in enumerate(toks):
                if pad_value in row:
                    toks = toks[:idx]
                    probs = probs[:idx] if logprobs else None
                    break
            out_tokens.append(toks)
            out_logprobs.append(probs)
            
        out_tokens_no_cond_tokens = []
        for i, condition_token_length in enumerate(condition_token_lengths):
            out_tokens_no_cond_tokens.append(out_tokens[i][condition_token_length:])

        return (out_tokens_no_cond_tokens, out_logprobs if logprobs else None)

    def music_completion(self, prompt_tokens, bpm_condition, time_signature_condition, num_measures_condition,
                         metadata_condition=None, chord_condition=None, temperature=0.6, top_p=0.9, max_gen_len=None,
                         logprobs=False, condition_token_lengths=None, chord_token_indices=None, chord_dict=None,
                         if_return_chords=True, forced_token_streams=None):
        if max_gen_len is None: max_gen_len = self.config.max_len - 1

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens, bpm_condition=bpm_condition, time_signature_condition=time_signature_condition,
            num_measures_condition=num_measures_condition, metadata_condition=metadata_condition, chord_condition=chord_condition,
            max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, logprobs=logprobs, echo=True,
            condition_token_lengths=condition_token_lengths, chord_dict=chord_dict, forced_token_streams=forced_token_streams,
        )
        
        chord_tokens = [prompt_tokens[i][chord_token_indices[i][0]+1:chord_token_indices[i][1]] for i in range(len(prompt_tokens))] if chord_token_indices else [generation_tokens[0] for _ in range(len(prompt_tokens))]
        generation_tokens_post_proc = [self.postprocess_split(t) if len(set(row[4] for row in t)) > 15 else [t] for t in generation_tokens]

        if if_return_chords:
            return [{"generation": {"role": "assistant", "content": self.tokenizer.compound_to_midi_multi(t), "chord": self.tokenizer.compound_to_midi_multi([chord]), "tokens": t, "chord_tokens": [chord]}} for chord, t in zip(chord_tokens, generation_tokens_post_proc)]
        else:
            return [{"generation": {"role": "assistant", "content": self.tokenizer.compound_to_midi_multi(t), "chord": None, "tokens": t, "chord_tokens": None}} for t in generation_tokens_post_proc]

    @staticmethod
    def postprocess_split(tokens):
        split2instrument, instrument2split, split2token = {}, {}, {}
        for token in tokens:
            instrument = int(token[4])
            if instrument in instrument2split:
                split2token[instrument2split[instrument]].append(token)
            else:
                if not split2instrument:
                    split2token[0], split2instrument[0], instrument2split[instrument] = [token], [instrument], 0
                else:
                    last_split = list(split2instrument.keys())[-1]
                    if len(split2instrument[last_split]) < 15:
                        split2token[last_split].append(token)
                        split2instrument[last_split].append(instrument)
                        instrument2split[instrument] = last_split
                    else:
                        new_split = last_split + 1
                        split2token[new_split], split2instrument[new_split], instrument2split[instrument] = [token], [instrument], new_split
        return list(split2token.values())

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    probs_sort[probs_sum - probs_sort > p] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    return torch.gather(probs_idx, -1, torch.multinomial(probs_sort, num_samples=1))

def onset2bar_beat_chord(onsets, chord_condition, time_signature_condition, bpm_condition, num_measures_condition, chord_dict):
    output = []
    for i in range(onsets.size(0)):
        onset_abs = onsets[i].item()
        numerator, denominator = int(time_signature_condition[i].split("/")[0]), int(time_signature_condition[i].split("/")[1])
        bpm = bpm_condition[i]
        chords = chord_condition[i] if num_measures_condition[i] % 4 == 0 else ["s"]*8 + chord_condition[i]
        
        total_beats = ((onset_abs / 100.0 * (120 / bpm)) * bpm) / 60.0
        bar_len_in_beats = numerator * (4 / denominator)
        bar_number = min(int(total_beats // bar_len_in_beats), num_measures_condition[i]-1)
        quantized_32nd = round((total_beats % numerator) * 8) % (bar_len_in_beats * 8)
        
        chord_symbol = chord_dict[chords[int(total_beats * 2.0) % len(chords)]]
        output.append([bar_number, quantized_32nd, chord_symbol])

    # 🚀 FIX: Ensure returned tensor is on the same device as the input onsets
    return torch.tensor(output, device=onsets.device)