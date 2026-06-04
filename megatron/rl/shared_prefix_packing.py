# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pure-tensor layout helpers for shared-prefix ("tree") packing of a GRPO group.

Milestone 1 (oracle stage) of the shared-prefix-packing optimization described in
``docs/sequence_packing_prefix_sharing.md``. In a GRPO group of size G the prompt P is
shared by all G completions C_i; today the packer duplicates P into G separate
``[P + C_i]`` blocks. These helpers instead describe a single ``[P, C_1, ..., C_G]``
layout plus:

  * prefix-continued ``position_ids`` (P -> 0..Lp-1, each C_i -> Lp..Lp+Lc_i-1),
  * a tree attention mask (FlexAttention ``mask_mod`` + a dense boolean fallback) where
    each C_i attends to all of P and causally within itself, and never to a sibling C_j,
  * the fanned-out ``(prev_position, target_position)`` index pairs that make the shared
    forward's logprob extraction equivalent to the contiguous ``[P + C_i]`` shift -- in
    particular each branch's FIRST completion token is scored from the shared prefix's
    last-position logit (the same logit for all branches), not from the preceding branch.

NOTHING HERE IS WIRED INTO THE LIVE TRAINING FORWARD. It is exercised by
``mrl_extras/test/test_shared_prefix_equivalence.py`` (the numerical oracle), which must
pass on GPU before the forward integration (Milestone 1b). Keeping these as pure functions
(torch + stdlib only, no megatron-core/TE imports) lets the oracle run standalone.
"""

import contextlib
import types
from dataclasses import dataclass, field
from typing import Callable, List

import torch


@dataclass
class SharedPrefixLayout:
    """Describes one ``[P, C_1, ..., C_G]`` shared-prefix bin (single group).

    All index tensors are 1-D and live on ``device``. ``comp_*`` tensors have one entry
    per completion token, concatenated branch-by-branch in order C_0, C_1, ... .
    """

    prefix_len: int
    completion_lens: List[int]
    total_len: int
    branch_starts: List[int]          # absolute packed start offset of each completion
    position_ids: torch.Tensor        # [total_len]  long; prefix-continued positions
    segment_ids: torch.Tensor         # [total_len]  long; 0 = prefix, i+1 = completion i
    comp_positions: torch.Tensor      # [n_comp_tok] packed index of each completion token
    prev_positions: torch.Tensor      # [n_comp_tok] packed index whose logit predicts it
    branch_of_token: torch.Tensor     # [n_comp_tok] which completion each token belongs to
    n_completion_tokens: int = field(default=0)


def build_shared_prefix_layout(
    prefix_len: int, completion_lens: List[int], device="cpu"
) -> SharedPrefixLayout:
    """Build the layout for a single group from its prefix length and completion lengths.

    The crux is the logprob fan-out: for the first token of each completion (local t == 0)
    the predicting position is the prefix's last token ``Lp - 1`` -- the same shared logit
    for every branch. For t >= 1 it is the immediately preceding (same-branch) token.
    """
    Lp = int(prefix_len)
    Lc = [int(x) for x in completion_lens]
    assert Lp >= 1, "prefix must be non-empty (need a last-prefix logit to score C_i[0])"

    positions: List[int] = list(range(Lp))   # prefix: 0..Lp-1
    segments: List[int] = [0] * Lp            # prefix segment id = 0
    branch_starts: List[int] = []
    comp_positions: List[int] = []
    prev_positions: List[int] = []
    branch_of_token: List[int] = []

    cursor = Lp
    for i, lc in enumerate(Lc):
        branch_starts.append(cursor)
        positions.extend(range(Lp, Lp + lc))  # completion continues from Lp
        segments.extend([i + 1] * lc)
        for t in range(lc):
            p = cursor + t
            comp_positions.append(p)
            prev_positions.append(Lp - 1 if t == 0 else p - 1)
            branch_of_token.append(i)
        cursor += lc

    total = cursor

    def _t(xs):
        return torch.tensor(xs, dtype=torch.long, device=device)

    return SharedPrefixLayout(
        prefix_len=Lp,
        completion_lens=Lc,
        total_len=total,
        branch_starts=branch_starts,
        position_ids=_t(positions),
        segment_ids=_t(segments),
        comp_positions=_t(comp_positions),
        prev_positions=_t(prev_positions),
        branch_of_token=_t(branch_of_token),
        n_completion_tokens=len(comp_positions),
    )


def dense_tree_mask(layout: SharedPrefixLayout, device=None) -> torch.Tensor:
    """Return a ``[total_len, total_len]`` boolean ``allowed[q, k]`` mask.

    ``allowed`` iff k is causally before q (by packed index) AND (k is in the prefix OR k
    is in the same completion branch as q). This is the reference (dense-SDPA) realization
    of the tree mask; the kernel realization is ``flex_mask_mod`` below.
    """
    device = device if device is not None else layout.segment_ids.device
    seg = layout.segment_ids.to(device)
    n = layout.total_len
    idx = torch.arange(n, device=device)
    causal = idx[None, :] <= idx[:, None]                  # [q, k]: k <= q
    k_is_prefix = seg[None, :] == 0                        # [1, k]
    same_branch = seg[None, :] == seg[:, None]             # [q, k]
    return causal & (k_is_prefix | same_branch)


def flex_mask_mod(layout: SharedPrefixLayout) -> Callable:
    """Return a FlexAttention ``mask_mod(b, h, q_idx, kv_idx) -> bool`` for this layout.

    Captures ``segment_ids`` by closure; index it on the same device the kernel runs on.
    Use with ``torch.nn.attention.flex_attention.create_block_mask``.
    """
    seg = layout.segment_ids

    def mask_mod(b, h, q_idx, kv_idx):
        s = seg.to(q_idx.device)
        causal = kv_idx <= q_idx
        k_is_prefix = s[kv_idx] == 0
        same_branch = s[kv_idx] == s[q_idx]
        return causal & (k_is_prefix | same_branch)

    return mask_mod


def build_packed_group(prompt_ids, completion_ids_list, device=None):
    """Pack one GRPO group ``[P, C_1, ..., C_G]`` for a single shared-prefix forward.

    Args:
        prompt_ids: 1-D LongTensor, the shared prompt P.
        completion_ids_list: list of 1-D LongTensors, the per-rollout completions C_i.

    Returns:
        packed_tokens: ``[1, total_len]`` token ids.
        layout: the :class:`SharedPrefixLayout` (carries ``position_ids`` to feed RoPE and
            the fan-out indices for logprob extraction).
        attn_mask: ``[1, 1, total_len, total_len]`` bool tree mask in MEGATRON convention
            (``True`` == masked out), i.e. ``~allowed`` -- pass straight as the model's
            ``attention_mask`` on the dense (non-THD) attention path.

    IMPORTANT: the model must apply RoPE using ``layout.position_ids`` (prefix-continued:
    P -> 0..Lp-1, each C_i -> Lp..Lp+Lc_i-1). Standard Megatron RoPE derives positions from
    the packed sequence length (``get_rotary_seq_len``) and IGNORES ``position_ids``, which
    would give C_i its packed index instead of Lp+t and break equivalence -- making RoPE
    position-aware is the core Milestone-1b integration task.
    """
    device = device if device is not None else prompt_ids.device
    prompt = prompt_ids.to(device)
    comps = [c.to(device) for c in completion_ids_list]
    layout = build_shared_prefix_layout(prompt.numel(), [c.numel() for c in comps], device)
    packed_tokens = torch.cat([prompt] + comps).unsqueeze(0)              # [1, total]
    allowed = dense_tree_mask(layout, device)                            # [T, T] True==allowed
    attn_mask = (~allowed).unsqueeze(0).unsqueeze(0)                     # [1,1,T,T] True==masked
    return packed_tokens, layout, attn_mask


def positionwise_rotary_emb(rotary_module, position_ids: torch.Tensor) -> torch.Tensor:
    """Per-token RoPE embedding for ARBITRARY (e.g. prefix-continued) positions.

    Megatron's ``RotaryEmbedding`` indexes the rotary table by absolute sequence position
    (``get_emb(max_seq_len)`` -> ``[max_seq_len, 1, 1, dim]``, applied positionally by the
    decoder). For a shared-prefix bin ``[P, C_1, ..., C_G]`` we instead want token i to use
    the rotary for ``position_ids[i]`` (P -> 0..Lp-1, each C_i -> Lp..). This computes the
    table up to ``max(position_ids)+1`` and gathers, returning the same ``[T, 1, 1, dim]``
    shape the decoder expects.
    """
    max_pos = int(position_ids.max().item()) + 1
    emb = rotary_module.get_emb(max_pos)                       # [max_pos, 1, 1, dim]
    return emb.index_select(0, position_ids.to(emb.device))   # [T, 1, 1, dim]


def _find_rotary_module(model):
    """Locate the ``rotary_pos_emb`` module, unwrapping Float16Module/DDP wrappers."""
    m = model
    for _ in range(5):
        rot = getattr(m, 'rotary_pos_emb', None)
        if rot is not None:
            return rot
        m = getattr(m, 'module', None)
        if m is None:
            break
    raise AttributeError("model has no rotary_pos_emb (is RoPE enabled?)")


@contextlib.contextmanager
def rotary_position_aware(model, position_ids: torch.Tensor):
    """Temporarily make the model's RoPE honor ``position_ids`` instead of packed-index.

    Standard Megatron RoPE derives positions from the sequence length and ignores the
    ``position_ids`` argument, which is wrong for a shared-prefix bin (every branch after
    the first would be mis-phased -- see the oracle's ``[rope]`` check). This context
    manager monkeypatches the rotary module's ``forward`` to return
    ``positionwise_rotary_emb(..., position_ids)`` for the duration of one forward, then
    restores the original. Intended for the shared-prefix forward (Milestone 1b); a
    production path would thread positions through instead of patching.
    """
    rot = _find_rotary_module(model)
    pos = position_ids

    def _patched(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        return positionwise_rotary_emb(self, pos)

    had_own = 'forward' in rot.__dict__
    saved = rot.__dict__.get('forward', None)
    object.__setattr__(rot, 'forward', types.MethodType(_patched, rot))
    try:
        yield
    finally:
        if had_own:
            object.__setattr__(rot, 'forward', saved)
        else:
            try:
                object.__delattr__(rot, 'forward')
            except AttributeError:
                pass


def extract_completion_logprobs(
    logits: torch.Tensor, packed_tokens: torch.Tensor, layout: SharedPrefixLayout
) -> torch.Tensor:
    """Gather per-completion-token logprobs from a shared-prefix forward.

    Args:
        logits: ``[total_len, vocab]`` (single packed sequence, batch dim already removed).
        packed_tokens: ``[total_len]`` token ids of the packed ``[P, C_1, ..., C_G]`` seq.
        layout: the layout used to build the packed sequence.

    Returns:
        ``[n_completion_tokens]`` logprobs, ordered branch-by-branch (see
        ``layout.branch_of_token``). Each branch's first token is scored from the shared
        ``logits[Lp - 1]``; the rest from the preceding same-branch position.
    """
    logp = torch.log_softmax(logits[layout.prev_positions], dim=-1)   # [n_tok, vocab]
    targets = packed_tokens[layout.comp_positions].unsqueeze(-1)      # [n_tok, 1]
    return logp.gather(-1, targets).squeeze(-1)                        # [n_tok]
