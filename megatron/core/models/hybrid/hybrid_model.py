# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
from collections.abc import Iterator
from typing import Literal, Optional

import torch
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.inference.utils import InferenceMode
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.hybrid.shared_prefix import (
    SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY,
    SHARED_PREFIX_TRAINING_CAPABILITIES,
    SharedPrefixLayout,
    _validate_shared_prefix_physical_length,
    forward_hybrid_stack_shared_prefix,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import InferenceCudaGraphScope, ModelType
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.moe.paged_stash import paged_stash_init_chunk_handler
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    mtp_on_this_rank,
    process_mtp_loss,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import (
    WrappedTensor,
    deprecate_inference_params,
    is_using_quantization_scales,
    log_single_rank,
)

logger = logging.getLogger(__name__)


def _hybrid_logging_pg_kwargs(pg_collection: ProcessGroupCollection) -> dict:
    tp_group = getattr(pg_collection, 'tp', None)
    dp_cp_group = getattr(pg_collection, 'dp_cp', None)
    if (tp_group is None) != (dp_cp_group is None):
        raise ValueError(
            "pg_collection.tp and pg_collection.dp_cp must both be set or both be unset."
        )
    if tp_group is None:
        return {}
    return {'tp_group': tp_group, 'dp_cp_group': dp_cp_group}


def _canonicalize_shared_prefix_cp_sequence(
    local_tensor: Tensor,
    layout: SharedPrefixLayout,
    physical_len: int,
    cp_group: torch.distributed.ProcessGroup,
    *,
    reduce_scatter_grad: bool,
) -> Tensor:
    """Gather one zigzag CP shard and restore canonical global star order."""
    cp_size = cp_group.size()
    if local_tensor.shape[0] * cp_size != physical_len:
        raise ValueError(
            "shared-prefix CP-local tensor length does not match the physical star: "
            f"{local_tensor.shape[0]} * {cp_size} != {physical_len}"
        )
    if cp_size == 1:
        return local_tensor

    rank_order_tensor = gather_from_sequence_parallel_region(
        local_tensor, tensor_parallel_output_grad=reduce_scatter_grad, group=cp_group
    )
    rank_order_indices = torch.cat(
        [
            layout.cp_local_indices(physical_len, cp_size, rank, local_tensor.device)
            for rank in range(cp_size)
        ]
    )
    inverse_order = torch.empty_like(rank_order_indices)
    inverse_order[rank_order_indices] = torch.arange(
        physical_len, device=local_tensor.device, dtype=torch.long
    )
    return rank_order_tensor.index_select(0, inverse_order)


def _iter_shared_prefix_mtp_branches(
    global_hidden_states: Tensor,
    global_input_ids: Tensor,
    global_loss_mask: Tensor,
    layout: SharedPrefixLayout,
    *,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """Yield independent dense MTP branches from one canonical star.

    The main Hybrid stack can share its prompt, but MTP shifts future tokens and
    therefore needs a conventional sequence boundary per completion.  Keeping
    only the current branch materialized avoids retaining the full dense
    expansion on every CP rank. When CP is enabled, global-star indices are
    composed directly with that branch's zigzag CP ownership so the full dense
    branch is never allocated.
    """
    physical_len = global_hidden_states.shape[0]
    if global_hidden_states.ndim != 3 or global_hidden_states.shape[1] != 1:
        raise ValueError("shared-prefix MTP hidden states must have shape [tokens, 1, hidden]")
    if global_input_ids.shape != (1, physical_len):
        raise ValueError("shared-prefix MTP input IDs must have shape [1, physical_tokens]")
    if global_loss_mask.shape != (1, physical_len):
        raise ValueError("shared-prefix MTP loss mask must have shape [1, physical_tokens]")
    if physical_len < layout.total_len:
        raise ValueError(
            f"shared-prefix MTP physical length {physical_len} is shorter than layout "
            f"length {layout.total_len}"
        )
    if layout.logical_completion_lens is None or layout.padding_multiple is None:
        raise NotImplementedError(
            "shared-prefix MTP requires explicit logical completion lengths and physical padding"
        )
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("shared-prefix MTP received an invalid CP size/rank")

    if torch.count_nonzero(global_loss_mask[:, : layout.prefix_len]).item():
        raise ValueError("shared-prefix MTP loss mask must exclude every prompt token")
    if torch.count_nonzero(global_loss_mask[:, layout.total_len :]).item():
        raise ValueError("shared-prefix MTP loss mask must exclude topology-only padding")

    prompt_indices = torch.arange(
        layout.prefix_len, device=global_hidden_states.device, dtype=torch.long
    )
    for branch_slice, logical_len in zip(
        layout.completion_slices(), layout.logical_completion_lens, strict=True
    ):
        physical_padding_start = branch_slice.start + logical_len
        if torch.count_nonzero(
            global_loss_mask[:, physical_padding_start : branch_slice.stop]
        ).item():
            raise ValueError(
                "shared-prefix MTP loss mask must exclude ordinary per-sequence padding"
            )
        indices = torch.cat(
            (
                prompt_indices,
                torch.arange(
                    branch_slice.start,
                    branch_slice.stop,
                    device=global_hidden_states.device,
                    dtype=torch.long,
                ),
            )
        )
        if cp_size > 1:
            cp_positions = SharedPrefixLayout.cp_local_indices(
                indices.numel(), cp_size, cp_rank, indices.device
            )
            indices = indices.index_select(0, cp_positions)
        yield (
            global_hidden_states.index_select(0, indices),
            global_input_ids.index_select(1, indices),
            global_loss_mask.index_select(1, indices),
        )


def _reconstruct_shared_prefix_mtp_branches(
    global_hidden_states: Tensor,
    global_input_ids: Tensor,
    global_loss_mask: Tensor,
    layout: SharedPrefixLayout,
) -> tuple[tuple[Tensor, Tensor, Tensor], ...]:
    """Materialize the iterator for focused reconstruction/parity tests."""
    return tuple(
        _iter_shared_prefix_mtp_branches(
            global_hidden_states, global_input_ids, global_loss_mask, layout
        )
    )


def _validate_shared_prefix_mtp_pattern(mtp_pattern: Optional[str]) -> None:
    """Keep the promoted predictor scope narrower than the Hybrid backbone scope."""
    if mtp_pattern is not None and 'M' in mtp_pattern:
        raise NotImplementedError(
            "shared-prefix MTP supports a Mamba Hybrid backbone but does not yet "
            "support a Mamba layer inside the MTP predictor; use an attention/MLP "
            "or attention/MoE MTP pattern such as '*-' or '*E'"
        )


def _hybrid_mtp_is_enabled(
    configured_num_layers: Optional[int], mtp_pattern: Optional[str], mtp_pattern_depths: int
) -> bool:
    """Return whether this Hybrid runtime should construct and execute MTP.

    Native Nemotron-H configs can retain MTP pattern metadata while a training
    recipe explicitly overrides ``mtp_num_layers=0``.  The pattern still
    describes the checkpoint architecture, but zero must disable the runtime
    MTP block and its post-processing path.
    """
    return bool(
        configured_num_layers is not None
        and configured_num_layers > 0
        and mtp_pattern is not None
        and mtp_pattern_depths > 0
    )


class HybridModel(LanguageModule, GraphableMegatronModule):
    """Hybrid language model.

    Args:
        config (TransformerConfig): Model config
        hybrid_stack_spec (ModuleSpec): Specifies the modules to use for the various layer types
        vocab_size (int): Vocabulary size
        max_sequence_length (int): maximum size of sequence.
            This is used for positional embedding
        hybrid_layer_pattern (str): Unified hybrid layer pattern with optional MTP and
            pipeline stage boundaries.
            Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."
            The main pattern may contain "|" to define pipeline stage boundaries.
            Examples:
                - "M*M*" -> main decoder only, no MTP
                - "M*M*/MM/MM" -> main="M*M*", mtp="MM", 2 depths
                - "M-M-|M-M*-|M-M-|M-M*-" -> 4 pipeline segments
        hybrid_attention_ratio (float, optional): Deprecated. Use hybrid_layer_pattern instead.
            If set to a value > 0.0 and hybrid_layer_pattern is None, a pattern will be
            generated from the ratio with a deprecation warning.
        hybrid_mlp_ratio (float, optional): Deprecated. Use hybrid_layer_pattern instead.
            If set to a value > 0.0 and hybrid_layer_pattern is None, a pattern will be
            generated from the ratio with a deprecation warning.
        hybrid_override_pattern (str, optional): Deprecated. Use hybrid_layer_pattern instead.
            If set and hybrid_layer_pattern is None, the value is copied to hybrid_layer_pattern
            with a deprecation warning.
        pre_process (bool, optional): Include embedding layer
            (used with pipeline parallelism). Defaults to True.
        post_process (bool, optional): Include an output layer (used with pipeline parallelism).
            Defaults to True.
        fp16_lm_cross_entropy (bool, optional): Defaults to False.
        logit_dtype (torch.dtype, optional): Dtype for the output-layer GEMM result.
            Defaults to None, which uses the hidden-state dtype.
        parallel_output (bool, optional): Do not gather the outputs, keep them split across tensor
            parallel ranks. Defaults to True.
        share_embeddings_and_output_weights (bool, optional): When True, input embeddings and
            output logit weights are shared. Defaults to False.
        position_embedding_type (Literal[learned_absolute,rope,yarn,none], optional):  Position
            embedding type. Defaults to 'none'.
        rotary_percent (float, optional): Percent of rotary dimension to use for rotary position
            embeddings. Ignored unless position_embedding_type is 'rope'. Defaults to 1.0.
        rotary_base (int, optional): Base period for rotary position embeddings. Ignored unless
            position_embedding_type is 'rope'. Defaults to 10000.
        seq_len_interpolation_factor (Optional[float], optional): scale of linearly
            interpolating RoPE for longer sequences. The value must be a float larger than 1.0.
             Defaults to None.
        pg_collection (ProcessGroupCollection, optional): Model communication process groups.
        vp_stage (Optional[int], optional): Virtual pipeline stage index. Defaults to None.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hybrid_stack_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        hybrid_layer_pattern: Optional[str] = None,
        hybrid_attention_ratio: Optional[float] = None,
        hybrid_mlp_ratio: Optional[float] = None,
        hybrid_override_pattern: Optional[str] = None,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        logit_dtype: Optional[torch.dtype] = None,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        # Mamba with no attention has no need for position embeddings, so none is default
        position_embedding_type: Literal['learned_absolute', 'rope', 'yarn', 'none'] = 'none',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection)

        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        if self.config.use_mup and not getattr(HybridModel, "mup_warning_printed", False):
            log_single_rank(
                logger,
                logging.WARNING,
                "MuP for HybridModel is experimental and not fully validated yet.",
            )
            HybridModel.mup_warning_printed = True

        self.hybrid_stack_spec: ModuleSpec = hybrid_stack_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.hybrid_layer_pattern = hybrid_layer_pattern
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.logit_dtype = logit_dtype
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type
        self.vp_stage = vp_stage
        self.disable_param_offloading = True

        # Backward compatibility for deprecated hybrid parameters
        if hybrid_override_pattern is not None:
            if self.hybrid_layer_pattern is None:
                log_single_rank(
                    logger,
                    logging.WARNING,
                    "hybrid_override_pattern has been deprecated. "
                    "Use hybrid_layer_pattern instead.",
                )
                self.hybrid_layer_pattern = hybrid_override_pattern
            else:
                raise ValueError(
                    "hybrid_override_pattern and hybrid_layer_pattern cannot both be set. "
                    "hybrid_override_pattern has been deprecated; use hybrid_layer_pattern instead."
                )
        if (hybrid_attention_ratio is not None and hybrid_attention_ratio > 0.0) or (
            hybrid_mlp_ratio is not None and hybrid_mlp_ratio > 0.0
        ):
            if hybrid_layer_pattern is not None:
                raise ValueError(
                    "hybrid_layer_pattern cannot be used together with "
                    "hybrid_attention_ratio or hybrid_mlp_ratio. "
                    "These ratios have been deprecated; use hybrid_layer_pattern alone."
                )
            log_single_rank(
                logger,
                logging.WARNING,
                "hybrid_attention_ratio and hybrid_mlp_ratio have been deprecated. "
                "Use hybrid_layer_pattern instead.",
            )
            if self.hybrid_layer_pattern is None:
                from megatron.core.models.hybrid.hybrid_layer_allocation import pattern_from_ratios

                attn_ratio = hybrid_attention_ratio if hybrid_attention_ratio else 0.0
                mlp_ratio = hybrid_mlp_ratio if hybrid_mlp_ratio else 0.0
                self.hybrid_layer_pattern = pattern_from_ratios(
                    config.num_layers, attn_ratio, mlp_ratio
                )

        # Parse unified pattern to extract main and MTP components, and
        # determine the pipeline segment for this model instance.
        from megatron.core.models.hybrid.hybrid_layer_allocation import (
            parse_hybrid_pattern,
            select_pipeline_segment,
        )

        parsed = parse_hybrid_pattern(self.hybrid_layer_pattern)
        self.mtp_pattern = parsed.mtp_pattern
        self.mtp_num_depths = parsed.mtp_num_depths

        logging_pg_kwargs = _hybrid_logging_pg_kwargs(self.pg_collection)

        layer_type_list, layer_offset = select_pipeline_segment(
            parsed.main_pattern or '',
            self.pg_collection.pp,
            vp_stage,
            first_stage_layers=self.config.num_layers_in_first_pipeline_stage,
            last_stage_layers=self.config.num_layers_in_last_pipeline_stage,
            **logging_pg_kwargs,
        )

        # Determine if MTP is needed (based on pattern parsing)
        self.mtp_process = (
            _hybrid_mtp_is_enabled(
                self.config.mtp_num_layers, self.mtp_pattern, self.mtp_num_depths
            )
            # The following forces MTP to be on the final pipeline stage. It might be more optimal
            # to split the hybrid layer pattern into pipeline stages before parsing the pattern for
            # the current pipeline stage. This could also enable MTP standalone (MTP in a pipeline
            # stage separate from loss) to be supported in the hybrid model.
            and mtp_on_this_rank(
                layout=self.config.pipeline_model_parallel_layout,
                mtp_num_layers=self.config.mtp_num_layers,
                ignore_virtual=False,
                vp_stage=self.vp_stage,
            )
        )

        # megatron core pipelining currently depends on model type
        # TODO: remove this dependency ?
        self.model_type = ModelType.encoder_or_decoder

        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                scatter_to_sequence_parallel=scatter_embedding_sequence_parallel,
                tp_group=self.pg_collection.tp,
                pg_collection=self.pg_collection,
            )

        # MLA (also used by DeepSeek Sparse Attention) uses its own decoupled RoPE, therefore we do
        # not build standard RoPE here when using MLA.
        if self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )
        elif self.position_embedding_type == 'yarn':
            self.rotary_pos_emb = YarnRotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                scaling_factor=getattr(self.config, "yarn_rotary_scaling_factor"),
                original_max_position_embeddings=getattr(
                    self.config, "yarn_original_max_position_embeddings"
                ),
                beta_fast=getattr(self.config, "yarn_beta_fast"),
                beta_slow=getattr(self.config, "yarn_beta_slow"),
                mscale=getattr(self.config, "yarn_mscale"),
                mscale_all_dim=getattr(self.config, "yarn_mscale_all_dim"),
                correction_range_round_to_int=getattr(
                    self.config, "yarn_correction_range_round_to_int"
                ),
                use_cpu_initialization=self.config.use_cpu_initialization,
                cp_group=self.pg_collection.cp,
            )
        self.decoder = build_module(
            hybrid_stack_spec,
            self.config,
            pre_process=self.pre_process,
            layer_type_list=layer_type_list,
            pp_layer_offset=layer_offset,
            post_process=self.post_process,
            dtype=config.params_dtype,
            pg_collection=self.pg_collection,
            name="decoder",
        )

        # MTP block - uses mtp_block_spec from hybrid_stack_spec.submodules
        if self.mtp_process:
            hybrid_submodules = hybrid_stack_spec.submodules
            mtp_block_spec = hybrid_submodules.mtp_block_spec
            assert mtp_block_spec is not None, (
                "MTP pattern specified but mtp_block_spec is None in hybrid_stack_spec.submodules. "
                "Ensure hybrid_stack_spec includes mtp_block_spec for MTP support."
            )

            self.mtp = MultiTokenPredictionBlock(
                config=self.config,
                spec=mtp_block_spec,
                pg_collection=self.pg_collection,
                vp_stage=self.vp_stage,
                mtp_layer_pattern=self.mtp_pattern,
                mtp_num_depths=self.mtp_num_depths,
                hybrid_submodules=hybrid_submodules,
                name="mtp",
            )
            self._setup_mtp_cuda_graphs()

        # Output
        if post_process or self.mtp_process:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=(
                    config.embedding_init_method
                    if config.use_mup and not self.share_embeddings_and_output_weights
                    else config.init_method
                ),
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                tp_group=self.pg_collection.tp,
                output_dtype=self.logit_dtype,
                pg_collection=self.pg_collection,
            )

        if self.pre_process or self.post_process or self.mtp_process:
            self.setup_embeddings_and_output_layer()

        for name, module in self.named_modules():
            if hasattr(module, 'finish_init'):
                quant_config = get_quant_config_or_none(name, self.config.quant_recipe)
                module.finish_init(quant_config)

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        self.decoder.set_input_tensor(input_tensor[0])

    def preprocess_for_fine_grained_offloading(self):
        """Preprocess for fine-grained activation offloading."""
        off_interface.init_chunk_handler(
            pp_rank=self.pg_collection.pp.rank(),
            vp_size=self.config.virtual_pipeline_model_parallel_size,
            vp_stage=self.vp_stage,
            min_offloaded_tensor_size=self.config.min_offloaded_tensor_size,
            delta_offload_bytes_across_pp_ranks=self.config.delta_offload_bytes_across_pp_ranks,
            activation_offload_fraction=self.config.activation_offload_fraction,
            max_inflight_offloads=self.config.fine_grained_offloading_max_inflight_offloads,
        )
        if self.disable_param_offloading:
            for param in self.decoder.parameters():
                off_interface.mark_not_offload(param)
            if self.mtp_process:
                for param in self.mtp.parameters():
                    off_interface.mark_not_offload(param)
            if self.post_process:
                for param in self.output_layer.parameters():
                    off_interface.mark_not_offload(param)
            self.disable_param_offloading = False

    def preprocess_for_paged_stash(self):
        """Preprocess for paged stash."""
        return paged_stash_init_chunk_handler(
            vp_size=self.config.virtual_pipeline_model_parallel_size, vp_stage=self.vp_stage
        )

    def _should_call_local_cudagraph(self, *args, **kwargs):
        """
        Check if we should call the local cudagraph path.
        """
        if (
            InferenceMode.is_active()
            and hasattr(self, 'cudagraph_manager')
            and (
                kwargs.get('inference_context') is not None
                or kwargs.get('inference_params') is not None
            )
            and self.config.inference_cuda_graph_scope == InferenceCudaGraphScope.block
        ):
            if kwargs['inference_context'].is_static_batching():
                using_cuda_graph = kwargs['inference_context'].is_decode_only()
            else:
                using_cuda_graph = kwargs['inference_context'].using_cuda_graph_this_step()

            if using_cuda_graph:
                return True
        return False

    def __call__(self, *args, **kwargs):
        if self._should_call_local_cudagraph(*args, **kwargs):
            return super().__call__(*args, **kwargs)[0]
        return super().__call__(*args, **kwargs)

    def create_mcore_cudagraph_manager(self, config):
        """
        Create the cudagraph manager for the full iteration inference scope
        """
        if config.inference_cuda_graph_scope == InferenceCudaGraphScope.block:
            from megatron.core.transformer.cuda_graphs import CudaGraphManager

            self.cudagraph_manager = CudaGraphManager(config)

    def _forward_shared_prefix_mtp(
        self,
        *,
        hidden_states: Tensor,
        input_ids: Tensor,
        loss_mask: Tensor,
        layout: SharedPrefixLayout,
        output_weight: Optional[Tensor],
        runtime_gather_output: Optional[bool],
    ) -> Tensor:
        """Run MTP heads on dense branches while preserving the shared backbone graph."""
        tp_group = self.pg_collection.tp
        cp_group = self.pg_collection.cp
        tp_size = tp_group.size()
        cp_size = cp_group.size()
        physical_len = input_ids.shape[1] * cp_size

        # The shared backbone is SP-sharded over TP.  Gather without reducing
        # gradients because the reconstructed branches are scattered over TP
        # again below, so each sequence position has one downstream owner.
        cp_local_hidden = hidden_states
        if tp_size > 1:
            cp_local_hidden = gather_from_sequence_parallel_region(
                cp_local_hidden, tensor_parallel_output_grad=False, group=tp_group
            )

        global_hidden = _canonicalize_shared_prefix_cp_sequence(
            cp_local_hidden, layout, physical_len, cp_group, reduce_scatter_grad=True
        )
        global_input_ids = _canonicalize_shared_prefix_cp_sequence(
            input_ids.transpose(0, 1).contiguous(),
            layout,
            physical_len,
            cp_group,
            reduce_scatter_grad=False,
        ).transpose(0, 1)
        global_loss_mask = _canonicalize_shared_prefix_cp_sequence(
            loss_mask.transpose(0, 1).contiguous(),
            layout,
            physical_len,
            cp_group,
            reduce_scatter_grad=False,
        ).transpose(0, 1)
        global_branch_lengths = [
            layout.prefix_len + completion_len for completion_len in layout.completion_lens
        ]
        if any(branch_len % cp_size for branch_len in global_branch_lengths):
            raise ValueError("shared-prefix MTP branch length must be divisible by CP size")
        combined_cp_length = sum(global_branch_lengths) // cp_size
        if combined_cp_length % tp_size:
            raise ValueError(
                "shared-prefix MTP combined CP-local sequence must be divisible by TP size"
            )
        packed_sp_length = combined_cp_length // tp_size
        packed_sp_start = tp_group.rank() * packed_sp_length
        packed_sp_stop = packed_sp_start + packed_sp_length

        depth_count = 1 + self.config.mtp_num_layers
        outputs_by_depth = [[] for _ in range(depth_count)]
        branch_input_ids = []
        branch_loss_masks = []
        branch_cp_offset = 0
        branch_iterator = _iter_shared_prefix_mtp_branches(
            global_hidden,
            global_input_ids,
            global_loss_mask,
            layout,
            cp_size=cp_size,
            cp_rank=cp_group.rank(),
        )
        for (branch_hidden, branch_ids, branch_mask), branch_len in zip(
            branch_iterator, global_branch_lengths, strict=True
        ):
            required_alignment = tp_size if cp_size == 1 else 2 * cp_size * tp_size
            if branch_len % required_alignment:
                raise ValueError(
                    "shared-prefix MTP physical branch length must be divisible by "
                    f"the CP/TP sequence quantum {required_alignment}, got {branch_len}"
                )
            branch_cp_indices = (
                torch.arange(branch_len, device=branch_ids.device, dtype=torch.long)
                if cp_size == 1
                else SharedPrefixLayout.cp_local_indices(
                    branch_len, cp_size, cp_group.rank(), branch_ids.device
                )
            )
            local_hidden = branch_hidden
            local_ids = branch_ids
            local_mask = branch_mask
            local_position_ids = torch.arange(
                branch_len, device=branch_ids.device, dtype=torch.long
            ).index_select(0, branch_cp_indices)[None, :]
            if tp_size > 1:
                local_hidden = tensor_parallel.scatter_to_sequence_parallel_region(
                    local_hidden, group=tp_group
                )

            # RotaryEmbedding owns the standard CP zigzag slice.  MTP receives
            # one ordinary sequence here, so no forest positions are involved.
            if self.position_embedding_type == 'rope':
                branch_rotary_pos_emb = self.rotary_pos_emb(branch_len)
            elif self.position_embedding_type == 'none':
                branch_rotary_pos_emb = None
            else:
                raise NotImplementedError(
                    "shared-prefix MTP supports only RoPE or positionless Hybrid models"
                )
            branch_output = self.mtp(
                input_ids=local_ids,
                position_ids=local_position_ids,
                hidden_states=local_hidden,
                attention_mask=None,
                inference_params=None,
                rotary_pos_emb=branch_rotary_pos_emb,
                packed_seq_params=None,
                embedding=self.embedding,
            )
            branch_input_ids.append(local_ids)
            branch_loss_masks.append(local_mask)
            branch_chunks = torch.chunk(branch_output, depth_count, dim=0)
            if len(branch_chunks) != depth_count:
                raise RuntimeError("shared-prefix MTP returned an invalid prediction depth count")
            branch_cp_length = branch_len // cp_size
            for depth, chunk in enumerate(branch_chunks):
                if tp_size == 1:
                    outputs_by_depth[depth].append(chunk)
                    continue

                # Gather only the current branch/depth, then retain only this
                # rank's slice of the eventual branch-major combined SP shard.
                # reduce-scatter backward is required because different ranks
                # select disjoint pieces from their replicated gathered branch.
                full_branch_chunk = gather_from_sequence_parallel_region(
                    chunk, tensor_parallel_output_grad=True, group=tp_group
                )
                overlap_start = max(branch_cp_offset, packed_sp_start)
                overlap_stop = min(branch_cp_offset + branch_cp_length, packed_sp_stop)
                if overlap_start < overlap_stop:
                    source_start = overlap_start - branch_cp_offset
                    source_stop = overlap_stop - branch_cp_offset
                    outputs_by_depth[depth].append(
                        full_branch_chunk[source_start:source_stop].contiguous()
                    )
                else:
                    # Retain this gather in every TP rank's autograd graph even
                    # when the rank owns no tokens from this branch.  Omitting
                    # it lets backward issue the same number of reduce-scatters
                    # in a different branch order on different ranks, silently
                    # pairing unrelated branch gradients.
                    outputs_by_depth[depth].append(full_branch_chunk[:0])
                del full_branch_chunk
            branch_cp_offset += branch_cp_length
            del branch_output, branch_hidden

        # MTP layers run independently so variable-length Mamba branches never
        # see synthetic batch padding.  Repack their outputs by prediction depth
        # for one globally normalized loss calculation.  PackedSeqParams is used
        # only by process_mtp_loss's token/mask shifts, not by an MTP model layer.
        packed_depth_hidden_states = []
        for depth_outputs in outputs_by_depth:
            packed_depth = torch.cat(depth_outputs, dim=0)
            if packed_depth.shape[0] != packed_sp_length:
                raise RuntimeError(
                    "shared-prefix MTP branch-major SP shard has an invalid length: "
                    f"{packed_depth.shape[0]} != {packed_sp_length}"
                )
            packed_depth_hidden_states.append(packed_depth)
        packed_hidden_states = torch.cat(packed_depth_hidden_states, dim=0)
        packed_input_ids = torch.cat(branch_input_ids, dim=1)
        packed_loss_mask = torch.cat(branch_loss_masks, dim=1)
        cumulative_lengths = [0]
        for branch_len in global_branch_lengths:
            cumulative_lengths.append(cumulative_lengths[-1] + branch_len)
        cu_seqlens = torch.tensor(
            cumulative_lengths, device=packed_input_ids.device, dtype=torch.int32
        )
        packed_seq_params = PackedSeqParams(
            qkv_format='thd',
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(global_branch_lengths),
            max_seqlen_kv=max(global_branch_lengths),
            local_cp_size=cp_size,
            cp_group=cp_group,
        )
        processed_mtp_hidden = process_mtp_loss(
            hidden_states=packed_hidden_states,
            labels=None,
            loss_mask=packed_loss_mask,
            output_layer=self.output_layer,
            output_weight=output_weight,
            runtime_gather_output=runtime_gather_output,
            is_training=self.training,
            compute_language_model_loss=self.compute_language_model_loss,
            config=self.config,
            cp_group=cp_group,
            tp_group=self.tp_group,
            packed_seq_params=packed_seq_params,
            scale_logits_fn=self._scale_logits if self.config.use_mup else None,
            input_ids=packed_input_ids,
        )

        # The external RL loss consumes star logits.  A zero-valued attachment
        # retains process_mtp_loss's MTPLossAutoScaler nodes without changing
        # those logits; its backward hook supplies the auxiliary-loss gradient.
        mtp_loss_anchor = processed_mtp_hidden.reshape(-1)[0] * 0.0
        return hidden_states + mtp_loss_anchor

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context: BaseInferenceContext = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params: Optional[BaseInferenceContext] = None,
        loss_mask: Optional[Tensor] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        padding_mask: Optional[Tensor] = None,
        shared_prefix_layout: Optional[SharedPrefixLayout] = None,
    ) -> Tensor:
        """Forward function of the Hybrid model. This function passes the input tensors
        through the embedding layer, and then the decoder and finally into the post
        processing layer (optional).

        It either returns the Loss values if labels are given or the final hidden units

        ``shared_prefix_layout`` explicitly selects the star path. The global packed input is
        ``[prefix, completion_1, ..., completion_G, optional_topology_padding]`` with batch size
        one; CP1 receives it whole, while CP>1 receives standard two-chunk zigzag sequence shards.
        With TP>1, input IDs remain CP-local and replicated over TP; sequence parallelism starts at
        the embedding output. The layout owns the exact tree mask and prefix-continued RoPE
        positions. The normal decoder path is unchanged when the argument is ``None``. With
        ``labels=None`` and parallel output, logits remain
        ``[1, physical_len/CP, padded_vocab/TP]`` in the input shard's zigzag token order; this
        model gathers neither sequence across CP nor vocabulary across TP.
        """
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.

        if self.config.fine_grained_activation_offloading:
            self.preprocess_for_fine_grained_offloading()

        if self.config.moe_paged_stash:
            self.preprocess_for_paged_stash()

        inference_context = deprecate_inference_params(inference_context, inference_params)

        in_inference_mode = InferenceMode.is_active()

        if shared_prefix_layout is not None:
            if in_inference_mode or inference_context is not None:
                raise NotImplementedError("shared-prefix Hybrid forward is training-only")
            if attention_mask is not None:
                raise ValueError(
                    "shared-prefix Hybrid forward owns its tree mask and requires attention_mask=None"
                )
            if labels is not None:
                raise NotImplementedError(
                    "shared-prefix Hybrid forward requires external next-token loss computation"
                )
            if packed_seq_params is not None:
                raise ValueError(
                    "shared_prefix_layout and packed_seq_params are mutually exclusive"
                )
            if padding_mask is not None:
                raise ValueError("shared-prefix Hybrid forward does not accept padding_mask")
            if not self.pre_process or not self.post_process or self.vp_stage is not None:
                raise NotImplementedError(
                    "shared-prefix HybridModel forward currently requires a complete PP1 model"
                )
            if (
                self.mtp_process
                and self.training
                and SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY
                not in SHARED_PREFIX_TRAINING_CAPABILITIES
            ):
                raise NotImplementedError(
                    "shared-prefix Hybrid MTP dense-head support is implemented but not "
                    "advertised for production; distributed forward/backward parity must "
                    f"promote capability {SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY!r}"
                )
            if self.mtp_process and self.training:
                _validate_shared_prefix_mtp_pattern(self.mtp_pattern)
            if self.mtp_process and self.training and loss_mask is None:
                raise ValueError("shared-prefix Hybrid MTP requires an explicit loss mask")
            if self.position_embedding_type not in ('rope', 'none'):
                raise NotImplementedError(
                    "shared-prefix Hybrid forward supports only RoPE or positionless attention"
                )
            if self.config.multi_latent_attention:
                raise NotImplementedError(
                    "shared-prefix Hybrid forward does not support multi-latent attention"
                )
            tp_size = self.pg_collection.tp.size()
            cp_size = self.pg_collection.cp.size()
            if tp_size > 1 and not self.config.sequence_parallel:
                raise NotImplementedError(
                    "shared-prefix HybridModel TP>1 requires sequence parallelism"
                )
            if tp_size == 1 and self.config.sequence_parallel:
                raise NotImplementedError(
                    "shared-prefix HybridModel sequence parallelism requires TP>1"
                )
            if self.config.tensor_model_parallel_size != tp_size:
                raise RuntimeError(
                    "shared-prefix HybridModel tensor-parallel config does not match its "
                    "process group"
                )
            if tp_size > 1 and (not self.parallel_output or runtime_gather_output):
                raise NotImplementedError(
                    "shared-prefix HybridModel TP/SP requires TP-sharded parallel output logits"
                )
            if decoder_input is None:
                if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1:
                    raise ValueError(
                        "shared-prefix Hybrid input_ids must have shape [1, physical_len/CP]"
                    )
                physical_len = input_ids.shape[1] * cp_size
                _validate_shared_prefix_physical_length(
                    shared_prefix_layout,
                    physical_len,
                    tp_size=tp_size,
                    cp_size=cp_size,
                    sequence_parallel=self.config.sequence_parallel,
                )

        if in_inference_mode:
            assert runtime_gather_output, "Inference must always gather TP logits"

        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)

            # Clear the outputs for padding tokens when using dynamic batching with
            # quantization scales to avoid corrupting amax calculations
            if (
                in_inference_mode
                and inference_context is not None
                and inference_context.is_dynamic_batching()
                and is_using_quantization_scales(self.config)
            ):
                decoder_input[inference_context.padding_slice] = 0.0

            if self.config.sequence_parallel and not self.embedding.scatter_to_sequence_parallel:
                # The embedding skips SP scatter for models whose outer wrapper scatters instead
                # (e.g. VLM LMs); scatter here so a standalone LM forward isn't double-gathered.
                decoder_input = tensor_parallel.scatter_to_sequence_parallel_region(
                    decoder_input, group=self.pg_collection.tp
                )
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        rotary_pos_emb = None
        if shared_prefix_layout is not None and self.position_embedding_type == 'rope':
            if decoder_input is None:
                raise RuntimeError("shared-prefix Hybrid embedding did not produce decoder input")
            cp_group = self.pg_collection.cp
            cp_size = cp_group.size()
            tp_size = self.pg_collection.tp.size()
            sequence_shards = tp_size if self.config.sequence_parallel else 1
            physical_len = decoder_input.shape[0] * cp_size * sequence_shards
            if decoder_input.ndim != 3 or decoder_input.shape[1] != 1:
                raise ValueError(
                    "shared-prefix Hybrid decoder input must have shape "
                    "[physical_len/(TP*CP), 1, hidden] when sequence parallelism is enabled"
                )
            rotary_table = self.rotary_pos_emb.get_emb(
                shared_prefix_layout.prefix_len + max(shared_prefix_layout.completion_lens)
            )
            global_position_ids = shared_prefix_layout.padded_position_ids(
                physical_len, rotary_table.device
            )
            if cp_size > 1:
                local_indices = shared_prefix_layout.cp_local_indices(
                    physical_len, cp_size, cp_group.rank(), rotary_table.device
                )
                global_position_ids = global_position_ids.index_select(0, local_indices)
            rotary_pos_emb = rotary_table.index_select(0, global_position_ids)
        elif self.position_embedding_type == 'rope' and not self.config.multi_latent_attention:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config, packed_seq_params
            )
            rotary_pos_emb = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
            )
        elif self.position_embedding_type == 'yarn':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_context, self.decoder, decoder_input, self.config, packed_seq_params
            )
            # YarnRotaryEmbedding.forward returns (emb, mscale); discard mscale here
            rotary_pos_emb, _ = self.rotary_pos_emb(
                rotary_seq_len,
                packed_seq=packed_seq_params is not None and packed_seq_params.qkv_format == 'thd',
            )

        # Wrap decoder_input to allow the decoder (HybridStack) to delete the
        # reference held by this caller function, enabling early garbage collection
        # for inference.
        if in_inference_mode:
            decoder_input = WrappedTensor(decoder_input)

        # The following assert will currently fail when running inference.
        # Commented out for now.
        # TODO (duncan/rwaleffe): (1) confirm that the externally-generated
        #   attention mask is not needed and is ignored by the model in
        #   inference mode, (2) reduce the size of the externally-generated
        #   attention mask to prevent CPU OOM (as we did for training), (3)
        #   force the attention mask passed to the model in inference mode to
        #   be None, so this assert will succeed.
        # assert attention_mask is None, "The attention mask is ignored and should be set to None"

        # Run decoder.
        if shared_prefix_layout is not None:
            hidden_states = forward_hybrid_stack_shared_prefix(
                self.decoder,
                decoder_input,
                shared_prefix_layout,
                rotary_pos_emb=rotary_pos_emb,
                position_embedding_type=self.position_embedding_type,
            )
        else:
            hidden_states = self.decoder(
                hidden_states=decoder_input,
                attention_mask=attention_mask,
                inference_context=inference_context,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                padding_mask=padding_mask,
            )

        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()

        # Check if speculative decoding is active. When it is, MTP must be
        # computed *after* verification so that it is conditioned on verified
        # tokens rather than stale speculative tokens from the previous step.
        is_spec_decode = (
            in_inference_mode
            and inference_context is not None
            and inference_context.is_dynamic_batching()
            and inference_context.num_speculative_tokens > 0
        )

        mtp_forward_ran = self.mtp_process and not (
            in_inference_mode
            or is_spec_decode
            or (shared_prefix_layout is not None and not self.training)
        )
        if mtp_forward_ran:
            if shared_prefix_layout is not None:
                hidden_states = self._forward_shared_prefix_mtp(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    loss_mask=loss_mask,
                    layout=shared_prefix_layout,
                    output_weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
            else:
                hidden_states = self.mtp(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    inference_params=inference_params,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                    embedding=self.embedding,
                )

        if not self.post_process:
            return hidden_states

        if self.mtp_process:
            assert self.config.mtp_num_layers > 0
            if is_spec_decode:
                assert inference_context is not None
                if self.config.inference_cuda_graph_scope == InferenceCudaGraphScope.block:
                    # Block-scope CUDA graph mode: copy_() into the
                    # pre-allocated buffer so every graph replay writes to
                    # the same fixed GPU address regardless of batch size.
                    assert inference_context.mtp_decoder_hidden_states is not None
                    inference_context.mtp_decoder_hidden_states[: hidden_states.shape[0]].copy_(
                        hidden_states
                    )
                else:
                    # Non-block scope: direct assignment; the controller will set
                    # this back to None after reading to allow GC.
                    inference_context.mtp_decoder_hidden_states = hidden_states
            elif not in_inference_mode and shared_prefix_layout is None:
                # For RL (labels is None), process_mtp_loss derives labels from
                # input_ids to match the SFT label format.
                hidden_states = process_mtp_loss(
                    hidden_states=hidden_states,
                    labels=labels,
                    loss_mask=loss_mask,
                    output_layer=self.output_layer,
                    output_weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                    is_training=self.training,
                    compute_language_model_loss=self.compute_language_model_loss,
                    config=self.config,
                    cp_group=self.pg_collection.cp,
                    tp_group=self.tp_group,
                    packed_seq_params=packed_seq_params,
                    scale_logits_fn=self._scale_logits if self.config.use_mup else None,
                    input_ids=input_ids,
                )
        sequence_parallel_override = False
        if (
            in_inference_mode
            and inference_context is not None
            and inference_context.config.materialize_only_last_token_logits
        ):
            if inference_context.is_static_batching():
                hidden_states = hidden_states[-1:, :, :]
            else:
                if self.output_layer.sequence_parallel:
                    # Perform the sequence parallel gather here instead of after the output layer
                    # because we need to slice the last token logits from the full view of the
                    # packed logits across all requests.
                    hidden_states = gather_from_sequence_parallel_region(
                        hidden_states, group=self.pg_collection.tp
                    )
                    self.output_layer.sequence_parallel = False
                    sequence_parallel_override = True

                # Reshape [S, B, H] (with B=1) to [1, S, H] for logit extraction,
                # then back to [S', B, H] for the output layer.
                reshaped = hidden_states.squeeze(1).unsqueeze(0)
                hidden_states = inference_context.last_token_logits(reshaped).unsqueeze(1)

        logits, _ = self.output_layer(
            hidden_states, weight=output_weight, runtime_gather_output=runtime_gather_output
        )
        logits = self._scale_logits(logits)

        # Restore sequence parallel execution to the output layer if necessary.
        if sequence_parallel_override:
            assert (
                in_inference_mode
                and inference_context.is_dynamic_batching()
                and inference_context.config.materialize_only_last_token_logits
            )
            self.output_layer.sequence_parallel = True

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()

        loss = self.compute_language_model_loss(labels, logits)

        return loss
