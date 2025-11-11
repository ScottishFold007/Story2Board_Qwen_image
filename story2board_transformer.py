from typing import Any, Dict, Optional, Union

import numpy as np
import torch

from diffusers.configuration_utils import register_to_config
from diffusers.utils import USE_PEFT_BACKEND, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel
from .story2board_attention_processor import Story2BoardAttentionProcessor


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Story2BoardQwenTransformer2DModel(QwenImageTransformer2DModel):
    @register_to_config
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_attn_processor(Story2BoardAttentionProcessor())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_hidden_states_mask: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes=None,
        txt_seq_lens=None,
        guidance: torch.Tensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.Tensor, Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        attn_store = attention_kwargs.get('attn_store', None)
        n_prompt_tokens = attention_kwargs.get('n_prompt_tokens', 0)
        n_image_tokens = attention_kwargs.get('n_image_tokens', 0)

        x_emb = self.x_embedder(hidden_states)

        timestep = timestep.to(x_emb.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(x_emb.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        context_emb = self.context_embedder(encoder_hidden_states)

        ids = self.build_ids(img_shapes, txt_seq_lens, device=x_emb.device)
        image_rotary_emb = self.pos_embed(ids)

        for index_block, block in enumerate(self.transformer_blocks):
            if attn_store:
                attn_store.increment()

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                context_emb, x_emb = self._gradient_checkpointing_func(
                    block,
                    x_emb,
                    context_emb,
                    temb,
                    image_rotary_emb,
                    encoder_hidden_states_mask,
                    attention_kwargs,
                )
            else:
                context_emb, x_emb = block(
                    hidden_states=x_emb,
                    encoder_hidden_states=context_emb,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    attention_kwargs=attention_kwargs,
                )

            # LPA: 覆写上半图像 token（双路阶段）
            x_emb[1:, : x_emb.shape[1] // 2, :] = x_emb[0, : x_emb.shape[1] // 2, :]

            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                if controlnet_blocks_repeat:
                    x_emb = x_emb + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                else:
                    x_emb = x_emb + controlnet_block_samples[index_block // interval_control]

        x_and_ctx = torch.cat([context_emb, x_emb], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if attn_store:
                attn_store.increment()

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x_and_ctx = self._gradient_checkpointing_func(
                    block,
                    x_and_ctx,
                    temb,
                    image_rotary_emb,
                    encoder_hidden_states_mask,
                    attention_kwargs,
                )

            else:
                x_and_ctx = block(
                    hidden_states=x_and_ctx,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    attention_kwargs=attention_kwargs,
                )

            # LPA: 覆写上半图像 token（单路阶段，跳过文本 token ）
            x_and_ctx[1:, n_prompt_tokens : n_prompt_tokens + n_image_tokens // 2, :] = \
                x_and_ctx[0, n_prompt_tokens : n_prompt_tokens + n_image_tokens // 2, :]

            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                x_and_ctx[:, context_emb.shape[1] :, ...] = (
                    x_and_ctx[:, context_emb.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        x_emb = x_and_ctx[:, context_emb.shape[1] :, ...]

        x_emb = self.norm_out(x_emb, temb)
        output = self.proj_out(x_emb)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


