import torch
from einops import pack, reduce


class AttentionStore:
    def __init__(self, batch_size, n_diff_steps, n_trans_blocks, n_image_tokens, n_attn_heads, dtype, device):
        self.n_diff_steps = n_diff_steps
        self.n_trans_blocks = n_trans_blocks
        self.n_image_tokens = n_image_tokens
        self.n_attn_heads = n_attn_heads
        self.concept_token_indices = range(self.n_image_tokens)
        self.curr_iter = -1
        self.internal_device = device

        self.attn_map_shape = (batch_size, self.n_image_tokens, self.n_image_tokens // 2)

        self.avg_attn_map = torch.zeros(self.attn_map_shape, device=self.internal_device, dtype=dtype)

        self.curr_diff_step_maps = []
        self.attn_map_decay = 0.8

    def increment(self):
        self.curr_iter += 1

    def _get_curr_diffusion_step(self):
        return self.curr_iter // self.n_trans_blocks

    def _is_first_layer(self):
        return self.curr_iter % self.n_trans_blocks == 0

    def _is_last_block(self):
        return self.curr_iter % self.n_trans_blocks == self.n_trans_blocks - 1

    def _get_curr_trans_block(self):
        return self.curr_iter % self.n_trans_blocks
    
    def store_attention_map(self, attn_map):
        assert attn_map.shape == self.attn_map_shape, "Attention map dimensions are incorrect"

        attn_map = attn_map.to(device=self.internal_device)
        self.curr_diff_step_maps.append(attn_map)

        if self._is_last_block():
            step_avg_attn_map, _ = pack(self.curr_diff_step_maps, 'c * v_toks v_toks2')
            step_avg_attn_map = reduce(step_avg_attn_map, 'channel layer v_toks v_toks2 -> channel v_toks v_toks2', 'mean')

            curr_step = self._get_curr_diffusion_step()
            self.curr_diff_step_maps = []

            new_observation = step_avg_attn_map - self.avg_attn_map
            self.avg_attn_map = (self.attn_map_decay * self.avg_attn_map
                                       + (1 - self.attn_map_decay) * new_observation / (curr_step + 1))

    def aggregate_attn_maps(self):
        return self.avg_attn_map


