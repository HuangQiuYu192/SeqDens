import copy

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss


class WEARecLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.num_heads = config["num_heads"]
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.head_dim = self.hidden_size // self.num_heads
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.freq_bins = self.seq_len // 2 + 1
        self.alpha = config["alpha"]
        self.combine_mode = "gate"
        self.adaptive = True
        self.layer_norm_eps = config["layer_norm_eps"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]

        self.complex_weight = nn.Parameter(
            torch.randn(1, self.num_heads, self.freq_bins, self.head_dim, 2) * 0.02
        )
        self.base_filter = nn.Parameter(torch.ones(self.num_heads, self.freq_bins, 1))
        self.base_bias = nn.Parameter(torch.full((self.num_heads, self.freq_bins, 1), -0.1))
        self.wavelet_weight = nn.Parameter(
            torch.randn(1, self.num_heads, 1, self.head_dim) * 0.02
        )

        if self.adaptive:
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.GELU(),
                nn.Linear(self.hidden_size, self.num_heads * self.freq_bins * 2),
            )

        if self.combine_mode == "concat":
            self.concat_proj = nn.Linear(2 * self.hidden_size, self.hidden_size)
        elif self.combine_mode == "gate":
            self.concat_proj = None
        else:
            raise ValueError("combine_mode must be 'gate' or 'concat'")

        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def _fft_branch(self, x_heads):
        batch_size = x_heads.size(0)
        freq_repr = torch.fft.rfft(x_heads, dim=2, norm="ortho")

        if self.adaptive:
            context = x_heads.mean(dim=2).reshape(batch_size, self.hidden_size)
            adapt_params = self.adaptive_mlp(context)
            adapt_params = adapt_params.view(batch_size, self.num_heads, self.freq_bins, 2)
            adaptive_scale = adapt_params[..., 0:1]
            adaptive_bias = adapt_params[..., 1:2]
        else:
            adaptive_scale = torch.zeros(
                batch_size,
                self.num_heads,
                self.freq_bins,
                1,
                device=x_heads.device,
                dtype=x_heads.dtype,
            )
            adaptive_bias = torch.zeros_like(adaptive_scale)

        effective_filter = self.base_filter.unsqueeze(0) * (1.0 + adaptive_scale)
        effective_bias = self.base_bias.unsqueeze(0) + adaptive_bias

        complex_filter = torch.view_as_complex(self.complex_weight)
        freq_repr = freq_repr * complex_filter
        freq_repr = freq_repr * effective_filter

        bias_real = effective_bias.expand(-1, -1, -1, self.head_dim)
        bias_complex = torch.complex(bias_real, torch.zeros_like(bias_real))
        freq_repr = freq_repr + bias_complex

        return torch.fft.irfft(freq_repr, n=self.seq_len, dim=2, norm="ortho")

    def _wavelet_branch(self, x_heads):
        batch_size, num_heads, seq_len, head_dim = x_heads.shape
        even_length = seq_len if seq_len % 2 == 0 else seq_len - 1
        x_trimmed = x_heads[:, :, :even_length, :]

        x_even = x_trimmed[:, :, 0::2, :]
        x_odd = x_trimmed[:, :, 1::2, :]

        approx = 0.5 * (x_even + x_odd)
        detail = 0.5 * (x_even - x_odd)
        detail = detail * (1.0 + self.wavelet_weight)

        x_even_recon = approx + detail
        x_odd_recon = approx - detail

        output = torch.zeros_like(x_trimmed)
        output[:, :, 0::2, :] = x_even_recon
        output[:, :, 1::2, :] = x_odd_recon

        if even_length < seq_len:
            pad = torch.zeros(
                batch_size,
                num_heads,
                1,
                head_dim,
                device=x_heads.device,
                dtype=x_heads.dtype,
            )
            output = torch.cat([output, pad], dim=2)

        return output

    def forward(self, input_tensor):
        batch_size, seq_len, hidden_size = input_tensor.shape
        x_heads = input_tensor.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        x_fft = self._fft_branch(x_heads)
        x_wavelet = self._wavelet_branch(x_heads)

        if self.combine_mode == "gate":
            x_combined = (1.0 - self.alpha) * x_wavelet + self.alpha * x_fft
        else:
            x_wavelet_flat = x_wavelet.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
            x_fft_flat = x_fft.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
            x_combined = self.concat_proj(torch.cat([x_wavelet_flat, x_fft_flat], dim=-1))
            x_combined = x_combined.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        x_out = x_combined.permute(0, 2, 1, 3).reshape(batch_size, seq_len, hidden_size)
        x_out = self.out_dropout(x_out)
        x_out = self.LayerNorm(x_out + input_tensor)
        return x_out


class WEARecFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_act = config["hidden_act"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]

        self.dense_1 = nn.Linear(self.hidden_size, self.inner_size)
        self.dense_2 = nn.Linear(self.inner_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.activation = nn.GELU() if self.hidden_act == "gelu" else nn.ReLU()

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.layer_norm(hidden_states + residual)


class WEARecBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = WEARecLayer(config)
        self.feed_forward = WEARecFeedForward(config)

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        return hidden_states


class WEARecEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        block = WEARecBlock(config)
        self.blocks = nn.ModuleList([copy.deepcopy(block) for _ in range(config["n_layers"])])

    def forward(self, hidden_states, output_all_encoded_layers=False):
        all_encoder_layers = [hidden_states]
        for layer_module in self.blocks:
            hidden_states = layer_module(hidden_states)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class WEARec(SequentialRecommender):
    def __init__(self, config, dataset):
        super().__init__(config, dataset)

        self.hidden_size = config["hidden_size"]
        self.n_layers = config["n_layers"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]

        self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.item_encoder = WEARecEncoder(config)

        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, item_seq, item_seq_len, return_all_layers=False):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        encoded_layers = self.item_encoder(input_emb, output_all_encoded_layers=True)
        if return_all_layers:
            return encoded_layers[-1]
        seq_output = self.gather_indexes(encoded_layers[-1], item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]

        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)
            return self.loss_fct(pos_score, neg_score)

        test_item_emb = self.item_embedding.weight
        logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
        return self.loss_fct(logits, pos_items)

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        return torch.sum(seq_output * test_item_emb, dim=1)

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding.weight
        return torch.matmul(seq_output, test_item_emb.transpose(0, 1))


WEARecModel = WEARec
