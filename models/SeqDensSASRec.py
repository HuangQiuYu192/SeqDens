# -*- coding: utf-8 -*-
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.layers import TransformerEncoder
from recbole.model.loss import BPRLoss


class BehaviorCompressionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        n_heads,
        compression_length,
        dropout_prob,
        layer_norm_eps,
    ):
        super().__init__()
        self.compression_length = compression_length
        self.latent_queries = nn.Parameter(torch.empty(compression_length, hidden_size))
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=n_heads,
            dropout=dropout_prob,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, history_emb, history_mask):
        batch_size = history_emb.size(0)
        if history_emb.size(1) == 0:
            return history_emb.new_zeros(batch_size, self.compression_length, history_emb.size(-1))

        has_history = history_mask.any(dim=1)
        safe_key_padding_mask = ~history_mask
        safe_key_padding_mask = safe_key_padding_mask.clone()
        safe_key_padding_mask[~has_history, 0] = False

        safe_history_emb = history_emb.clone()
        safe_history_emb[~history_mask] = 0.0

        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        compressed, _ = self.cross_attention(
            query=queries,
            key=safe_history_emb,
            value=safe_history_emb,
            key_padding_mask=safe_key_padding_mask,
            need_weights=False,
        )
        compressed = self.LayerNorm(queries + self.dropout(compressed))
        compressed = compressed * has_history.view(batch_size, 1, 1).to(compressed.dtype)
        return compressed


class SeqDensSASRec(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SeqDensSASRec, self).__init__(config, dataset)

        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]
        self.inner_size = config["inner_size"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        self.compression_length = config["compression_length"]
        self.recent_length = config["recent_length"]
        self.compact_seq_length = self.compression_length + self.recent_length

        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            self.compact_seq_length, self.hidden_size
        )
        self.compression_encoder = BehaviorCompressionEncoder(
            hidden_size=self.hidden_size,
            n_heads=self.n_heads,
            compression_length=self.compression_length,
            dropout_prob=self.attn_dropout_prob,
            layer_norm_eps=self.layer_norm_eps,
        )
        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps,
        )

        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

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
        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=self.initializer_range)
            module.out_proj.weight.data.normal_(mean=0.0, std=self.initializer_range)
            if module.in_proj_bias is not None:
                module.in_proj_bias.data.zero_()
            if module.out_proj.bias is not None:
                module.out_proj.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        if isinstance(module, BehaviorCompressionEncoder):
            module.latent_queries.data.normal_(mean=0.0, std=self.initializer_range)

    def _build_compact_sequence(self, item_seq, item_seq_len):
        recent_len = min(self.recent_length, item_seq.size(1))
        old_seq = item_seq[:, :-recent_len] if recent_len > 0 else item_seq
        recent_seq = item_seq[:, -recent_len:] if recent_len > 0 else item_seq[:, :0]

        old_emb = self.item_embedding(old_seq)
        old_mask = old_seq != 0
        compressed_emb = self.compression_encoder(old_emb, old_mask)

        recent_emb = self.item_embedding(recent_seq)
        input_emb = torch.cat([compressed_emb, recent_emb], dim=1)

        compact_mask = torch.cat(
            [
                old_mask.any(dim=1, keepdim=True).expand(-1, self.compression_length),
                recent_seq != 0,
            ],
            dim=1,
        )

        position_ids = torch.arange(
            input_emb.size(1), dtype=torch.long, device=item_seq.device
        )
        position_embedding = self.position_embedding(position_ids).unsqueeze(0)
        input_emb = self.LayerNorm(input_emb + position_embedding)
        input_emb = self.dropout(input_emb)

        old_token_count = (
            old_mask.any(dim=1).long() * self.compression_length
        )
        recent_token_count = torch.minimum(
            item_seq_len, item_seq_len.new_full(item_seq_len.shape, recent_len)
        )
        compact_seq_len = old_token_count + recent_token_count
        return input_emb, compact_mask, compact_seq_len

    def forward(self, item_seq, item_seq_len, return_all_layers=False):
        input_emb, compact_mask, compact_seq_len = self._build_compact_sequence(
            item_seq, item_seq_len
        )
        mask = self.get_attention_mask(compact_mask.long())
        trm_out = self.trm_encoder(input_emb, mask, output_all_encoded_layers=True)
        if return_all_layers:
            return trm_out[-1]
        return self.gather_indexes(trm_out[-1], compact_seq_len - 1)

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
            loss = self.loss_fct(pos_score, neg_score)
        else:
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            loss = self.loss_fct(logits, pos_items)
        return loss

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))
        return scores
