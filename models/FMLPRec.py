import torch
import torch.nn as nn
import copy

from recbole.model.abstract_recommender import SequentialRecommender


class FMLPRecLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config["hidden_size"]
        self.seq_len = config["MAX_ITEM_LIST_LENGTH"]
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.layer_norm_eps = config["layer_norm_eps"]
        # frequency bins
        self.freq_bins = self.seq_len // 2 + 1

        self.complex_weight = nn.Parameter(
            torch.randn(1, self.freq_bins, self.hidden_size, 2) * 0.02
        )
        self.out_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)

    def forward(self, input_tensor):
        # input_tensor: [B, L, H]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm="ortho")  # [B, F, H]

        weight = torch.view_as_complex(self.complex_weight)  # [1, F, H]
        x = x * weight
        sequence_emb_fft = torch.fft.irfft(x, n=seq_len, dim=1, norm="ortho")

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = hidden_states + input_tensor
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class FMLPRecBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = FMLPRecLayer(config)
        self.ffn = nn.Sequential(
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.GELU(),
            nn.Dropout(config["hidden_dropout_prob"]),
            nn.Linear(config["hidden_size"], config["hidden_size"]),
            nn.Dropout(config["hidden_dropout_prob"]),
            nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"]),
        )

    def forward(self, hidden_states):
        hidden_states = self.layer(hidden_states)
        hidden_states = self.ffn(hidden_states) + hidden_states
        return hidden_states


class FMLPRecEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        block = FMLPRecBlock(config)
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


class FMLPRec(SequentialRecommender):
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

        self.item_encoder = FMLPRecEncoder(config)

        if self.loss_type == "BPR":
            from recbole.model.loss import BPRLoss

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

    def forward(self, item_seq, item_seq_len):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        item_encoded_layers = self.item_encoder(input_emb, output_all_encoded_layers=True)
        seq_output = self.gather_indexes(item_encoded_layers[-1], item_seq_len - 1)
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


FMLPRecModel = FMLPRec
