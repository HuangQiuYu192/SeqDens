# -*- coding: utf-8 -*-
import math

import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder
import torch.nn.functional as F

class IOCRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(IOCRec, self).__init__(config, dataset)

        # load parameters info
        self.n_layers = config["n_layers"]
        self.n_heads = config["n_heads"]
        self.hidden_size = config["hidden_size"]  # same as embedding_size
        self.inner_size = config[
            "inner_size"
        ]  # the dimensionality in feed-forward layer
        self.hidden_dropout_prob = config["hidden_dropout_prob"]
        self.attn_dropout_prob = config["attn_dropout_prob"]
        self.hidden_act = config["hidden_act"]
        self.layer_norm_eps = config["layer_norm_eps"]
        self.initializer_range = config["initializer_range"]
        self.loss_type = config["loss_type"]
        
        self.aug_views = 2
        self.lamda = config['lamda']
        self.k_intention = config['k_intention']
        self.all_hidden = config['all_hidden']

        self.global_seq_encoder = GlobalSeqEncoder(embed_size=self.hidden_size,
                                                   max_len=self.max_seq_length,
                                                   dropout=self.hidden_dropout_prob)
        self.disentangle_encoder = DisentangleEncoder(k_intention=self.k_intention,
                                                      embed_size=self.hidden_size,
                                                      max_len=self.max_seq_length)
        self.nce_loss = InfoNCELoss(temperature=config['tao'], similarity_type='dot')

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)
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
            
        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def local_seq_encoding(self, item_seq, item_seq_len, return_all=False):
        position_ids = torch.arange(
            item_seq.size(1), dtype=torch.long, device=item_seq.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)  # [B, L, H]
        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        mask = self.get_attention_mask(item_seq)

        trm_out = self.trm_encoder(
            input_emb, mask, output_all_encoded_layers=True
        )  # list([B, L, H])
        if return_all:
            return trm_out[-1]
        else:
            seq_output = self.gather_indexes(trm_out[-1], item_seq_len - 1)  # [B, H]
            return seq_output

    def global_seq_encoding(self, item_seq, item_seq_len):
        return self.global_seq_encoder(item_seq, item_seq_len, self.item_embedding)

    def forward(self, item_seq, item_seq_len):
        local_seq_emb = self.local_seq_encoding(item_seq, item_seq_len, return_all=True)  # [B, L, D]
        global_seq_emb = self.global_seq_encoding(item_seq, item_seq_len)  # [B, L, D]
        disentangled_intention_emb = self.disentangle_encoder(local_seq_emb, global_seq_emb, item_seq_len)  # [B, K, L, D]

        gather_index = item_seq_len.view(-1, 1, 1, 1).repeat(1, self.k_intention, 1, self.hidden_size)  # [B, K, 1, D]
        disentangled_intention_emb = disentangled_intention_emb.gather(2, gather_index - 1).squeeze()  # [B, K, D]
        candidates = self.item_embedding.weight.unsqueeze(0)  # [1, num_items, D]
        logits = disentangled_intention_emb @ candidates.permute(0, 2, 1)  # [B, K, num_items]
        max_logits, _ = torch.max(logits, 1)

        return max_logits


    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        max_logits = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = (
            interaction['item_id_aug1_list'],
            interaction['item_id_aug1_length'],
            interaction["item_id_aug2_list"],
            interaction["item_id_aug2_length"],
        )
        loss = self.loss_fct(max_logits, pos_items)
        B = pos_items.size(0)
        aug_local_emb_1 = self.local_seq_encoding(aug_item_seq1, aug_len1, return_all=self.all_hidden)
        aug_global_emb_1 = self.global_seq_encoding(aug_item_seq1, aug_len1)
        disentangled_intention_1 = self.disentangle_encoder(aug_local_emb_1, aug_global_emb_1, aug_len1)
        disentangled_intention_1 = disentangled_intention_1.view(B * self.k_intention, -1)  # [B * K, L * D]

        aug_local_emb_2 = self.local_seq_encoding(aug_item_seq2, aug_len2, return_all=self.all_hidden)
        aug_global_emb_2 = self.global_seq_encoding(aug_item_seq2, aug_len2)
        disentangled_intention_2 = self.disentangle_encoder(aug_local_emb_2, aug_global_emb_2, aug_len2)
        disentangled_intention_2 = disentangled_intention_2.view(B * self.k_intention, -1)  # [B * K, L * D]

        cl_loss = self.nce_loss(disentangled_intention_1, disentangled_intention_2)

        return loss + self.lamda * cl_loss
        

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        scores = self.forward(item_seq, item_seq_len)
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        scores = self.forward(item_seq, item_seq_len)
        return scores
    


class GlobalSeqEncoder(nn.Module):
    def __init__(self, embed_size, max_len, dropout=0.5):
        super(GlobalSeqEncoder, self).__init__()
        self.embed_size = embed_size
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        self.Q_s = nn.Parameter(torch.randn(max_len, embed_size))
        self.K_linear = nn.Linear(embed_size, embed_size)
        self.V_linear = nn.Linear(embed_size, embed_size)

    def forward(self, item_seq, seq_len, item_embeddings):
        """
        Args:
            item_seq (tensor): [B, L]
            seq_len (tensor): [B]
            item_embeddings (tensor): [num_items, D], item embedding table

        Returns:
            global_seq_emb: [B, L, D]
        """
        item_emb = item_embeddings(item_seq)  # [B, L, D]
        item_key = self.K_linear(item_emb)
        item_value = self.V_linear(item_emb)

        attn_logits = self.Q_s @ item_key.permute(0, 2, 1)  # [B, L, L]
        attn_score = F.softmax(attn_logits, -1)
        global_seq_emb = self.dropout(attn_score @ item_value)

        return global_seq_emb


class DisentangleEncoder(nn.Module):
    def __init__(self, k_intention, embed_size, max_len):
        super(DisentangleEncoder, self).__init__()
        self.embed_size = embed_size

        self.intentions = nn.Parameter(torch.randn(k_intention, embed_size))
        self.pos_fai = nn.Embedding(max_len, embed_size)
        self.rou = nn.Parameter(torch.randn(embed_size, ))
        self.W = nn.Linear(embed_size, embed_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.layer_norm_3 = nn.LayerNorm(embed_size)
        self.layer_norm_4 = nn.LayerNorm(embed_size)
        self.layer_norm_5 = nn.LayerNorm(embed_size)

    def forward(self, local_item_emb, global_item_emb, seq_len):
        """
        Args:
            local_item_emb: [B, L, D]
            global_item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            disentangled_intention_emb: [B, K, L, D]
        """
        local_disen_emb = self.intention_disentangling(local_item_emb, seq_len)
        global_siden_emb = self.intention_disentangling(global_item_emb, seq_len)
        disentangled_intention_emb = local_disen_emb + global_siden_emb

        return disentangled_intention_emb

    def item2IntentionScore(self, item_emb):
        """
        Args:
            item_emb: [B, L, D]
        Returns:
            score: [B, L, K]
        """
        item_emb_norm = self.layer_norm_1(item_emb)  # [B, L, D]
        intention_norm = self.layer_norm_2(self.intentions).unsqueeze(0)  # [1, K, D]

        logits = item_emb_norm @ intention_norm.permute(0, 2, 1)  # [B, L, K]
        score = F.softmax(logits / math.sqrt(self.embed_size), -1)

        return score

    def item2AttnWeight(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B, L, D]
            seq_len: [B]
        Returns:
            score: [B, L]
        """
        B, L = item_emb.size(0), item_emb.size(1)
        dev = item_emb.device
        safe_seq_len = seq_len.clamp(min=1, max=L)
        item_query_row = item_emb[torch.arange(B, device=dev), safe_seq_len - 1]  # [B, D]
        item_query_row += self.pos_fai(safe_seq_len - 1) + self.rou
        item_query = self.layer_norm_3(item_query_row).unsqueeze(1)  # [B, 1, D]

        pos_fai_tensor = self.pos_fai(torch.arange(L).to(dev)).unsqueeze(0)  # [1, L, D]
        item_key_hat = self.layer_norm_4(item_emb + pos_fai_tensor)
        item_key = item_key_hat + torch.relu(self.W(item_key_hat))

        logits = item_query @ item_key.permute(0, 2, 1)  # [B, 1, L]
        logits = logits.squeeze() / math.sqrt(self.embed_size)
        score = F.softmax(logits, -1)

        return score

    def intention_disentangling(self, item_emb, seq_len):
        """
        Args:
            item_emb: [B. L, D]
            seq_len: [B]
        Returns:
            item_disentangled_emb: [B, K, L, D]
        """
        # get score
        item2intention_score = self.item2IntentionScore(item_emb)
        item_attn_weight = self.item2AttnWeight(item_emb, seq_len)

        # get disentangled embedding
        score_fuse = item2intention_score * item_attn_weight.unsqueeze(-1)  # [B, L, K]
        score_fuse = score_fuse.permute(0, 2, 1).unsqueeze(-1)  # [B, K, L, 1]
        item_emb_k = item_emb.unsqueeze(1)  # [B, 1, L, D]
        disentangled_item_emb = self.layer_norm_5(score_fuse * item_emb_k)
        return disentangled_item_emb
    

class InfoNCELoss(nn.Module):
    """
    Pair-wise Noise Contrastive Estimation Loss
    """

    def __init__(self, temperature, similarity_type):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature  # temperature
        self.sim_type = similarity_type  # cos or dot
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, aug_hidden_view1, aug_hidden_view2, mask=None):
        """
        Args:
            aug_hidden_view1 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1
            aug_hidden_view2 (FloatTensor, [batch, max_len, dim] or [batch, dim]): augmented sequence representation1

        Returns: nce_loss (FloatTensor, (,)): calculated nce loss
        """
        if aug_hidden_view1.ndim > 2:
            # flatten tensor
            aug_hidden_view1 = aug_hidden_view1.view(aug_hidden_view1.size(0), -1)
            aug_hidden_view2 = aug_hidden_view2.view(aug_hidden_view2.size(0), -1)

        if self.sim_type not in ['cos', 'dot']:
            raise Exception(f"Invalid similarity_type for cs loss: [current:{self.sim_type}]. "
                            f"Please choose from ['cos', 'dot']")

        if self.sim_type == 'cos':
            sim11 = self.cosinesim(aug_hidden_view1, aug_hidden_view1)
            sim22 = self.cosinesim(aug_hidden_view2, aug_hidden_view2)
            sim12 = self.cosinesim(aug_hidden_view1, aug_hidden_view2)
        elif self.sim_type == 'dot':
            # calc similarity
            sim11 = aug_hidden_view1 @ aug_hidden_view1.t()
            sim22 = aug_hidden_view2 @ aug_hidden_view2.t()
            sim12 = aug_hidden_view1 @ aug_hidden_view2.t()
        # mask non-calc value
        sim11[..., range(sim11.size(0)), range(sim11.size(0))] = float('-inf')
        sim22[..., range(sim22.size(0)), range(sim22.size(0))] = float('-inf')

        cl_logits1 = torch.cat([sim12, sim11], -1)
        cl_logits2 = torch.cat([sim22, sim12.t()], -1)
        cl_logits = torch.cat([cl_logits1, cl_logits2], 0) / self.temperature
        if mask is not None:
            cl_logits = torch.masked_fill(cl_logits, mask, float('-inf'))
        target = torch.arange(cl_logits.size(0)).long().to(aug_hidden_view1.device)
        cl_loss = self.criterion(cl_logits, target)

        return cl_loss

    def cosinesim(self, aug_hidden1, aug_hidden2):
        h = torch.matmul(aug_hidden1, aug_hidden2.T)
        h1_norm2 = aug_hidden1.pow(2).sum(dim=-1).sqrt().view(h.shape[0], 1)
        h2_norm2 = aug_hidden2.pow(2).sum(dim=-1).sqrt().view(1, h.shape[0])
        return h / (h1_norm2 @ h2_norm2)