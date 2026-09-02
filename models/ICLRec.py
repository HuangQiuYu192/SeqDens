# -*- coding: utf-8 -*-
import faiss
import torch
from torch import nn

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
from recbole.model.layers import TransformerEncoder

class ICLRec(SequentialRecommender):

    def __init__(self, config, dataset):
        super(ICLRec, self).__init__(config, dataset)

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

        self.temperature = config["temperature"]
        self.device = config["device"]
        self.num_intent_cluster = config['num_intent_cluster']
        self.seq_representation_type = config['seq_representation_type']
        self.seq_representation_instancecl_type = config['seq_representation_instancecl_type']
        self.seed = config['seed']
        self.gpu_id = config['gpu_id']
        self.cf_weight = config['cf_weight']
        self.intent_cf_weight = config['intent_cf_weight']
        self.de_noise = config['de_noise']
        self.stage = 'warm_up'


        
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

        # initialize Kmeans
        if self.seq_representation_type == "mean":
            cluster = KMeans(
                num_cluster=self.num_intent_cluster,
                seed=self.seed,
                hidden_size=self.hidden_size, 
                gpu_id=self.gpu_id,
                device=0,
            )
            self.cluster = cluster
        else:
            cluster = KMeans(
                num_cluster=self.num_intent_cluster,
                seed=self.seed,
                hidden_size=self.hidden_size * self.max_seq_length,
                gpu_id=self.gpu_id,
                device=0,
            )
            self.cluster = cluster
        self.cf_criterion = NCELoss(self.temperature, self.device)
        self.pcl_criterion = PCLoss(self.temperature, self.device)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def forward(self, item_seq, item_seq_len):
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
        return trm_out[-1]  # [B, L, H]

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        sequence_output = self.forward(item_seq, item_seq_len)  # [B, L, H]
        seq_output = self.gather_indexes(sequence_output, item_seq_len - 1)  # [B, H]
        pos_items = interaction[self.POS_ITEM_ID]
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            rec_loss = self.loss_fct(pos_score, neg_score)
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))
            rec_loss = self.loss_fct(logits, pos_items)

        cl_losses = []
        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = (
            interaction['item_id_aug1_list'],
            interaction['item_id_aug1_length'],
            interaction["item_id_aug2_list"],
            interaction["item_id_aug2_length"],
        )
        cl_batch = (aug_item_seq1, aug_len1, aug_item_seq2, aug_len2)
        cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
            cl_batch, intent_ids=pos_items
        )
        cl_losses.append(self.cf_weight * cl_loss1)
        if self.stage != 'warm_up':
            if self.seq_representation_type == "mean":
                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
            sequence_output = sequence_output.detach().cpu().numpy()
            # query on clusters
            intent_id, seq2intent = self.cluster.query(sequence_output)
            seq2intent = seq2intent
            intent_id = intent_id
            cl_loss2 = self._pcl_one_pair_contrastive_learning(
                cl_batch, intents=[seq2intent], intent_ids=[intent_id]
            )
            cl_losses.append(self.intent_cf_weight * cl_loss2)

        joint_loss = rec_loss
        for cl_loss in cl_losses:
            joint_loss += cl_loss
        return joint_loss
        

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        seq_output = self.gather_indexes(seq_output, item_seq_len - 1)  # [B, H]
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(seq_output, test_items_emb.transpose(0, 1))  # [B n_items]
        return scores

    def _instance_cl_one_pair_contrastive_learning(self, batch, intent_ids=None):

        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = batch
        cl1_sequence_output = self.forward(aug_item_seq1, aug_len1)
        cl2_sequence_output = self.forward(aug_item_seq2, aug_len2)
        if self.seq_representation_instancecl_type == "mean":
            cl1_sequence_output = torch.mean(cl1_sequence_output, dim=1, keepdim=False)
            cl2_sequence_output = torch.mean(cl2_sequence_output, dim=1, keepdim=False)
        cl1_sequence_flatten = cl1_sequence_output.view(cl1_sequence_output.shape[0], -1)
        cl2_sequence_flatten = cl2_sequence_output.view(cl2_sequence_output.shape[0], -1)
        if self.de_noise:
            cl_loss = self.cf_criterion(cl1_sequence_flatten, cl2_sequence_flatten, intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion(cl1_sequence_flatten, cl2_sequence_flatten, intent_ids=None)
        return cl_loss
    
    def _pcl_one_pair_contrastive_learning(self, batch, intents, intent_ids):

        aug_item_seq1, aug_len1, aug_item_seq2, aug_len2 = batch
        cl1_sequence_output = self.forward(aug_item_seq1, aug_len1)
        cl2_sequence_output = self.forward(aug_item_seq2, aug_len2)
        if self.seq_representation_type == "mean":
            cl1_sequence_output = torch.mean(cl1_sequence_output, dim=1, keepdim=False)
            cl2_sequence_output = torch.mean(cl2_sequence_output, dim=1, keepdim=False)
        cl1_output_slice = cl1_sequence_output.view(cl1_sequence_output.shape[0], -1)
        cl2_output_slice = cl2_sequence_output.view(cl2_sequence_output.shape[0], -1)
        if self.de_noise:
            cl_loss = self.pcl_criterion(cl1_output_slice, cl2_output_slice, intents=intents, intent_ids=intent_ids)
        else:
            cl_loss = self.pcl_criterion(cl1_output_slice, cl2_output_slice, intents=intents, intent_ids=None)
        return cl_loss


class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if intent_ids is not None:
            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            sim22[mask == 1] = float("-inf")

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss
    

class PCLoss(nn.Module):
    """ Reference: https://github.com/salesforce/PCL/blob/018a929c53fcb93fd07041b1725185e1237d2c0e/pcl/builder.py#L168
    """

    def __init__(self, temperature, device, contrast_mode="all"):
        super(PCLoss, self).__init__()
        self.contrast_mode = contrast_mode
        self.criterion = NCELoss(temperature, device)

    def forward(self, batch_sample_one, batch_sample_two, intents, intent_ids):
        """
        features: 
        intents: num_clusters x batch_size x hidden_dims
        """
        # instance contrast with prototypes
        mean_pcl_loss = 0
        # do de-noise
        if intent_ids is not None:
            for intent, intent_id in zip(intents, intent_ids):
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_id)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_id)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        # don't do de-noise
        else:
            for intent in intents:
                pos_one_compare_loss = self.criterion(batch_sample_one, intent, intent_ids=None)
                pos_two_compare_loss = self.criterion(batch_sample_two, intent, intent_ids=None)
                mean_pcl_loss += pos_one_compare_loss
                mean_pcl_loss += pos_two_compare_loss
            mean_pcl_loss /= 2 * len(intents)
        return mean_pcl_loss


class KMeans(object):
    def __init__(self, num_cluster, seed, hidden_size, gpu_id=0, device="cpu"):
        """
        Args:
            k: number of clusters
        """
        self.seed = seed
        self.num_cluster = num_cluster
        self.max_points_per_centroid = 4096
        self.min_points_per_centroid = 0
        self.gpu_id = 0
        self.device = device
        self.first_batch = True
        self.hidden_size = hidden_size
        self.clus, self.index = self.__init_cluster(self.hidden_size)
        self.centroids = []

    def __init_cluster(
        self, hidden_size, verbose=False, niter=20, nredo=5, max_points_per_centroid=4096, min_points_per_centroid=0
    ):
        print(" cluster train iterations:", niter)
        clus = faiss.Clustering(hidden_size, self.num_cluster)
        clus.verbose = verbose
        clus.niter = niter
        clus.nredo = nredo
        clus.seed = self.seed
        clus.max_points_per_centroid = max_points_per_centroid
        clus.min_points_per_centroid = min_points_per_centroid

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = self.gpu_id
        index = faiss.GpuIndexFlatL2(res, hidden_size, cfg)
        return clus, index

    def train(self, x):
        # train to get centroids
        if x.shape[0] > self.num_cluster:
            self.clus.train(x, self.index)
        # get cluster centroids
        centroids = faiss.vector_to_array(self.clus.centroids).reshape(self.num_cluster, self.hidden_size)
        # convert to cuda Tensors for broadcast
        centroids = torch.Tensor(centroids).to(self.device)
        self.centroids = nn.functional.normalize(centroids, p=2, dim=1)

    def query(self, x):
        # self.index.add(x)
        D, I = self.index.search(x, 1)  # for each sample, find cluster distance and assignments
        seq2cluster = [int(n[0]) for n in I]
        # print("cluster number:", self.num_cluster,"cluster in batch:", len(set(seq2cluster)))
        seq2cluster = torch.LongTensor(seq2cluster).to(self.device)
        return seq2cluster, self.centroids[seq2cluster]