# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import os
import random

import numpy as np
import torch
from tqdm import tqdm

from recbole.data.dataset import KGSeqDataset, SequentialDataset
from recbole.data.dataset.dataset import Dataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.utils.enum_type import FeatureType, FeatureSource

from .data_augmentation import AUGMENTATIONS


class GRU4RecKGDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class KSRDataset(KGSeqDataset):
    def __init__(self, config):
        super().__init__(config)


class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """

    def __init__(self, config):
        super().__init__(config)

        list_suffix = config["LIST_SUFFIX"]
        neg_prefix = config["NEG_PREFIX"]
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(
            self.inter_feat[self.iid_field]
        )

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug("data_augmentation")

        self._aug_presets()

        self._check_field("uid_field", "time_field")
        max_item_list_len = self.config["MAX_ITEM_LIST_LENGTH"]
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f"{field}_list_field")
                list_len = self.field2seqlen[list_field]
                shape = (
                    (new_length, list_len)
                    if isinstance(list_len, int)
                    else (new_length,) + list_len
                )
                if (
                    self.field2type[field] in [FeatureType.FLOAT, FeatureType.FLOAT_SEQ]
                    and field in self.config["numerical_features"]
                ):
                    shape += (2,)
                list_ftype = self.field2type[list_field]
                dtype = (
                    torch.int64
                    if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]
                    else torch.float64
                )
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(
                    zip(item_list_index, item_list_length)
                ):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(
                        zip(item_list_index, item_list_length)
                    ):
                        new_dict[self.neg_item_list_field][i][:length] = (
                            self.neg_item_list[index]
                        )

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data


class PairRandomAugDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)
    def __init__(self, config):
        aug_types = config["aug_types"]
        self.aug_types = []
        assert isinstance(aug_types, list) and len(aug_types) > 0, "aug_types should be a non-empty list."
        
        self.crop_ratio = config["crop_ratio"]
        self.mask_ratio = config["mask_ratio"]
        self.reorder_ratio = config["reorder_ratio"]
        
        for aug_type in aug_types:
            assert aug_type in AUGMENTATIONS, f"augmentation type '{aug_type}' is not supported."
            self.aug_types.append(AUGMENTATIONS[aug_type](getattr(self, f"{aug_type}_ratio")))

        self.aug1_item_list_field = None
        self.aug1_item_list_length_field = None
        self.aug2_item_list_field = None
        self.aug2_item_list_length_field = None

        super(PairRandomAugDataset, self).__init__(config)

    def _aug_presets(self):
        super()._aug_presets()

        self.aug1_item_list_field = (
            self.iid_field + "_aug1" + self.config["LIST_SUFFIX"]
        )
        self.aug1_item_list_length_field = (
            self.iid_field + "_aug1_length"
        )
        self.aug2_item_list_field = (
            self.iid_field + "_aug2" + self.config["LIST_SUFFIX"]
        )
        self.aug2_item_list_length_field = (
            self.iid_field + "_aug2_length"
        )

        self.set_field_property(
            self.aug1_item_list_field,
            FeatureType.TOKEN_SEQ,
            FeatureSource.INTERACTION,
            self.max_item_list_len,
        )
        self.set_field_property(
            self.aug1_item_list_length_field,
            FeatureType.TOKEN,
            FeatureSource.INTERACTION,
            1
        )
        self.set_field_property(
            self.aug2_item_list_field,
            FeatureType.TOKEN_SEQ,
            FeatureSource.INTERACTION,
            self.max_item_list_len,
        )
        self.set_field_property(
            self.aug2_item_list_length_field,
            FeatureType.TOKEN,
            FeatureSource.INTERACTION,
            1
        )

    def build(self):

        datasets = super().build()

        if len(datasets) != 3:
            return datasets
        
        train_dataset, valid_dataset, test_dataset = datasets

        print("Applying augmentation to training dataset...")
        self._pair_random_aug(train_dataset)

        return [train_dataset, valid_dataset, test_dataset]

    def _pair_random_aug(self, train_dataset):

        inter_feat = train_dataset.inter_feat

        item_lists = inter_feat[self.item_id_list_field]
        lengths = inter_feat[self.item_list_length_field]

        aug1_lists = torch.zeros_like(item_lists)
        aug2_lists = torch.zeros_like(item_lists)
        aug1_lengths = torch.zeros_like(lengths)
        aug2_lengths = torch.zeros_like(lengths)
        
        for i in range(len(item_lists)):
            seq = item_lists[i]
            seq_len = int(lengths[i].item())
            
            if seq_len == 0:
                continue

            valid_seq = seq[:seq_len].cpu().numpy().tolist()
            
            for aug_lists, aug_lengths, aug_type in [
                (aug1_lists, aug1_lengths, random.choice(self.aug_types)),
                (aug2_lists, aug2_lengths, random.choice(self.aug_types))
            ]:
                aug_seq = aug_type(valid_seq)
                aug_len = len(aug_seq)
                
                aug_seq_tensor = torch.tensor(aug_seq, dtype=seq.dtype, device=seq.device)
                
                if aug_len > self.max_item_list_len:
                    aug_seq_tensor = aug_seq_tensor[:self.max_item_list_len]
                    aug_len = self.max_item_list_len
                
                if aug_len < self.max_item_list_len:
                    padding = torch.zeros(self.max_item_list_len - aug_len, dtype=seq.dtype, device=seq.device)
                    aug_seq_tensor = torch.cat([aug_seq_tensor, padding])
                
                aug_lists[i] = aug_seq_tensor
                aug_lengths[i] = aug_len
        
        inter_feat[self.aug1_item_list_field] = aug1_lists
        inter_feat[self.aug1_item_list_length_field] = aug1_lengths
        inter_feat[self.aug2_item_list_field] = aug2_lists
        inter_feat[self.aug2_item_list_length_field] = aug2_lengths
        
        print(f"Training dataset augmented with {len(self.aug_types)} augmentation types.")

    def data_augmentation(self):
        super().data_augmentation()


class SameTargetAugDataset(SequentialDataset):
    def __init__(self, config):
        self.aug_item_list_field = None
        self.aug_item_list_length_field = None

        self.same_target_indices = None
        self.same_target_path = config['same_target_path']

        super(SameTargetAugDataset, self).__init__(config)

    def _aug_presets(self):
        super()._aug_presets()

        self.aug_item_list_field = (
            self.iid_field + "_aug" + self.config["LIST_SUFFIX"]
        )
        self.aug_item_list_length_field = (
            self.iid_field + "_aug_length"
        )

        self.set_field_property(
            self.aug_item_list_field,
            FeatureType.TOKEN_SEQ,
            FeatureSource.INTERACTION,
            self.max_item_list_len,
        )
        self.set_field_property(
            self.aug_item_list_length_field,
            FeatureType.TOKEN,
            FeatureSource.INTERACTION,
            1
        )

    def build(self):

        datasets = super().build()

        if len(datasets) != 3:
            return datasets
        
        train_dataset, valid_dataset, test_dataset = datasets

        print("Applying same-target augmentation to training dataset...")
        self._same_target_aug(train_dataset)

        return [train_dataset, valid_dataset, test_dataset]

    def _same_target_aug(self, train_dataset):
        self._build_or_load_same_target_indices(train_dataset)

        inter_feat = train_dataset.inter_feat
        item_lists = inter_feat[self.item_id_list_field]
        lengths = inter_feat[self.item_list_length_field]
        target_items = inter_feat[self.iid_field].cpu().numpy()

        aug_lists = torch.zeros_like(item_lists)
        aug_lengths = torch.zeros_like(lengths)

        for i in range(len(item_lists)):
            seq = item_lists[i]
            seq_len = int(lengths[i].item())
            target_item = int(target_items[i])

            if seq_len == 0 or target_item == 0:
                continue

            if target_item >= len(self.same_target_indices):
                aug_seq = seq[:seq_len].clone()
                aug_len = seq_len
            else:
                candidate_seqs = self.same_target_indices[target_item]

                if len(candidate_seqs) == 0:
                    aug_seq = seq[:seq_len].clone()
                    aug_len = seq_len
                else:
                    current_seq = seq[:seq_len].cpu().numpy().tolist()
                    other_seqs = []
                    for s in candidate_seqs:
                        if isinstance(s, np.ndarray):
                            s = s.tolist()
                        if s != current_seq:
                            other_seqs.append(s)

                    if other_seqs:
                        selected_seq = random.choice(other_seqs)
                    else:
                        selected_seq = current_seq

                    if isinstance(selected_seq, np.ndarray):
                        aug_seq = torch.from_numpy(selected_seq).to(seq.device)
                    else:
                        aug_seq = torch.tensor(selected_seq, dtype=seq.dtype, device=seq.device)

                    aug_len = len(selected_seq)

            if aug_len > self.max_item_list_len:
                aug_seq = aug_seq[:self.max_item_list_len]
                aug_len = self.max_item_list_len

            if aug_len < self.max_item_list_len:
                padding = torch.zeros(
                    self.max_item_list_len - aug_len,
                    dtype=seq.dtype,
                    device=seq.device
                )
                aug_seq = torch.cat([aug_seq, padding])

            aug_lists[i] = aug_seq
            aug_lengths[i] = aug_len

        inter_feat[self.aug_item_list_field] = aug_lists
        inter_feat[self.aug_item_list_length_field] = aug_lengths

        print(f"Training dataset augmented with same-target sequences.")

    def _build_or_load_same_target_indices(self, train_dataset):
        if self.same_target_path and os.path.exists(self.same_target_path):
            print(f"Loading same-target indices from: {self.same_target_path}")
            self.same_target_indices = np.load(
                self.same_target_path, allow_pickle=True
            )
            return

        print("Building same-target indices from training data...")
        self.same_target_indices = self._get_same_target_index(train_dataset)

        if self.same_target_path:
            os.makedirs(os.path.dirname(self.same_target_path), exist_ok=True)
            np.save(
                self.same_target_path,
                np.array(self.same_target_indices, dtype=object),
            )

    def _get_same_target_index(self, train_dataset):
        inter_feat = train_dataset.inter_feat
        item_lists = inter_feat[self.item_id_list_field].cpu().numpy()
        target_items = inter_feat[self.iid_field].cpu().numpy()

        max_item = 0
        for seq in item_lists:
            if seq.size > 0:
                max_item = max(max_item, seq.max())
        for t in target_items:
            if t > 0:
                max_item = max(max_item, int(t))

        same_target_index = [[] for _ in range(max_item + 1)]

        for seq, target_item in zip(item_lists, target_items):
            if seq.size == 0 or target_item == 0:
                continue
            same_target_index[int(target_item)].append(seq.copy())

        return same_target_index

    def data_augmentation(self):

        super().data_augmentation()


class DraftAndSameTargetDataset(Dataset):
    def __init__(self, config):
        self.max_item_list_len = config['MAX_ITEM_LIST_LENGTH']
        self.item_list_length_field = config['ITEM_LIST_LENGTH_FIELD']
        super().__init__(config)
        if config['benchmark_filename'] is not None:
            self._benchmark_presets()

    def _change_feat_format(self):
        """Change feat format from :class:`pandas.DataFrame` to :class:`Interaction`,
           then perform data augmentation.
        """
        super()._change_feat_format()

        if self.config['benchmark_filename'] is not None:
            return
        self.logger.debug('Augmentation for sequential recommendation.')
        self.data_augmentation()

    def _aug_presets(self):
        list_suffix = self.config['LIST_SUFFIX']
        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
                ftype = self.field2type[field]

                if ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ]:
                    list_ftype = FeatureType.TOKEN_SEQ
                else:
                    list_ftype = FeatureType.FLOAT_SEQ

                if ftype in [FeatureType.TOKEN_SEQ, FeatureType.FLOAT_SEQ]:
                    list_len = (self.max_item_list_len, self.field2seqlen[field])
                else:
                    list_len = self.max_item_list_len

                self.set_field_property(list_field, list_ftype, FeatureSource.INTERACTION, list_len)

        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug('data_augmentation')

        self._aug_presets()

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

        # interest drift quantization, calculate the IDM, transforms into IDQ
        # =====
        old_repr = 1
        newc_degrees = []
        target_items = self.inter_feat[target_index][self.iid_field]
        item_seqs = new_dict[getattr(self, self.iid_field + '_list_field')]
        item_cates = self.get_item_feature()['categories']
        item_seq_lens = new_dict[self.item_list_length_field]
        for i, uid in enumerate(uid_list):
            target_item = target_items[i]
            tar_item_cate = item_cates[target_item]
            seq_len = item_seq_lens[i]
            item_seq = item_seqs[i][:seq_len]

            if old_repr == 1 and target_item in list(item_seq.numpy()):
                newc_degrees.append(0)
                continue

            item_seq_cates_set = set((item_cates[item_seq]).numpy().reshape(-1))
            # 0 for removing the padding token, 1 for removing "Beauty" or "Sports", such a large category (dataset name)
            item_seq_cates_set = item_seq_cates_set - {0, 1}

            tar_item_cate_set = set(tar_item_cate.numpy().reshape(-1))
            tar_item_cate_set = tar_item_cate_set - {0, 1}
            tar_item_cate_num = len(tar_item_cate_set)

            same_cate_num = len(item_seq_cates_set & tar_item_cate_set)
            same_ratio = same_cate_num / tar_item_cate_num if tar_item_cate_num != 0 else 0

            if old_repr == 1:
                if self.config['n_newc_repr'] == 2:
                    if same_ratio == 0:
                        newc_degrees.append(2)
                    else:
                        newc_degrees.append(1)
                elif self.config['n_newc_repr'] == 3:
                    if same_ratio == 0:
                        newc_degrees.append(3)
                    elif same_ratio == 1:
                        newc_degrees.append(1)
                    else:
                        newc_degrees.append(2)
                elif self.config['n_newc_repr'] == 4:
                    if same_ratio == 0:
                        newc_degrees.append(4)
                    elif same_ratio < 0.5:
                        newc_degrees.append(3)
                    elif same_ratio < 1:
                        newc_degrees.append(2)
                    else:
                        newc_degrees.append(1)
                elif self.config['n_newc_repr'] == 5:
                    if same_ratio == 0:
                        newc_degrees.append(5)
                    elif same_ratio < 0.33:
                        newc_degrees.append(4)
                    elif same_ratio < 0.66:
                        newc_degrees.append(3)
                    elif same_ratio < 1:
                        newc_degrees.append(2)
                    else:
                        newc_degrees.append(1)

        new_dict['newc_degree'] = torch.tensor(newc_degrees)
        new_data.update(Interaction(new_dict))
        # =====

        # IDRA sampling
        if self.config['idra'] == 1:
            same_target_i_c_index = self.semantic_augmentation(new_data)
            null_index = []
            sample_pos = []
            for i, targets in enumerate(same_target_i_c_index):
                if len(targets) == 0:
                    sample_pos.append(-1)
                    null_index.append(i)
                else:
                    sample_pos.append(np.random.choice(targets))

            sem_pos_seqs = new_data[getattr(self, self.iid_field + '_list_field')][sample_pos]
            sem_pos_lengths = new_data['item_length'][sample_pos]

            sem_aug_user_ids = new_data['user_id'][sample_pos]

            if null_index:
                sem_pos_seqs[null_index] = new_data['item_id_list'][null_index]

                sem_pos_lengths[null_index] = new_data['item_length'][null_index]

                sem_aug_user_ids[null_index] = new_data['user_id'][null_index]

            new_data.update(Interaction({'sem_aug': sem_pos_seqs, 'sem_aug_lengths': sem_pos_lengths,
                                         'sem_aug_user_ids': sem_aug_user_ids}))


        self.inter_feat = new_data

    def semantic_augmentation(self, aug_seqs):
        aug_path = self.config['data_path'] + '/semantic_augmentation.npy'
        import os
        if os.path.exists(aug_path):
            same_target_index = np.load(aug_path, allow_pickle=True)
        else:
            same_target_index = []
            target_item = aug_seqs['item_id'].numpy()

            target_newc_degree = aug_seqs['newc_degree'].numpy()

            for index, (item_id, newc_degree) in tqdm(
                enumerate(zip(target_item, target_newc_degree)),
                total=len(target_item),
                desc='semantic_augmentation',
                leave=False,
            ):
                all_index_same_id = np.where(target_item == item_id)[0]  # all index of a specific item id with self item
                delete_index = np.argwhere(all_index_same_id == index)
                all_index_same_id_wo_self = np.delete(all_index_same_id, delete_index)

                all_index_same_newcdegree = np.where(target_newc_degree == newc_degree)[0]
                delete_index = np.argwhere(all_index_same_newcdegree == index)
                all_index_same_newcdegree_wo_self = np.delete(all_index_same_newcdegree, delete_index)

                all_index_same_id_wo_self = np.array(list(set(all_index_same_id_wo_self) & set(all_index_same_newcdegree_wo_self)))
                same_target_index.append(all_index_same_id_wo_self)

            same_target_index = np.array(same_target_index, dtype=object)
            np.save(aug_path, same_target_index)

        return same_target_index

    def _benchmark_presets(self):
        list_suffix = self.config['LIST_SUFFIX']
        for field in self.inter_feat:
            if field + list_suffix in self.inter_feat:
                list_field = field + list_suffix
                setattr(self, f'{field}_list_field', list_field)
        self.set_field_property(self.item_list_length_field, FeatureType.TOKEN, FeatureSource.INTERACTION, 1)
        self.inter_feat[self.item_list_length_field] = self.inter_feat[self.item_id_list_field].agg(len)

    def inter_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe interactions between user_id and item_id.
        Sparse matrix has shape (user_num, item_num).
        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if not self.uid_field or not self.iid_field:
            raise ValueError('dataset does not exist uid/iid, thus can not converted to sparse matrix.')

        l1_idx = (self.inter_feat[self.item_list_length_field] == 1)
        l1_inter_dict = self.inter_feat[l1_idx].interaction
        new_dict = {}
        list_suffix = self.config['LIST_SUFFIX']
        candidate_field_set = set()
        for field in l1_inter_dict:
            if field != self.uid_field and field + list_suffix in l1_inter_dict:
                candidate_field_set.add(field)
                new_dict[field] = torch.cat([self.inter_feat[field], l1_inter_dict[field + list_suffix][:, 0]])
            elif (not field.endswith(list_suffix)) and (field != self.item_list_length_field):
                new_dict[field] = torch.cat([self.inter_feat[field], l1_inter_dict[field]])
        local_inter_feat = Interaction(new_dict)
        return self._create_sparse_matrix(local_inter_feat, self.uid_field, self.iid_field, form, value_field)

    def build(self):
        """Processing dataset according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole.config.eval_setting.EvalSetting` for details.

        Args:
            eval_setting (:class:`~recbole.config.eval_setting.EvalSetting`):
                Object contains evaluation settings, which guide the data processing procedure.

        Returns:
            list: List of built :class:`Dataset`.
        """
        ordering_args = self.config['eval_args']['order']
        if ordering_args != 'TO':
            raise ValueError(f'The ordering args for sequential recommendation has to be \'TO\'')

        return super().build()


class CL4SRecDataset(PairRandomAugDataset):
    def __init__(self, config):
        super(CL4SRecDataset, self).__init__(config)


class ICLRecDataset(PairRandomAugDataset):
    def __init__(self, config):
        super(ICLRecDataset, self).__init__(config)


class IOCRecDataset(PairRandomAugDataset):
    def __init__(self, config):
        super(IOCRecDataset, self).__init__(config)


class DuoRecDataset(SameTargetAugDataset):
    def __init__(self, config):
        super(DuoRecDataset, self).__init__(config)


class ICSRecDataset(SameTargetAugDataset):
    def __init__(self, config):
        super(ICSRecDataset, self).__init__(config)


class IDURLDataset(DraftAndSameTargetDataset):
    def __init__(self, config):
        super(IDURLDataset, self).__init__(config)

