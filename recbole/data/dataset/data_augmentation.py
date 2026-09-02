import copy
import math
import random
import math

class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self, crop_ratio=0.2):
        self.tao = crop_ratio

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        sub_seq_length = int(self.tao * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(0, len(copied_sequence) - sub_seq_length)
        if sub_seq_length < 1:
            return [copied_sequence[min(start_index, len(sequence) - 1)]]
        else:
            cropped_seq = copied_sequence[start_index:start_index + sub_seq_length]
            return cropped_seq


class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self, mask_ratio=0.7, mask_id=0):
        self.gamma = mask_ratio
        self.mask_id = mask_id

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        mask_nums = int(self.gamma * len(copied_sequence))
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx in mask_idx:
            copied_sequence[idx] = self.mask_id
        return copied_sequence


class Reorder(object):
    """Randomly shuffle a continuous sub-sequence"""

    def __init__(self, reorder_ratio=0.2):
        self.beta = reorder_ratio

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)

        sub_seq_len = int(self.beta * len(copied_sequence))
        start_index = random.randint(0, len(copied_sequence) - sub_seq_len)
        sub_seq = copied_sequence[start_index:start_index + sub_seq_len]
        random.shuffle(sub_seq)
        reordered_seq = copied_sequence[:start_index] + sub_seq + \
                        copied_sequence[start_index + sub_seq_len:]
        assert len(copied_sequence) == len(reordered_seq)
        return reordered_seq


class Repeat(object):
    """Randomly repeat p% of items in sequence"""

    def __init__(self, p=0.2, min_rep_size=1):
        self.p = p  # max repeat ratio
        self.min_rep_size = min_rep_size

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        max_repeat_nums = math.ceil(self.p * len(copied_sequence))
        repeat_nums = \
            random.sample([i for i in range(self.min_rep_size, max(self.min_rep_size, max_repeat_nums) + 1)], k=1)[0]
        repeat_idx = random.sample([i for i in range(len(copied_sequence))], k=repeat_nums)
        repeat_idx.sort()
        new_seq = []
        cur_idx = 0
        for i, item in enumerate(copied_sequence):
            new_seq.append(item)
            if cur_idx < len(repeat_idx) and i == repeat_idx[cur_idx]:
                new_seq.append(item)
                cur_idx += 1
        return new_seq


class Drop(object):
    """Randomly repeat p% of items in sequence"""

    def __init__(self, p=0.2):
        self.p = p  # max repeat ratio

    def __call__(self, sequence):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        drop_num = math.floor(self.p * len(copied_sequence))
        drop_idx = random.sample([i for i in range(len(copied_sequence))], k=drop_num)
        drop_idx.sort()
        new_seq = []
        cur_idx = 0
        for i, item in enumerate(copied_sequence):
            if cur_idx < len(drop_idx) and i == drop_idx[cur_idx]:
                cur_idx += 1
                continue
            new_seq.append(item)
        return new_seq


# def _ensmeble_sim_models(top_k_one, top_k_two):
#     # only support top k = 1 case so far
# #     print("offline: ",top_k_one, "online: ", top_k_two)
#     if top_k_one[0][1] >= top_k_two[0][1]:
#         return [top_k_one[0][0]]
#     else:
#         return [top_k_two[0][0]]


# class Insert(object):
#     """Insert similar items every time call"""
#     def __init__(self, item_similarity_model, insert_ratio=0.4, max_insert_num_per_pos=1,
#             augment_threshold=14):
#         self.augment_threshold = augment_threshold
#         if type(item_similarity_model) is list:
#             self.item_sim_model_1 = item_similarity_model[0]
#             self.item_sim_model_2 = item_similarity_model[1]
#             self.ensemble = True
#         else:
#             self.item_similarity_model = item_similarity_model
#             self.ensemble = False
#         self.insert_rate = insert_ratio
#         self.max_insert_num_per_pos = max_insert_num_per_pos
        
#     def __call__(self, sequence):
#         # make a deep copy to avoid original sequence be modified
#         copied_sequence = copy.deepcopy(sequence)
#         insert_nums = max(int(self.insert_rate*len(copied_sequence)), 1)
#         insert_idx = random.sample([i for i in range(len(copied_sequence))], k = insert_nums)
#         inserted_sequence = []
#         for index, item in enumerate(copied_sequence):
#             if index in insert_idx:
#                 top_k = random.randint(1, max(1, int(self.max_insert_num_per_pos/insert_nums)))
#                 if self.ensemble:
#                     top_k_one = self.item_sim_model_1.most_similar(item,
#                                             top_k=top_k, with_score=True)
#                     top_k_two = self.item_sim_model_2.most_similar(item,
#                                             top_k=top_k, with_score=True)
#                     inserted_sequence += _ensmeble_sim_models(top_k_one, top_k_two)
#                 else:
#                     inserted_sequence += self.item_similarity_model.most_similar(item,
#                                             top_k=top_k)
#             inserted_sequence += [item]

#         return inserted_sequence

                    
# class Substitute(object):
#     """Substitute with similar items"""
#     def __init__(self, item_similarity_model, substitute_ratio=0.1):
#         if type(item_similarity_model) is list:
#             self.item_sim_model_1 = item_similarity_model[0]
#             self.item_sim_model_2 = item_similarity_model[1]
#             self.ensemble = True
#         else:
#             self.item_similarity_model = item_similarity_model
#             self.ensemble = False
#         self.substitute_rate = substitute_ratio

#     def __call__(self, sequence):
#         # make a deep copy to avoid original sequence be modified
#         copied_sequence = copy.deepcopy(sequence)
#         substitute_nums = max(int(self.substitute_rate*len(copied_sequence)), 1)
#         substitute_idx = random.sample([i for i in range(len(copied_sequence))], k = substitute_nums)
#         inserted_sequence = []
#         for index in substitute_idx:
#             if self.ensemble:
#                 top_k_one = self.item_sim_model_1.most_similar(copied_sequence[index],
#                                         with_score=True)
#                 top_k_two = self.item_sim_model_2.most_similar(copied_sequence[index],
#                                         with_score=True)
#                 substitute_items = _ensmeble_sim_models(top_k_one, top_k_two)
#                 copied_sequence[index] = substitute_items[0]
#             else:
            
#                 copied_sequence[index] = copied_sequence[index] = self.item_similarity_model.most_similar(copied_sequence[index])[0]
#         return copied_sequence


# class Random(object):
#     """Randomly pick one data augmentation type every time call"""
#     def __init__(self, crop_ratio=0.2, mask_ratio=0.7, reorder_ratio=0.2, \
#                  insert_rate=0.3, max_insert_num_per_pos=3, substitute_rate=0.3, \
#                 augment_threshold=-1, augment_type_for_short='SIM', item_similarity_model=None):
#         self.augment_threshold = augment_threshold
#         self.augment_type_for_short = augment_type_for_short
#         if self.augment_threshold == -1:
#             self.data_augmentation_methods = [Crop(crop_ratio=crop_ratio), Mask(mask_ratio=mask_ratio), Reorder(reorder_ratio=reorder_ratio), 
#                                 Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                     max_insert_num_per_pos=max_insert_num_per_pos),
#                                 Substitute(item_similarity_model, substitute_ratio=substitute_rate)]
#             print("Total augmentation numbers: ", len(self.data_augmentation_methods))
#         elif self.augment_threshold > 0:
#             print("short sequence augment type:", self.augment_type_for_short)
#             if self.augment_type_for_short == 'SI':
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate)]
#             elif self.augment_type_for_short == 'SIM':
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate),
#                                     Mask(mask_ratio=mask_ratio)]

#             elif self.augment_type_for_short == 'SIR':
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate),
#                                     Reorder(reorder_ratio=mask_ratio)]
#             elif self.augment_type_for_short == 'SIC':
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate),
#                                     Crop(crop_ratio=crop_ratio)]
#             elif self.augment_type_for_short == 'SIMR':
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate),
#                                     Mask(mask_ratio=mask_ratio), Reorder(reorder_ratio=mask_ratio)]
#             elif self.augment_type_for_short == 'SIMC':
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate),
#                                     Mask(mask_ratio=mask_ratio), Crop(crop_ratio=crop_ratio)]
#             elif self.augment_type_for_short == 'SIRC':
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate),
#                                     Reorder(reorder_ratio=mask_ratio), Crop(crop_ratio=crop_ratio)]
#             else:
#                 # print("all aug set for short sequences")
#                 self.short_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                         max_insert_num_per_pos=max_insert_num_per_pos, 
#                                         augment_threshold=self.augment_threshold),
#                                     Substitute(item_similarity_model, substitute_ratio=substitute_rate),
#                                    Crop(crop_ratio=crop_ratio), Mask(mask_ratio=mask_ratio), Reorder(reorder_ratio=mask_ratio)]                
#             self.long_seq_data_aug_methods = [Insert(item_similarity_model, insert_ratio=insert_rate, 
#                                     max_insert_num_per_pos=max_insert_num_per_pos, 
#                                     augment_threshold=self.augment_threshold),
#                                 Crop(crop_ratio=crop_ratio), Mask(mask_ratio=mask_ratio), Reorder(reorder_ratio=mask_ratio),
#                                 Substitute(item_similarity_model, substitute_ratio=substitute_rate)]
#             # print("Augmentation methods for Long sequences:", len(self.long_seq_data_aug_methods))
#             # print("Augmentation methods for short sequences:", len(self.short_seq_data_aug_methods))
#         else:
#             raise ValueError("Invalid data type.")

#     def __call__(self, sequence):
#         if self.augment_threshold == -1:
#             #randint generate int x in range: a <= x <= b
#             augment_method_idx = random.randint(0, len(self.data_augmentation_methods)-1)
#             augment_method = self.data_augmentation_methods[augment_method_idx]
#             # print(augment_method.__class__.__name__) # debug usage
#             return augment_method(sequence)
#         elif self.augment_threshold > 0:
#             seq_len = len(sequence)
#             if seq_len > self.augment_threshold:
#                 #randint generate int x in range: a <= x <= b
#                 augment_method_idx = random.randint(0, len(self.long_seq_data_aug_methods)-1)
#                 augment_method = self.long_seq_data_aug_methods[augment_method_idx]
#                 # print(augment_method.__class__.__name__) # debug usage
#                 return augment_method(sequence)
#             elif seq_len <= self.augment_threshold:
#                 #randint generate int x in range: a <= x <= b
#                 augment_method_idx = random.randint(0, len(self.short_seq_data_aug_methods)-1)
#                 augment_method = self.short_seq_data_aug_methods[augment_method_idx]
#                 # print(augment_method.__class__.__name__) # debug usage
#                 return augment_method(sequence)


AUGMENTATIONS = {'crop': Crop, 'mask': Mask, 'reorder': Reorder, 'repeat': Repeat, 'drop': Drop}

