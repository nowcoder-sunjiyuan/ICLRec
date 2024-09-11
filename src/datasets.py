#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
import random
import torch
from torch.utils.data import Dataset

from data_augmentation import Crop, Mask, Reorder, Random
from utils import neg_sample, nCr
import copy


class RecWithContrastiveLearningDataset(Dataset):
    """
    最终返回的数据：
    (cur_rec_tensors, cf_tensors_list, seq_class_label)
    ([user_id, copied_inputs_ids, target_pos, target_neg, answer], cf_tensors_list, seq_class_label)
        copied_inputs_ids:          输入的序列，(0, n-4)也就是尾部保留三位
        target_pos:                 正例，(1, n-3)也就是copied_inputs_ids向右滑动一位，尾部保留两位
        target_neg:                 反例，随机抽取的物品，保证物品不存在于序列中即可
        answer:                     标签数据，但是这是对比学习，所以不需要，为0
        cf_tensors_list:            产生的两个数据增强例子，三种增强方法随机选择一个
        seq_class_label:
    """

    def __init__(self, args, user_seq, test_neg_items=None, data_type="train", similarity_model_type="offline"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length
        # currently apply one transform, will extend to multiples
        self.augmentations = {
            "crop": Crop(tao=args.tao),
            "mask": Mask(gamma=args.gamma),
            "reorder": Reorder(beta=args.beta),
            "random": Random(tao=args.tao, gamma=args.gamma, beta=args.beta),
        }
        if self.args.augment_type not in self.augmentations:
            raise ValueError(f"augmentation type: '{self.args.augment_type}' is invalided")
        print(f"Creating Contrastive Learning Dataset using '{self.args.augment_type}' data augmentation")
        self.base_transform = self.augmentations[self.args.augment_type]
        # number of augmentations for each sequences, current support two，对每个序列我们增强的数量
        self.n_views = self.args.n_views

    def _one_pair_data_augmentation(self, input_ids):  # [用户点击序列]
        """
        provides two positive samples given one sequence, 提供两个正例，也就是数据增强
        augmented_seqs: 返回两个数据增强的序列，增强的方式是从三种方式随机选择（裁剪，掩码）
        """
        augmented_seqs = []
        for i in range(2):
            augmented_input_ids = self.base_transform(input_ids)  # [增强序列]，裁剪，掩码产生的
            pad_len = self.max_len - len(augmented_input_ids)
            augmented_input_ids = [0] * pad_len + augmented_input_ids

            augmented_input_ids = augmented_input_ids[-self.max_len:]

            assert len(augmented_input_ids) == self.max_len

            cur_tensors = torch.tensor(augmented_input_ids, dtype=torch.long)   # Tensor(50, )
            augmented_seqs.append(cur_tensors)
        return augmented_seqs

    def _process_sequence_label_signal(self, seq_label_signal):
        seq_class_label = torch.tensor(seq_label_signal, dtype=torch.long)
        return seq_class_label

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        """
        数据例子产生：返回的数据，包含padding后的，锚点数据，正例，负例
        copied_inputs_ids:
        target_pos: 正例，是传进来的，这里只做了padding补充
        target_neg: 负例，是获取随机序列的列表（保证每个物品都不在点击序列中），这里也进行了padding
        cur_rec_tensors: [user_id, copied_inputs_ids, target_pos, target_neg, answer]
        """
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []  # 随机产生的负例，这个负例是通过随机数产生的，并且不存在用户序列中
        seq_set = set(items)
        for _ in copied_input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(copied_input_ids)  # 假设一共点击最长是50个，不足的要补全
        copied_input_ids = [0] * pad_len + copied_input_ids  # 注意是补在前面
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        copied_input_ids = copied_input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(copied_input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(copied_input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def _add_noise_interactions(self, items):
        copied_sequence = copy.deepcopy(items)
        insert_nums = max(int(self.args.noise_ratio * len(copied_sequence)), 0)
        if insert_nums == 0:
            return copied_sequence
        insert_idx = random.choices([i for i in range(len(copied_sequence))], k=insert_nums)
        inserted_sequence = []
        for index, item in enumerate(copied_sequence):
            if index in insert_idx:
                item_id = random.randint(1, self.args.item_size - 2)
                while item_id in copied_sequence:
                    item_id = random.randint(1, self.args.item_size - 2)
                inserted_sequence += [item_id]
            inserted_sequence += [item]
        return inserted_sequence

    def __getitem__(self, index):
        """
        input_ids:          输入的数据 [:-3]
        target_pos:         正例的获取，只是类似于一个移动[1:-2]
        seq_label_signal
        answer：
        cur_rec_tensors:    [userId，锚点数据，正例，负例，answer]
        cf_tensors_list：    input_ids的两个正例，也就是两个数据增强的正例，增强方式随机选择：[比例裁剪]
                            [[增强序列1], [增强序列2]]
        seq_class_label     seq_label_signal的tensor表示
        """
        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]
        if self.data_type == "train":
            input_ids = items[:-3]  # 输入的ids也进行了裁剪
            target_pos = items[1:-2]  # 这是正例产生的过程，就是裁剪
            seq_label_signal = items[-2]
            answer = [0]  # no use
        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            items_with_noise = self._add_noise_interactions(items)
            input_ids = items_with_noise[:-1]
            target_pos = items_with_noise[1:]
            answer = [items_with_noise[-1]]
        if self.data_type == "train":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            cf_tensors_list = []
            # if n_views == 2, then it's downgraded to pair-wise contrastive learning
            total_augmentaion_pairs = nCr(self.n_views, 2)
            for i in range(total_augmentaion_pairs):
                cf_tensors_list.append(self._one_pair_data_augmentation(input_ids))

            # add supervision of sequences
            seq_class_label = self._process_sequence_label_signal(seq_label_signal)  # Tensor(1,)
            return (cur_rec_tensors, cf_tensors_list, seq_class_label)
        elif self.data_type == "valid":
            cur_rec_tensors = self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)
            return cur_rec_tensors
        else:
            cur_rec_tensors = self._data_sample_rec_task(user_id, items_with_noise, input_ids, target_pos, answer)
            return cur_rec_tensors

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.user_seq)


class SASRecDataset(Dataset):
    def __init__(self, args, user_seq, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def _data_sample_rec_task(self, user_id, items, input_ids, target_pos, answer):
        # make a deep copy to avoid original sequence be modified
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(neg_sample(seq_set, self.args.item_size))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_rec_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_rec_tensors

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0]  # no use

        elif self.data_type == "valid":
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]

        return self._data_sample_rec_task(user_id, items, input_ids, target_pos, answer)

    def __len__(self):
        return len(self.user_seq)


if __name__ == "__main__":
    import argparse
    from utils import get_user_seqs, set_seed
    from torch.utils.data import DataLoader, RandomSampler
    from tqdm import tqdm

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="../data/", type=str)
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--model_idx", default=1, type=int, help="model idenfier 10, 20, 30...")

    # data augmentation args
    parser.add_argument(
        "--base_augment_type",
        default="reorder",
        type=str,
        help="data augmentation types. Chosen from mask, crop, reorder, random.",
    )
    # model args
    parser.add_argument("--model_name", default="ICLRec", type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--max_seq_length", default=50, type=int)

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=2, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)
    ## contrastive learning related
    parser.add_argument("--temperature", default=1.0, type=float, help="softmax temperature (default:  1.0)")
    parser.add_argument(
        "--n_views", default=2, type=int, metavar="N", help="Number of augmented data for each sequence"
    )
    parser.add_argument("--cf_weight", type=float, default=0.2, help="weight of contrastive learning task")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="weight of contrastive learning task")

    # learning related
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()
    set_seed(args.seed)
    args.data_file = args.data_dir + args.data_name + ".txt"
    user_seq, max_item, valid_rating_matrix, test_rating_matrix = get_user_seqs(args.data_file)
    args.item_size = max_item + 2
    train_dataset = RecWithContrastiveLearningDataset(args, user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)
    rec_cf_data_iter = tqdm(enumerate(train_dataloader), total=len(train_dataloader))

    for i, (rec_batch, cf_batch) in rec_cf_data_iter:
        for j in range(len(rec_batch)):
            print("tensor ", j, rec_batch[j])
        print("cf_batch:", cf_batch)
        if i > 2:
            break
