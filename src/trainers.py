# -*- coding: utf-8 -*-
#
# Copyright (c) 2022 salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import numpy as np
from tqdm import tqdm
import random

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler

from models import KMeans
from datasets import RecWithContrastiveLearningDataset
from modules import NCELoss, NTXent, SupConLoss, PCLoss
from utils import recall_at_k, ndcg_k, get_metric, get_user_seqs, nCr


class Trainer:
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model

        self.num_intent_clusters = [int(i) for i in self.args.num_intent_clusters.split(",")]
        self.clusters = []
        for num_intent_cluster in self.num_intent_clusters:
            # initialize Kmeans
            if self.args.seq_representation_type == "mean":
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)
            else:
                cluster = KMeans(
                    num_cluster=num_intent_cluster,
                    seed=self.args.seed,
                    hidden_size=self.args.hidden_size * self.args.max_seq_length,
                    gpu_id=self.args.gpu_id,
                    device=self.device,
                )
                self.clusters.append(cluster)

        self.total_augmentaion_pairs = nCr(self.args.n_views, 2)
        # projection head for contrastive learn task
        self.projection = nn.Sequential(
            nn.Linear(self.args.max_seq_length * self.args.hidden_size, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, self.args.hidden_size, bias=True),
        )
        if self.cuda_condition:
            self.model.cuda()
            self.projection.cuda()
        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.cluster_dataloader = cluster_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

        self.cf_criterion = NCELoss(self.args.temperature, self.device)
        self.pcl_criterion = PCLoss(self.args.temperature, self.device)

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader, self.cluster_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort=full_sort, train=False)  # full_sort: True

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort=full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def get_sample_scores(self, epoch, pred_list):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": "{:.4f}".format(HIT_1),
            "NDCG@1": "{:.4f}".format(NDCG_1),
            "HIT@5": "{:.4f}".format(HIT_5),
            "NDCG@5": "{:.4f}".format(NDCG_5),
            "HIT@10": "{:.4f}".format(HIT_10),
            "NDCG@10": "{:.4f}".format(NDCG_10),
            "MRR": "{:.4f}".format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": "{:.4f}".format(recall[0]),
            "NDCG@5": "{:.4f}".format(ndcg[0]),
            "HIT@10": "{:.4f}".format(recall[1]),
            "NDCG@10": "{:.4f}".format(ndcg[1]),
            "HIT@20": "{:.4f}".format(recall[3]),
            "NDCG@20": "{:.4f}".format(ndcg[3]),
        }
        print(post_fix)
        with open(self.args.log_file, "a") as f:
            f.write(str(post_fix) + "\n")
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self, seq_out, pos_ids, neg_ids):  # seq_out: (256, 50, 64) 这里是经过模型后，自注意力什么乱七八糟生成的序列数据
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)  # (256, 50) --> (256, 50, 64) 将物品数据转化为embedding, 你看，这里就没有考虑一个序列的状态
        neg_emb = self.model.item_embeddings(neg_ids)  # (256, 50) --> (256, 50, 64) 将物品数据转化为embedding
        # [batch*seq_len hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(2))  # (12800, 50) reshape操作
        neg = neg_emb.view(-1, neg_emb.size(2))  # (12800, 50) reshape操作
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # (12800, 50) [batch*seq_len hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # (12800, ) [batch*seq_len]
        neg_logits = torch.sum(neg * seq_emb, -1)  # (12800, )
        istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float()  # (12800,) 这个就是序列数据的掩码，返回的是float数据 [batch*seq_len]
        loss = torch.sum(
            -torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget   # (12800,) * (12800,) = (12800,)
            - torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget)

        return loss  # Tensor(1,)

    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch hidden_size]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits

    def predict_full(self, seq_out):  # (256, 64)
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight  # (12103, 64)
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))  # (256, 12103)
        return rating_pred


class ICLRecTrainer(Trainer):
    def __init__(self, model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args):
        super(ICLRecTrainer, self).__init__(
            model, train_dataloader, cluster_dataloader, eval_dataloader, test_dataloader, args
        )

    def _instance_cl_one_pair_contrastive_learning(self, inputs, intent_ids=None):
        """
        contrastive learning given one pair sequences (batch)
        inputs:         [batch1_augmented_data, batch2_augmented_data], 形状：[(256, 50), (256, 50)] 两个增强的序列，一组有256个
        intent_ids:     是结果标签，Tensor(256, ) 就是序列的结果 比如真实序列是 [1, 2, 3, 4], 对比学习用的是[1,2,3]标签数据就是 4
        """
        cl_batch = torch.cat(inputs, dim=0)  # (512, 64) ，这个是将两个增强对(256, 64)合并了一下
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)  # (512, 50, 64) 还是理解成序列化
        # cf_sequence_output = cf_sequence_output[:, -1, :]
        if self.args.seq_representation_instancecl_type == "mean":  # 'concatenate'
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)   # (512, 3200)
        # cf_output = self.projection(cf_sequence_flatten)
        batch_size = cl_batch.shape[0] // 2
        cl_output_slice = torch.split(cl_sequence_flatten, batch_size)  # [(256, 3200), (256, 3200)]
        if self.args.de_noise:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=intent_ids)
        else:
            cl_loss = self.cf_criterion(cl_output_slice[0], cl_output_slice[1], intent_ids=None)
        return cl_loss

    def _pcl_one_pair_contrastive_learning(self, inputs, intents, intent_ids):  # inputs:[(256, 50), (256, 50)] 两个增强向量, intents: [(256, 64)] 256个意图
        """
        contrastive learning given one pair sequences (batch)           intent_ids: [(256,)] 意图的id
        inputs: [batch1_augmented_data, batch2_augmentated_data]        inputs:[(256, 50), (256, 50)] 两个增强向量
        intents: [num_clusters batch_size hidden_dims]                  intents: [(256, 64)] 256个意图
        """
        n_views, (bsz, seq_len) = len(inputs), inputs[0].shape  # seq_len: 50 序列长度, n_view: 2 增强向量数量, bsz: 256 样本数
        cl_batch = torch.cat(inputs, dim=0)     # (512, 50)
        cl_batch = cl_batch.to(self.device)
        cl_sequence_output = self.model(cl_batch)   # (512, 50, 64)
        if self.args.seq_representation_type == "mean":
            cl_sequence_output = torch.mean(cl_sequence_output, dim=1, keepdim=False)  # (512, 64)
        cl_sequence_flatten = cl_sequence_output.view(cl_batch.shape[0], -1)  # (512, 64)
        cl_output_slice = torch.split(cl_sequence_flatten, bsz)  # ((256, 64), (256, 64))
        if self.args.de_noise:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=intent_ids)
        else:
            cl_loss = self.pcl_criterion(cl_output_slice[0], cl_output_slice[1], intents=intents, intent_ids=None)
        return cl_loss

    def iteration(self, epoch, dataloader, cluster_dataloader=None, full_sort=True, train=True):
        """
        rec_cf_data_iter:       ([user_id, copied_inputs_ids, target_pos, target_neg, answer], cf_tensors_list, seq_class_label)
        rec_batch:              [user_id, copied_inputs_ids, target_pos, target_neg, answer]
        """

        str_code = "train" if train else "test"
        if train:
            # ------ intentions clustering ----- #
            if self.args.contrast_type in ["IntentCL", "Hybrid"] and epoch >= self.args.warm_up_epoches:
                print("Preparing Clustering:")
                self.model.eval()
                kmeans_training_data = []
                rec_cf_data_iter = tqdm(enumerate(cluster_dataloader), total=len(cluster_dataloader))
                for i, (rec_batch, _, _) in rec_cf_data_iter:
                    rec_batch = tuple(t.to(self.device) for t in rec_batch)
                    _, input_ids, target_pos, target_neg, _ = rec_batch    # inputs_ids: (256, 50), target_pos:(256, 50), target_neg:(256, 50)
                    # 先简单理解这个模型就是建模序列数据的过程 (256, 50) --> (256, 50, 64)
                    sequence_output = self.model(input_ids)  # sequence_output: (256, 50, 64)
                    # average sum
                    if self.args.seq_representation_type == "mean":
                        sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)  # keepDim不保持维度，所以最后是：(256, 64)
                    sequence_output = sequence_output.view(sequence_output.shape[0], -1)  # (256, 64) 类似于reshape, 但上面已经减少维度了，这里相当关于多余
                    sequence_output = sequence_output.detach().cpu().numpy()    # (256, 64) 准备k-means的数据，这个看文档解析
                    kmeans_training_data.append(sequence_output)  # [(256, 64), (256, 64) ......]
                kmeans_training_data = np.concatenate(kmeans_training_data, axis=0)  # (22363, 64) 因为这里面一共22363个用户样本

                # train multiple clusters
                print("Training Clusters:")
                for i, cluster in tqdm(enumerate(self.clusters), total=len(self.clusters)):  # cluster就是一个kmeans的类
                    cluster.train(kmeans_training_data)
                    self.clusters[i] = cluster
                # clean memory
                del kmeans_training_data
                import gc

                gc.collect()

            # ------ model training -----#
            print("Performing Rec model Training:")
            self.model.train()  # 训练模式
            rec_avg_loss = 0.0
            cl_individual_avg_losses = [0.0 for i in range(self.total_augmentaion_pairs)]
            cl_sum_avg_loss = 0.0
            joint_avg_loss = 0.0

            print(f"rec dataset length: {len(dataloader)}")
            rec_cf_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))

            for i, (rec_batch, cl_batches, seq_class_label_batches) in rec_cf_data_iter:
                """
                rec_batch shape: key_name x batch_size x feature_dim
                cl_batches shape: 
                    list of n_views x batch_size x feature_dim tensors
                """
                # 0. batch_data will be sent into the device(GPU or CPU)
                rec_batch = tuple(t.to(self.device) for t in rec_batch)
                _, input_ids, target_pos, target_neg, _ = rec_batch

                # ---------- recommendation task ---------------#
                sequence_output = self.model(input_ids)   # (256, 50, 64) 这里我们还是简单理解成建模序列化数据
                rec_loss = self.cross_entropy(sequence_output, target_pos, target_neg)  # (1,) 这一批数据（256对）算出的对比学习损失函数

                # ---------- contrastive learning task -------------#
                cl_losses = []  # 其中每一个都是一对增强向量
                for cl_batch in cl_batches:  # cl_batch: (256, 50)
                    if self.args.contrast_type == "InstanceCL":
                        cl_loss = self._instance_cl_one_pair_contrastive_learning(
                            cl_batch, intent_ids=seq_class_label_batches
                        )
                        cl_losses.append(self.args.cf_weight * cl_loss)
                    elif self.args.contrast_type == "IntentCL":
                        # ------ performing clustering for getting users' intentions ----#
                        # average sum
                        if epoch >= self.args.warm_up_epoches:
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()

                            # query on multiple clusters
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss)
                        else:
                            continue
                    elif self.args.contrast_type == "Hybrid":
                        if epoch < self.args.warm_up_epoches:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                        else:
                            cl_loss1 = self._instance_cl_one_pair_contrastive_learning(
                                cl_batch, intent_ids=seq_class_label_batches
                            )
                            cl_losses.append(self.args.cf_weight * cl_loss1)
                            if self.args.seq_representation_type == "mean":
                                sequence_output = torch.mean(sequence_output, dim=1, keepdim=False)   # (256, 64)
                            sequence_output = sequence_output.view(sequence_output.shape[0], -1)
                            sequence_output = sequence_output.detach().cpu().numpy()
                            # query on multiple clusters
                            for cluster in self.clusters:
                                seq2intents = []
                                intent_ids = []
                                intent_id, seq2intent = cluster.query(sequence_output)  # (256,) (256, 64) 获取每个样本的聚类中心
                                seq2intents.append(seq2intent)
                                intent_ids.append(intent_id)
                            cl_loss3 = self._pcl_one_pair_contrastive_learning(
                                cl_batch, intents=seq2intents, intent_ids=intent_ids
                            )
                            cl_losses.append(self.args.intent_cf_weight * cl_loss3)

                joint_loss = self.args.rec_weight * rec_loss  # Tensor(1,)
                for cl_loss in cl_losses:
                    joint_loss += cl_loss  # 当前批次的损失函数
                self.optim.zero_grad()     # 每次进行参数更新之前，需要将梯度清零。这是因为如果不清零，梯度会累积到现有的梯度上
                joint_loss.backward()      # 这一步是反向传播的过程，它会计算joint_loss（即总损失）关于模型参数的梯度。
                self.optim.step()          # 这一步是参数更新的过程，它会根据之前计算的梯度来更新模型的参数。更新的方式依赖于你选择的优化器（如SGD、Adam等）。

                rec_avg_loss += rec_loss.item()  # 为了计算“总推荐损失”

                for i, cl_loss in enumerate(cl_losses):
                    cl_sum_avg_loss += cl_loss.item()
                joint_avg_loss += joint_loss.item()  # 为了计算“总平均损失”

            post_fix = {
                "epoch": epoch,
                "rec_avg_loss": "{:.4f}".format(rec_avg_loss / len(rec_cf_data_iter)),
                "joint_avg_loss": "{:.4f}".format(joint_avg_loss / len(rec_cf_data_iter)),
            }
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, "a") as f:
                f.write(str(post_fix) + "\n")

        else:
            rec_data_iter = tqdm(enumerate(dataloader), total=len(dataloader))
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:  # batch: [user_id, copied_inputs_ids, target_pos, target_neg, answer]
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers = batch
                    recommend_output = self.model(input_ids)  # (256, 50, 64)

                    recommend_output = recommend_output[:, -1, :]  # (256, 64)
                    # recommendation results

                    rating_pred = self.predict_full(recommend_output)  # (256, 12103)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()  # (256, )
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition T: O(n)  argsort O(nlogn)
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, target_neg, answers, sample_negs = batch
                    recommend_output = self.model.finetune(input_ids)
                    test_neg_items = torch.cat((answers, sample_negs), -1)
                    recommend_output = recommend_output[:, -1, :]

                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)

                return self.get_sample_scores(epoch, pred_list)
