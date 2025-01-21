import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as functional
# from proxy_anchor.utils import calc_recall_at_k
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import logging
from numpy.fft import fft, ifft
import os
import re


def set_seed(seed):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)  # 一些操作使用了原子操作，不是确定性算法，不能保证可复现，设置这个禁用原子操作，保证使用确定性算法
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    # torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.backends.cudnn.benchmark = False
    # torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class TCenter(object):
    def __init__(self, length):
        self.step = 0
        self.center = 0
        self.length = length

    def update(self, new_center):
        self.step += 1
        # new_center = new_center.detach()
        if self.step > 1:
            self.center = self.center.detach()
        if self.step <= self.length // 2:
            if self.step == 1:
                self.center = new_center
            else:
                self.center = ((self.step - 1) / self.step) * self.center + (1 / self.step) * new_center
        else:
            self.center = ((self.step - 1) / self.step) * self.center + (1 / self.step) * new_center
        return self.center


def cosine_sim(e0, e1, t_center=None, metric="p", tau=0.2, loss_type="cos3"):
    e0 = functional.normalize(e0, dim=-1)
    e1 = functional.normalize(e1, dim=-1)
    center = torch.mean(e0, dim=0)
    # if t_center is not None:
    #     center = t_center.update(center)
    center = center.unsqueeze(0)
    if loss_type == "cos1":
        logit00 = torch.cosine_similarity(e0.unsqueeze(1), e0, dim=-1) / tau
        logit01 = torch.cosine_similarity(e0.unsqueeze(1), e1, dim=-1) / tau
        logit0 = torch.sum(logit00, dim=-1, keepdim=True) / logit00.shape[0]
        logit_cos = torch.cat([logit0, logit01], dim=1)
        target = torch.zeros(e0.shape[0]).cuda()
    elif loss_type == "cos2":
        logit00 = torch.cosine_similarity(e0.unsqueeze(1), center, dim=-1) / tau
        logit01 = torch.cosine_similarity(e1.unsqueeze(1), center, dim=-1) / tau
        logit_cos = torch.cat([logit00, logit01], dim=-1)
        target = torch.zeros(e0.shape[0]).cuda()
    elif loss_type == "cos3":
        logit00 = torch.cosine_similarity(e0.unsqueeze(1), center, dim=-1) / tau
        min_true = torch.min(logit00)
        logit01 = torch.cosine_similarity(e1.unsqueeze(1), center, dim=-1) / tau
        logit_cos = torch.cat([min_true.unsqueeze(0), logit01.view(-1)], dim=-1)
        target = torch.as_tensor(0).cuda()
    elif loss_type == 'cos5':
        if metric == "p":
            logit00 = (2 - 2 * torch.cosine_similarity(e0.unsqueeze(1), center, dim=-1)) / tau
            logit01 = (2 - 2 * torch.cosine_similarity(e1.unsqueeze(1), center, dim=-1)) / tau
        else:
            logit00 = torch.cosine_similarity(e0.unsqueeze(1), center, dim=-1) / tau
            logit01 = torch.cosine_similarity(e1.unsqueeze(1), center, dim=-1) / tau
        min_true = torch.min(logit00)
        max_true = torch.max(logit00)
        logit1 = torch.sum(logit01, dim=0) / e0.shape[0]
        logit_cos = torch.cat([(min_true + max_true - 1).unsqueeze(0), logit1])
        target = torch.as_tensor(0).cuda()
    elif loss_type == "cos6":
        logit00 = torch.cosine_similarity(e0.unsqueeze(1), center, dim=-1) / tau
        min_true = torch.min(logit00)
        logit01 = torch.cosine_similarity(e1.unsqueeze(1), center, dim=-1) / tau
        max_false = torch.max(logit01)
        logit_cos = torch.cat([min_true.unsqueeze(0), max_false.unsqueeze(0)], dim=-1)
        target = torch.as_tensor(0).cuda()
    elif loss_type == "cos7":
        logit00 = torch.cosine_similarity(e0.unsqueeze(1), center, dim=-1) / tau
        logit01 = torch.cosine_similarity(e1.unsqueeze(1), center, dim=-1) / tau
        logit0 = torch.sum(logit00, dim=0) / e0.shape[0]
        logit1 = torch.sum(logit01, dim=0) / e0.shape[0]
        logit_cos = torch.cat([logit0, logit1])
        target = torch.as_tensor(0).cuda()
    else:
        raise RuntimeError
    loss = functional.cross_entropy(logit_cos, target.long())
    return loss


def cosine_sim_1(e0, e1, loss_type="cos1", margin_t=0.5, margin_f=0.0):
    # e0 = functional.normalize(e0, dim=-1)
    # e1 = functional.normalize(e1, dim=-1)
    center = torch.mean(e0, dim=0)
    center = center.unsqueeze(0)
    ns = e0.shape[0]
    center = center.expand(ns, center.shape[1])
    if loss_type == "cos1":
        cos_emb_loss_t = nn.CosineEmbeddingLoss().cuda()
        cos_emb_loss_f = nn.CosineEmbeddingLoss(margin=margin_f).cuda()
        tmp_y_t = torch.ones(ns).cuda()
        tmp_y_f = - torch.ones(ns).cuda()
        loss = cos_emb_loss_t(e0, center, tmp_y_t) + cos_emb_loss_f(e1, center, tmp_y_f)
    elif loss_type == "cos2":
        logit00 = torch.cosine_similarity(e0, center, dim=-1)
        idx_t = torch.where(logit00 < margin_t)[0]
        loss_t = 0
        if len(idx_t) > 0:
            logit0 = logit00[idx_t]
            loss_t = 1 - torch.sum(logit0) / len(idx_t)
        cos_emb_loss_f = nn.CosineEmbeddingLoss(margin=margin_f).cuda()
        tmp_y_f = - torch.ones(ns).cuda()
        loss_f = cos_emb_loss_f(e1, center, tmp_y_f)
        loss = loss_t + loss_f
    else:
        raise RuntimeError
    return loss


def filter_filename_by_datatype(base_dir, mode='all'):
    """
    筛选出所需指定数据集的文件名
    :param base_dir: 要搜索的目录
    :param mode: 0 for SAMM, 1 for SMIC , 2 for CASMEII, all for combined dataset
    """
    pattern = None
    if mode == '0':
        pattern = re.compile(r"^\d+")
    elif mode == '1':
        pattern = re.compile(r"^s(?!ub)")
    elif mode == '2':
        pattern = re.compile(r"^sub")

    if pattern is None:
        return os.listdir(base_dir)
    files = []
    for entry in os.listdir(base_dir):
        if pattern.match(entry):
            files.append(entry)
    return files


class Logger:
    """
    日志记录
    """

    def __init__(self, dir, filename):
        if not os.path.exists(dir):
            os.mkdir(dir)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=dir + '/' + filename,
            filemode='a'
        )

    def __call__(self, message):
        logging.info(message)
        print(message)


# TODO 损失函数设计
class LossFunction:

    def __init__(self, classification_loss_type="cross_entropy", contrastive_loss_type=0, alpha=0.05, metric="d"):
        """
        alpha: 对比损失的权重
        """
        self.metric = metric
        self.alpha = alpha

        self.classification_loss_fn = None
        if classification_loss_type == "cross_entropy":
            self.classification_loss_fn = nn.CrossEntropyLoss()

        self.contrastive_loss_fn = None
        if contrastive_loss_type == 0:
            self.contrastive_loss_fn = contrastive_loss_multi_class_0

    def __call__(self, x_p, y_hat, y):
        """
        :param x_p: 映射到双曲空间的特征
        :param y_hat: 分类结果
        :param y: 真实类别
        """
        contrastive_loss = self.alpha * self.contrastive_loss_fn(x_p, y, metric=self.metric)
        classification_loss = (1 - self.alpha) * self.classification_loss_fn(y_hat, y)
        loss = contrastive_loss + classification_loss
        # print("contrastive_loss:", contrastive_loss, "classification_loss:", classification_loss)
        return loss, contrastive_loss, classification_loss


def contrastive_loss_multi_class_0(embeddings, labels, metric="d", tau=0.2, hyp_c=0.1):
    """
    TODO:对比损失函数 微表情分类(3分类)
    :param embeddings: 图像的特征向量，(batch_size, embedding_dim)
    :param labels: 图像的真实标签，(batch_size,)
    :param metric: 距离度量方式，e欧氏距离 p双曲距离 d双曲空间的点积形式
    :param tau: 温度参数，用于控制相似度得分的分布
    :param hyp_c: 双曲空间的曲率参数，仅在 metric 为 "p" 或 "d" 时使用
    :return: 对比损失值
    """
    if metric == "e":
        dist_f = lambda x, y: -torch.cdist(x, y, p=2)
    elif metric == "p":
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    elif metric == "d":
        dist_f = lambda x, y: -dist_matrix_d(x, y, c=hyp_c)
    else:
        print(f"{metric}: No such type metric!")
        exit(1)

    bsize = embeddings.shape[0]
    # 特征向量在双曲空间的距离矩阵 即两两样本之间的距离
    dist = dist_f(embeddings, embeddings) / tau
    # print("embeddings:", embeddings.shape, "labels:", labels.shape)

    # 计算正负样本对 正样本对为1， 负样本对为0
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().cuda()
    # print("mask:", mask.shape)

    # 排除对角线元素，即自身与自身的距离
    logits = dist - torch.diag(dist.diag())

    # 正样本对间距
    logits_pos = logits * mask
    # 负样本对间距
    logits_neg = logits * (1 - mask)

    # 计算每个样本的正样本对平均距离之和
    logits_pos_sum = torch.sum(logits_pos, dim=1, keepdim=True) / logits_pos.shape[0]

    # 计算每个样本的负样本对平均距离之和 求和以减少计算量
    logits_neg_sum = torch.sum(logits_neg, dim=1, keepdim=True) / logits_neg.shape[0]

    # 计算每个样本的对比损失
    logits = torch.cat([logits_pos_sum, logits_neg_sum], dim=1)
    logits -= logits.max(1, keepdim=True)[0].detach()
    # print("logits:", logits.shape)

    # 目标标签，正样本对的标签为0，负样本对的标签为1，全0表示希望正样本对距离减小，负样本对距离加大
    target = torch.zeros(bsize).long().cuda()

    # 对比损失
    loss = functional.cross_entropy(logits, target)

    return loss


def contrastive_loss(e0, e1, metric="e", tau=0.2, hyp_c=0.1):
    if metric == "e":
        dist_f = lambda x, y: -torch.cdist(x, y, p=2)
    elif metric == "p":
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    elif metric == "d":
        dist_f = lambda x, y: -dist_matrix_d(x, y, c=hyp_c)
    else:
        print(f"{metric}: No such type metric!")
        exit(1)
    bsize = e0.shape[0]
    target = torch.zeros(bsize).cuda()
    # logits00: 正样本对间距
    logits00 = dist_f(e0, e0) / tau
    diag = torch.diag(logits00)
    diag = torch.diag_embed(diag)
    logits00 -= diag
    # logits01: 负样本对间距
    logits01 = dist_f(e0, e1) / tau
    # logits0: 正样本和其他样本对的平均间距
    logits0 = torch.sum(logits00, dim=-1, keepdim=True) / logits00.shape[0]
    # logits: 在列上拼接，每一行表示一个样本，对其所有正样本平均间距 和 对其所有负样本对间距
    logits = torch.cat([logits0, logits01], dim=1)
    # 每一行都减去其最大值，保持数值稳定，因为后面计算交叉熵还要经过softmax处理
    logits -= logits.max(1, keepdim=True)[0].detach()
    # target是全0张量，表示希望logits每一行的第一列最小，即类内间距最小，从而实现正样本对之间的距离较小，而正负样本对之间的距离较大。
    # 第一列最小也能反映其他列的情况，因为logits经过了-max和softmax的操作
    loss = functional.cross_entropy(logits, target.long())
    return loss


def contrastive_loss1(e0, e1, t_center=None, metric="e", tau=0.2, hyp_c=0.1, loss_type="loss3"):
    if metric == "e":
        dist_f = lambda x, y: -torch.cdist(x, y, p=2)
    elif metric == "p":
        dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
    elif metric == "d":
        dist_f = lambda x, y: -dist_matrix_d(x, y, c=hyp_c)
    else:
        print(f"{metric}: No such type metric!")
        exit(1)
    bsize = e0.shape[0]
    center0 = torch.mean(e0, dim=0)
    if t_center is not None:
        center0 = t_center.update(center0)
    center0 = center0.unsqueeze(0)
    logit00 = dist_f(e0, center0) / tau
    min_true = torch.min(logit00)
    logit01 = dist_f(e1, center0) / tau
    if loss_type == "loss2":
        logit_s = torch.cat([logit00, logit01], dim=-1)
        target = torch.zeros(bsize).cuda()
    elif loss_type == "loss3":
        # min_true = min_true.detach()
        logit_s = torch.cat([min_true.unsqueeze(0), logit01.view(-1)], dim=-1)
        target = torch.as_tensor(0).cuda()
    elif loss_type == "loss4":
        logit1 = torch.sum(logit01, dim=0) / bsize
        logit_s = torch.cat([min_true.unsqueeze(0), logit1])
        target = torch.as_tensor(0).cuda()
    elif loss_type == "loss5":  # 内部差异指导的度量学习
        max_true = torch.max(logit00)
        logit1 = torch.sum(logit01, dim=0) / bsize
        logit_s = torch.cat([(min_true + max_true).unsqueeze(0), logit1])
        target = torch.as_tensor(0).cuda()
    elif loss_type == "loss6":
        max_false = torch.max(logit01)
        index = torch.where(logit00 <= max_false)[0]
        if index.shape[0] == 0:
            return 0
        logit0 = torch.sum(logit00[index], dim=0) / index.shape[0]
        logit_s = torch.cat([logit0, logit01.view(-1)], dim=-1)
        target = torch.as_tensor(0).cuda()
    elif loss_type == "loss7":
        logit0 = torch.sum(logit00, dim=0) / bsize
        logit1 = torch.sum(logit01, dim=0) / bsize
        logit_s = torch.cat([logit0, logit1])
        target = torch.as_tensor(0).cuda()
    elif loss_type == "loss8":
        # logit0 = torch.sum(logit00, dim=0) / bsize
        index = torch.where(logit01 >= min_true)[0]
        if index.shape[0] == 0:
            return 0
        logit1 = torch.sum(logit01[index], dim=0) / index.shape[0]
        logit_s = torch.cat([min_true.unsqueeze(0), logit1], dim=-1)
        target = torch.as_tensor(0).cuda()
    else:
        print("loss type error")
        raise RuntimeError
    loss = functional.cross_entropy(logit_s, target.long())
    return loss


def contrastive_loss2(e0, e1, metric="e", tau=0.2, hyp_c=0.1, loss_type="loss_e1"):
    if metric == "e":
        dist_f = lambda x, y: torch.cdist(x, y, p=2)
    elif metric == "p":
        dist_f = lambda x, y: dist_matrix(x, y, c=hyp_c)
    elif metric == "d":
        dist_f = lambda x, y: dist_matrix_d(x, y, c=hyp_c)
    else:
        print(f"{metric}: No such type metric!")
        exit(1)
    ns = e0.shape[0]
    center0 = torch.mean(e0, dim=0)
    center0 = center0.unsqueeze(0)
    logit01 = dist_f(e1, center0) / tau  # 假样本距离真样本中心的距离
    logit00 = dist_f(e0, center0) / tau  # 真样本距离真样本中心的距离
    loss_t = torch.sum(logit00, dim=0) / ns
    loss_f = torch.sum(logit01, dim=0) / ns
    margin = 0
    if loss_type == "loss_e1":
        loss = loss_t - loss_f
    elif loss_type == "loss_e2":
        log_soft = nn.LogSoftmax(dim=-1)
        t_f = torch.cat([-loss_t, -loss_f])
        t_f -= t_f.max().detach()
        ls = -log_soft(t_f)
        loss = ls[0]
    elif loss_type == "loss_e3":
        loss = (loss_t / loss_f) + margin
    elif loss_type == "loss_e4":
        log_soft = nn.LogSoftmax(dim=-1)
        t_f = torch.cat([-loss_t, -logit01.reshape(-1)])
        t_f -= t_f.max().detach()
        ls = -log_soft(t_f)
        loss = ls[0]
    elif loss_type == "loss_e5":  # 只拉远假样本和真样本中心之间的距离
        logit_01 = - dist_f(e0, e1) / tau
        loss = torch.sum(logit_01) / (ns * ns)
    elif loss_type == "loss_e6":
        loss = -loss_f
    else:
        print("loss type error")
        raise RuntimeError
    return loss


def contrastive_patch_loss(feature, metric="e", tau=0.2, hyp_c=0.1, loss_type="loss_e1", p1_area=None):
    if metric == "e":
        dist_f = lambda x, y: torch.cdist(x, y, p=2)
    elif metric == "p":
        dist_f = lambda x, y: dist_matrix(x, y, c=hyp_c)
    elif metric == "d":
        dist_f = lambda x, y: dist_matrix_d(x, y, c=hyp_c)
    else:
        print(f"{metric}: No such type metric!")
        exit(1)
    all_s = feature.shape[0]
    bs = all_s // 2
    ns = bs // 2
    x1 = feature[:bs]
    x2 = feature[bs:]
    p2_area = 1 - p1_area
    e0 = torch.cat([x1[:ns], x2[:ns]], dim=0)
    e1 = torch.cat([x1[ns:], x2[ns:]], dim=0)
    e0_area = (torch.cat([p1_area[:ns], p2_area[:ns]], dim=0)).unsqueeze(1)
    e1_area = (torch.cat([p1_area[ns:], p2_area[ns:]], dim=0)).unsqueeze(1)
    center0 = torch.mean(e0, dim=0)
    center0 = center0.unsqueeze(0)
    logit01 = dist_f(e1, center0) / tau  # 假样本距离真样本中心的距离
    logit00 = dist_f(e0, center0) / tau  # 真样本距离真样本中心的距离
    logit01 = logit01 * e1_area
    logit00 = logit00 / e0_area
    loss_t = torch.sum(logit00, dim=0) / bs
    loss_f = torch.sum(logit01, dim=0) / bs
    margin = 0
    if loss_type == "loss_e1":
        loss = loss_t - loss_f
    elif loss_type == "loss_e2":
        log_soft = nn.LogSoftmax(dim=-1)
        t_f = torch.cat([-loss_t, -loss_f])
        t_f -= t_f.max().detach()
        ls = -log_soft(t_f)
        loss = ls[0]
    elif loss_type == "loss_e3":
        loss = (loss_t / loss_f) + margin
    elif loss_type == "loss_e4":
        log_soft = nn.LogSoftmax(dim=-1)
        t_f = torch.cat([-loss_t, -logit01.reshape(-1)])
        t_f -= t_f.max().detach()
        ls = -log_soft(t_f)
        loss = ls[0]
    elif loss_type == "loss_e5":  # 只拉远假样本和真样本中心之间的距离
        logit_01 = - dist_f(e0, e1) / tau
        loss = torch.sum(logit_01) / (ns * ns)
    elif loss_type == "loss_e6":
        loss = -loss_f
    else:
        print("loss type error")
        raise RuntimeError
    return loss


def cosine_sim2(e0, e1, t_center=None, metric="p", tau=0.2, loss_type="loss_cos1"):
    e0 = functional.normalize(e0, dim=-1)
    e1 = functional.normalize(e1, dim=-1)
    bsize = e0.shape[0]
    center0 = torch.mean(e0, dim=0)
    center0 = center0.unsqueeze(0)
    if loss_type == "loss_cos1":
        logit01 = torch.cosine_similarity(e1.unsqueeze(1), center0, dim=-1) / tau
        loss = torch.sum(logit01, dim=0) / bsize
    elif loss_type == "loss_cos2":
        logit01 = torch.cosine_similarity(e1.unsqueeze(1), center0, dim=-1) / tau
        dis1 = torch.sum(logit01, dim=0) / bsize
        center1 = torch.mean(e1, dim=0)
        center1 = center1.unsqueeze(0)
        logit02 = torch.cosine_similarity(e0.unsqueeze(1), center1, dim=-1) / tau
        dis2 = torch.sum(logit02, dim=0) / bsize
        loss = dis1 + dis2
    elif loss_type == "loss_cos3":
        dis = torch.cosine_similarity(e1.unsqueeze(1), e0, dim=-1) / tau
        loss = torch.sum(dis) / (bsize * bsize)
    else:
        print("loss type error")
        raise RuntimeError
    return loss


def loss_f_compare(x_compare, num_samples, alpha, num_gpus=1):
    if num_gpus == 1:
        x_keep = x_compare[:num_samples // 2]
        x_exchange = x_compare[num_samples // 2:]
    else:
        half_size = num_samples // 2
        small_size = half_size // 2
        x_keep = torch.concat([x_compare[:small_size], x_compare[half_size:half_size + small_size]], dim=-1)
        x_exchange = torch.concat([x_compare[small_size:half_size], x_compare[half_size + small_size:]], dim=-1)
    max_keep = torch.maximum(1 + alpha - x_keep, torch.as_tensor(1))
    max_exchange = torch.maximum(1 + alpha + x_exchange, torch.as_tensor(1))
    loss_keep = torch.sum(torch.log(max_keep))
    loss_exchange = torch.sum(torch.log(max_exchange))
    loss = (loss_keep + loss_exchange) / num_samples
    return loss


def loss_f_compare_log_sum(x_compare, num_samples):
    x_keep = x_compare[:num_samples // 2]
    x_exchange = x_compare[num_samples // 2:]
    max_keep = torch.exp(-x_keep)
    max_exchange = torch.exp(x_exchange)
    loss_keep = torch.sum(torch.log(1 + max_keep))
    loss_exchange = torch.sum(torch.log(1 + max_exchange))
    loss = (loss_keep + loss_exchange)
    return loss


def loss_f_compare_hinge(x_compare, num_samples, alpha):
    x_keep = x_compare[:num_samples // 2]
    x_exchange = x_compare[num_samples // 2:]
    max_keep = torch.maximum(alpha - x_keep, torch.as_tensor(0))
    max_exchange = torch.maximum(alpha + x_exchange, torch.as_tensor(0))
    loss_keep = torch.sum(max_keep)
    loss_exchange = torch.sum(max_exchange)
    loss = loss_keep + loss_exchange
    return loss


def loss_f_compare_hinge_1(x_compare, num_samples):
    x_keep = x_compare[:num_samples // 2]
    x_exchange = x_compare[num_samples // 2:]
    loss_keep = torch.sum(-x_keep)
    loss_exchange = torch.sum(x_exchange)
    loss = loss_keep + loss_exchange
    return loss


def _tensor_dot(x, y):
    res = torch.einsum("ij,kj->ik", (x, y))
    return res


class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-5, 1 - 1e-5)
        ctx.save_for_backward(x)
        res = (torch.log_(1 + x).sub_(torch.log_(1 - x))).mul_(0.5)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        return grad_output / (1 - input ** 2)


def artanh(x):
    return Artanh.apply(x)


def _mobius_addition_batch(x, y, c):
    xy = _tensor_dot(x, y)  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    num = 1 + 2 * c * xy + c * y2.permute(1, 0)  # B x C
    num = num.unsqueeze(2) * x.unsqueeze(1)
    num = num + (1 - c * x2).unsqueeze(2) * y  # B x C x D
    denom_part1 = 1 + 2 * c * xy  # B x C
    denom_part2 = c ** 2 * x2 * y2.permute(1, 0)
    denom = denom_part1 + denom_part2
    res = num / (denom.unsqueeze(2) + 1e-5)
    return res


def _dist_matrix(x, y, c):
    sqrt_c = c ** 0.5
    return (
            2
            / sqrt_c
            * artanh(sqrt_c * torch.norm(_mobius_addition_batch(-x, y, c=c), dim=-1))
    )


def dist_matrix(x, y, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix(x, y, c)


def _dist_matrix_d(x, y, c):
    xy = torch.einsum("ij,kj->ik", (x, y))  # B x C
    x2 = x.pow(2).sum(-1, keepdim=True)  # B x 1
    y2 = y.pow(2).sum(-1, keepdim=True)  # C x 1
    sqrt_c = c ** 0.5
    num1 = 2 * c * (x2 + y2.permute(1, 0) - 2 * xy) + 1e-3
    num2 = torch.mul((1 + c * x2), (1 + c * y2.permute(1, 0)))
    return (1 / sqrt_c * torch.acos(1 - num1 / num2))


def dist_matrix_d(x, y, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _dist_matrix_d(x, y, c)


def eval_dataset(model, dl):
    all_x, all_y = [], []
    for x, y in dl:
        with torch.no_grad():
            x = x.cuda(non_blocking=True)
            all_x.append(model(x))
        all_y.append(y)
    return torch.cat(all_x), torch.cat(all_y)


def get_emb(model, dl_eval, use_metric="p"):
    model.eval()
    if use_metric == "epd":
        all_xe, all_xp, all_xd, all_y = [], [], [], []
        for x, y in dl_eval:
            with torch.no_grad():
                x = x.cuda(non_blocking=True)
                _, e, p, d = model(x)
                all_xe.append(e)
                all_xp.append(p)
                all_xd.append(d)
            all_y.append(y)
        model.train()
        return torch.cat(all_xe), torch.cat(all_xp), torch.cat(all_xd), torch.cat(all_y).cuda()
    else:
        all_x, all_y = [], []
        for x, y in dl_eval:
            with torch.no_grad():
                x = x.cuda(non_blocking=True)
                _, z = model(x)
                all_x.append(z)
            all_y.append(y)
        model.train()
        return torch.cat(all_x), torch.cat(all_y).cuda()


# def evaluate(get_emb_f, use_metric, hyp_c):
#     emb_head = get_emb_f(use_metric=use_metric)
#     if use_metric == "epd":
#         recall_head = get_recall_epd(*emb_head, hyp_c)
#     else:
#         recall_head = get_recall(*emb_head, use_metric, hyp_c)
#     return recall_head
#
#
# def get_recall(x, y, use_metric, hyp_c):
#     k_list = [1, 2, 4, 8]
#     dist_m = torch.empty(len(x), len(x), device="cuda")
#     if use_metric == "e":
#         dist_f = lambda x, y: -torch.cdist(x, y, p=2)
#     elif use_metric == "p":
#         dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
#     elif use_metric == "d":
#         dist_f = lambda x, y: -dist_matrix_d(x, y, c=hyp_c)
#     else:
#         print(f"{use_metric}: No such type metric!")
#         exit(1)
#     for i in range(len(x)):
#         dist_m[i: i + 1] = dist_f(x[i: i + 1], x)
#     y_cur = y[dist_m.topk(1 + max(k_list), largest=True)[1][:, 1:]]
#     y = y.cpu()
#     y_cur = y_cur.float().cpu()
#     recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
#     print("------")
#     print(recall)
#     return recall
#
#
# def get_recall_epd(e, p, d, y, hyp_c):
#     k_list = [1, 2, 4, 8]
#     dist_m = torch.empty(len(e), len(e), device="cuda")
#     for i in range(len(e)):
#         dist_m[i: i + 1] = -torch.cdist(e[i: i + 1], e, p=2) - dist_matrix_d(d[i: i + 1], d, hyp_c) - dist_matrix(
#             p[i: i + 1], p, hyp_c)
#
#     y_cur = y[dist_m.topk(1 + max(k_list), largest=True)[1][:, 1:]]
#     y = y.cpu()
#     y_cur = y_cur.float().cpu()
#     recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
#     print(recall)
#     return recall
#
#
# def get_recall_big_data(x, y, use_metric, hyp_c):
#     y_cur = torch.tensor([]).cuda().int()
#     number = 1000
#     k_list = [1, 10, 100, 1000]
#     if use_metric == "e":
#         dist_f = lambda x, y: -torch.cdist(x, y, p=2)
#     elif use_metric == "p":
#         dist_f = lambda x, y: -dist_matrix(x, y, c=hyp_c)
#     elif use_metric == "d":
#         dist_f = lambda x, y: -dist_matrix_d(x, y, c=hyp_c)
#     else:
#         print(f"{use_metric}: No such type metric!")
#         exit(1)
#     for i in range(len(x) // number + 1):
#         if (i + 1) * number > len(x):
#             x_s = x[i * number:]
#         else:
#             x_s = x[i * number: (i + 1) * number]
#         dist = torch.empty(len(x_s), len(x), device="cuda")
#         for i in range(len(x_s)):
#             dist[i: i + 1] = dist_f(x_s[i: i + 1], x)
#         dist = y[dist.topk(1 + max(k_list), largest=True)[1][:, 1:]]
#         y_cur = torch.cat([y_cur, dist])
#
#     y = y.cpu()
#     y_cur = y_cur.float().cpu()
#     recall = [calc_recall_at_k(y, y_cur, k) for k in k_list]
#     print(recall)
#     return recall


def auc_tpr(model, loss_func, loader, use_metric="no", srm=None):
    model.eval()
    sum_loss = 0.
    y_true_all = None
    y_pred_all = None
    all_length = len(loader)
    print("\neval", all_length)

    with torch.no_grad():
        for (j, batch) in enumerate(loader):
            x, y_true, _ = batch
            x = x.cuda()
            print("\r{}/{}".format(j, all_length), end="")
            if srm is not None:
                x = srm(x)
            if use_metric == "no":
                y_pred = model.forward(x)
            elif use_metric == "epd":
                y_pred, _, _, _ = model.forward(x)
            else:
                y_pred, _ = model.forward(x)

            loss = loss_func(y_pred, y_true.cuda())
            sum_loss += loss.detach() * len(x)

            y_pred = nn.functional.softmax(y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))
            # if j > 15:
            #     break

    sum_loss = sum_loss / len(y_true_all)
    y_true_all = y_true_all.detach()
    y_pred_all = y_pred_all.detach()

    ap, acc, AUC, TPR_4, eer = cal_res(y_true_all, y_pred_all)
    print("------")
    print("AUC:%.6f acc:%.6f loss:%.6f TPR_4:%.6f EER:%.6f" % (AUC, acc, sum_loss, TPR_4, eer))
    metrics = [AUC, acc, sum_loss.item(), TPR_4, eer]
    model.train()
    return metrics


def fuse_auc_tpr(model, loss_func, loader, use_metric="no", fuse_type=0):
    model.eval()
    sum_loss = 0.
    y_true_all = None
    y_pred_all = None
    all_length = len(loader)
    print("\neval", all_length)

    with torch.no_grad():
        for (j, batch) in enumerate(loader):
            x, y_true, x_f = batch
            print("\r{}/{}".format(j, all_length), end="")
            if fuse_type != 1 and fuse_type != 6:
                if use_metric == "no":
                    y_pred = model(x.cuda(), x_f.cuda())
                else:
                    y_pred, _, _ = model(x.cuda(), x_f.cuda())
            else:
                if use_metric == "no":
                    y_pred = model(x.cuda())
                else:
                    y_pred, _ = model(x.cuda())

            loss = loss_func(y_pred, y_true.cuda())
            sum_loss += loss.detach() * len(x)

            y_pred = nn.functional.softmax(y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))
            # break

    sum_loss = sum_loss / len(y_true_all)
    y_true_all = y_true_all.detach()
    y_pred_all = y_pred_all.detach()

    ap, acc, AUC, TPR_4, eer = cal_res(y_true_all, y_pred_all)
    print("------")
    print("AUC:%.6f acc:%.6f loss:%.6f TPR_4:%.6f EER:%.6f" % (AUC, acc, sum_loss, TPR_4, eer))
    metrics = [AUC, acc, sum_loss.item(), TPR_4, eer]
    model.train()
    return metrics


def block_auc_tpr(model, loss_func, loader, use_metric="no", only_eval=False):
    model.eval()
    sum_loss = 0.
    y_true_all = None
    y_pred_all = None
    all_length = len(loader)
    print("\neval", all_length)

    with torch.no_grad():
        for (j, batch) in enumerate(loader):
            x1, x2, x3, x4, y_true, _ = batch
            print("\r{}/{}".format(j, all_length), end="")
            x = torch.concat([x1, x2, x3, x4], dim=0)
            if use_metric == "no":
                y_pred = model.forward(x.cuda())
            else:
                y_pred, _ = model.forward(x.cuda())
            # z_all = torch.concat(
            #     [y_pred_1.unsqueeze(1), y_pred_2.unsqueeze(1), y_pred_3.unsqueeze(1), y_pred_4.unsqueeze(1)], dim=1)
            # y_pred = None
            # for i in range(z_all.shape[0]):
            #     index = 0
            #     max_dif = z_all[i][0][1] - z_all[i][0][0]
            #     for b in range(1, 4):
            #         if z_all[i][b][1] - z_all[i][b][0] > max_dif:
            #             max_dif = z_all[i][b][1] - z_all[i][b][0]
            #             index = b
            #     if y_pred is None:
            #         y_pred = z_all[i][index].unsqueeze(0)
            #     else:
            #         y_pred = torch.concat([y_pred, z_all[i][index].unsqueeze(0)], dim=0)

            loss = loss_func(y_pred, y_true.cuda())
            sum_loss += loss.detach() * len(x1)

            y_pred = nn.functional.softmax(y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))
            # if j > 15:
            #     break

    sum_loss = sum_loss / len(y_true_all)
    y_true_all = y_true_all.detach()
    y_pred_all = y_pred_all.detach()

    ap, acc, AUC, TPR_4, eer = cal_res(y_true_all, y_pred_all)
    print("------")
    print("AUC:%.6f acc:%.6f loss:%.6f TPR_4:%.6f EER:%.6f" % (AUC, acc, sum_loss, TPR_4, eer))
    metrics = [AUC, acc, sum_loss.item(), TPR_4, eer]
    model.train()
    return metrics


def compare_auc_tpr(model, loss_func, loader, use_metric="no", only_eval=False):
    model.eval()
    sum_loss = 0.
    y_true_all = None
    y_pred_all = None
    all_length = len(loader)
    print("\neval", all_length)

    with torch.no_grad():
        for (j, batch) in enumerate(loader):
            x, y_true, _ = batch
            print("\r{}/{}".format(j, all_length), end="")
            if use_metric == "no":
                y_pred, _, _ = model.forward(x.cuda(), keep_idx=[0], exchange_idx=[])
            else:
                y_pred, _, _, _ = model.forward(x.cuda(), keep_idx=[0], exchange_idx=[])

            loss = loss_func(y_pred, y_true.cuda())
            sum_loss += loss.detach() * len(x)

            y_pred = nn.functional.softmax(y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))
            # if j > 15:
            #     break

    sum_loss = sum_loss / len(y_true_all)
    y_true_all = y_true_all.detach()
    y_pred_all = y_pred_all.detach()

    ap, acc, AUC, TPR_4, eer = cal_res(y_true_all, y_pred_all)
    print("------")
    print("AUC:%.6f acc:%.6f loss:%.6f TPR_4:%.6f EER:%.6f" % (AUC, acc, sum_loss, TPR_4, eer))
    metrics = [AUC, acc, sum_loss.item(), TPR_4, eer]
    model.train()
    return metrics


def scale_auc_tpr(model, loss_func, loader, use_metric="no"):
    model.eval()
    sum_loss = 0.
    y_true_all = None
    y_pred_all = None
    all_length = len(loader)
    print("\neval", all_length)

    with torch.no_grad():
        for (j, batch) in enumerate(loader):
            x, _, y_true, _ = batch
            print("\r{}/{}".format(j, all_length), end="")
            if use_metric == "no":
                y_pred, _, _, _, _, _ = model.forward(x.cuda())
            else:
                y_pred, _, _, _, _, _, _ = model.forward(x.cuda())

            loss = loss_func(y_pred, y_true.cuda())
            sum_loss += loss.detach() * len(x)

            y_pred = nn.functional.softmax(y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))
            # if j > 15:
            #     break

    sum_loss = sum_loss / len(y_true_all)
    y_true_all = y_true_all.detach()
    y_pred_all = y_pred_all.detach()

    ap, acc, AUC, TPR_4, eer = cal_res(y_true_all, y_pred_all)
    print("------")
    print("AUC:%.6f acc:%.6f loss:%.6f TPR_4:%.6f EER:%.6f" % (AUC, acc, sum_loss, TPR_4, eer))
    metrics = [AUC, acc, sum_loss.item(), TPR_4, eer]
    model.train()
    return metrics


def fuse_fre_auc_tpr(model, loss_func, loader, use_metric="no"):
    model.eval()
    sum_loss = 0.
    y_true_all = None
    y_pred_all = None
    all_length = len(loader)
    print("\neval", all_length)

    with torch.no_grad():
        for (j, batch) in enumerate(loader):
            x, x_f, y_true, x_l = batch
            print("\r{}/{}".format(j, all_length), end="")
            if use_metric == "no":
                y_pred, _, _ = model.forward(x.cuda(), x_f.cuda(), x_l.cuda())
            else:
                y_pred, _, _, _ = model.forward(x.cuda(), x_f.cuda(), x_l.cuda())

            loss = loss_func(y_pred, y_true.cuda())
            sum_loss += loss.detach() * len(x)

            y_pred = nn.functional.softmax(y_pred.detach(), dim=1)[:, 1].flatten()

            if y_true_all is None:
                y_true_all = y_true
                y_pred_all = y_pred
            else:
                y_true_all = torch.cat((y_true_all, y_true))
                y_pred_all = torch.cat((y_pred_all, y_pred))
            # if j > 15:
            #     break

    sum_loss = sum_loss / len(y_true_all)
    y_true_all = y_true_all.detach()
    y_pred_all = y_pred_all.detach()

    ap, acc, AUC, TPR_4, eer = cal_res(y_true_all, y_pred_all)
    print("------")
    print("AUC:%.6f acc:%.6f loss:%.6f TPR_4:%.6f EER:%.6f" % (AUC, acc, sum_loss, TPR_4, eer))
    metrics = [AUC, acc, sum_loss.item(), TPR_4, eer]
    model.train()
    return metrics


def cal_eer(tpr, fpr):
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


def cal_res(y_true_all, y_pred_all):
    y_true_all, y_pred_all = np.array(
        y_true_all.cpu()), np.array(y_pred_all.cpu())

    fprs, tprs, ths = roc_curve(
        y_true_all, y_pred_all, pos_label=1, drop_intermediate=False)

    best_index = np.argmax(tprs - fprs)
    optimal_threshold = ths[best_index]
    acc = accuracy_score(y_true_all, np.where(y_pred_all >= optimal_threshold, 1, 0)) * 100.
    eer = cal_eer(tprs, fprs)

    ind = 0
    for fpr in fprs:
        if fpr > 1e-4:
            break
        ind += 1
    TPR_4 = tprs[ind - 1]

    ap = average_precision_score(y_true_all, y_pred_all)
    return ap, acc, auc(fprs, tprs), TPR_4, eer


def my_fft(img_path, img=None):
    if img is None:
        img = cv2.imread(img_path)
    else:
        img = img
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    w, h, _ = img.shape
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]
    rf = np.fft.fftshift(np.fft.fft2(R))
    gf = np.fft.fftshift(np.fft.fft2(G))
    bf = np.fft.fftshift(np.fft.fft2(B))
    # rfff = np.log(1 + np.abs(rf))
    # gfff = np.log(1 + np.abs(gf))
    # bfff = np.log(1 + np.abs(bf))
    rr = np.fft.ifftshift(rf)
    rig = np.uint8(np.real(np.fft.ifft2(rr)))
    gr = np.fft.ifftshift(gf)
    gig = np.uint8(np.real(np.fft.ifft2(gr)))
    br = np.fft.ifftshift(bf)
    big = np.uint8(np.real(np.fft.ifft2(br)))
    # mr = np.max(rfff)
    # mg = np.max(gfff)
    # mb = np.max(bfff)
    # rrf = np.uint8(rfff * (255.0 / mr))
    # ggf = np.uint8(gfff * (255.0 / mg))
    # bbf = np.uint8(bfff * (255.0 / mb))
    # mri = np.max(rig)
    # mgi = np.max(gig)
    # mbi = np.max(big)
    # r1 = rrf.reshape((w, h, 1))
    # g1 = ggf.reshape((w, h, 1))
    # b1 = bbf.reshape((w, h, 1))
    # xxx = np.concatenate([r1, g1, b1], axis=-1)
    # xx = cat(3, rig * (255.0 / mri), gig * (255.0 / mgi), big * (255.0 / mbi))
    r2 = rig.reshape((w, h, 1))
    g2 = gig.reshape((w, h, 1))
    # b2 = (big * (255.0 / mbi)).reshape((w, h, 1))
    b2 = big.reshape((w, h, 1))
    xx = np.concatenate([r2, g2, b2], axis=-1)

    # f = np.fft.fft2(img)
    # fshift = np.fft.fftshift(f)
    # res = np.log(np.abs(fshift))
    #
    # # 傅里叶逆变换
    # ishift = np.fft.ifftshift(fshift)
    # iimg = np.fft.ifft2(ishift)
    # iimg = np.abs(iimg)
    # result = fft(img)
    # # result = np.where(np.absolute(result) < 9e3, 0, result)
    #
    # # 傅里叶反变换，保留实部
    # result = ifft(result)
    # result = np.uint8(np.real(result))
    #
    # img_out = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)

    # 显示图像
    plt.subplot(131), plt.imshow(img), plt.title('Original Image')
    plt.axis('off')
    plt.subplot(133), plt.imshow(xx), plt.title('OUT Image')
    plt.axis('off')
    plt.show()


def clear_fft(img_path, img=None, time=1.0):
    if img is None:
        img = cv2.imread(img_path)
        img = cv2.resize(img, (256, 256))
    else:
        img = img
    w, h, _ = img.shape
    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    rf = np.fft.fftshift(np.fft.fft2(R))
    gf = np.fft.fftshift(np.fft.fft2(G))
    bf = np.fft.fftshift(np.fft.fft2(B))

    rf = np.where(np.log(1 + np.absolute(rf)) < time * np.log(1 + np.absolute(rf)).mean(), 0, rf)
    gf = np.where(np.log(1 + np.absolute(gf)) < time * np.log(1 + np.absolute(gf)).mean(), 0, gf)
    bf = np.where(np.log(1 + np.absolute(bf)) < time * np.log(1 + np.absolute(bf)).mean(), 0, bf)

    rr = np.fft.ifftshift(rf)
    rig = np.uint8(np.real(np.fft.ifft2(rr)))
    gr = np.fft.ifftshift(gf)
    gig = np.uint8(np.real(np.fft.ifft2(gr)))
    br = np.fft.ifftshift(bf)
    big = np.uint8(np.real(np.fft.ifft2(br)))

    r2 = rig.reshape((w, h, 1))
    g2 = gig.reshape((w, h, 1))
    b2 = big.reshape((w, h, 1))
    xx = np.concatenate([r2, g2, b2], axis=-1)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.subplot(121), plt.imshow(img), plt.title('Original Image')
    # plt.axis('off')
    # plt.subplot(122), plt.imshow(xx), plt.title('OUT Image')
    # plt.axis('off')
    # plt.show()
    return xx


def only_fft(img_path, img=None):
    if img is None:
        img = Image.open(img_path)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
    else:
        img = img
    w, h, _ = img.shape
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    rf = np.fft.fftshift(np.fft.fft2(R))
    gf = np.fft.fftshift(np.fft.fft2(G))
    bf = np.fft.fftshift(np.fft.fft2(B))
    # plt.imshow(rf)
    # plt.show()

    rf = np.log(1 + np.absolute(rf))
    gf = np.log(1 + np.absolute(gf))
    bf = np.log(1 + np.absolute(bf))
    plt.imshow(rf)
    plt.show()
    r2 = rf.reshape((w, h, 1))
    g2 = gf.reshape((w, h, 1))
    b2 = bf.reshape((w, h, 1))
    x = np.concatenate([r2, g2, b2], axis=-1)
    return x


def whole_img_dct(img_path, thr):
    img_u8 = cv2.imread(img_path)
    img_u8 = cv2.resize(img_u8, (256, 256))
    img_f32 = img_u8.astype(np.float32)  # 数据类型转换 转换为浮点型
    # img_dct = cv2.dct(img_f32)
    # img_dct_log = np.log(abs(img_dct))  # 进行log处理
    # img_dct = np.where(img_dct > thr, 0, img_dct)
    # img_idct = cv2.idct(img_dct)
    # img_idct = np.uint8(img_idct)

    w, h, _ = img_f32.shape
    B = img_f32[:, :, 0]
    G = img_f32[:, :, 1]
    R = img_f32[:, :, 2]
    img_dct_b = cv2.dct(B)  # 进行离散余弦变换
    img_dct_g = cv2.dct(G)
    img_dct_r = cv2.dct(R)
    # img_mask = np.ones_like(B)
    # for i in range(thr):
    #     for j in range(thr - i):
    #         img_mask[i, j] = 0
    # img_dct_b = img_dct_b * img_mask
    # img_dct_g = img_dct_g * img_mask
    # img_dct_r = img_dct_r * img_mask
    img_dct_b = np.where(np.log(1 + np.abs(img_dct_b)) < thr * np.log(1 + np.abs(img_dct_b)).mean(), 0, img_dct_b)
    img_dct_g = np.where(np.log(1 + np.abs(img_dct_g)) < thr * np.log(1 + np.abs(img_dct_g)).mean(), 0, img_dct_g)
    img_dct_r = np.where(np.log(1 + np.abs(img_dct_r)) < thr * np.log(1 + np.abs(img_dct_r)).mean(), 0, img_dct_r)
    img_idct_b = cv2.idct(img_dct_b)  # 进行离散余弦反变换
    img_idct_g = cv2.idct(img_dct_g)
    img_idct_r = cv2.idct(img_dct_r)

    r = img_idct_r.reshape((w, h, 1))
    g = img_idct_g.reshape((w, h, 1))
    b = img_idct_b.reshape((w, h, 1))
    x = np.concatenate([r, g, b], axis=-1)
    x = np.uint8(x)
    return x
    # return img_u8, x


def only_dct(img_path):
    img_u8 = cv2.imread(img_path, 0)
    img_f32 = img_u8.astype(np.float)  # 数据类型转换 转换为浮点型
    img_dct = cv2.dct(img_f32)  # 进行离散余弦变换


def block_dct(img_f32, size, time=0.1):
    height, width = img_f32.shape[:2]
    block_y = height // size
    block_x = width // size
    height_ = block_y * size
    width_ = block_x * size
    img_f32_cut = img_f32[:height_, :width_]
    img_dct = np.zeros((height_, width_), dtype=np.float32)
    new_img = img_dct.copy()
    # img_mask = np.ones([size, size])
    # for i in range(1):
    #     for j in range(1 - i):
    #         img_mask[i, j] = 0
    for h in range(block_y):
        for w in range(block_x):
            # 对图像块进行dct变换
            img_block = img_f32_cut[size * h: size * (h + 1), size * w: size * (w + 1)]
            img_dct[size * h: size * (h + 1), size * w: size * (w + 1)] = cv2.dct(img_block)

            # 进行 idct 反变换
            dct_block = img_dct[size * h: size * (h + 1), size * w: size * (w + 1)]
            # dct_block = np.where(np.log(1 + np.abs(dct_block)) > np.log(1 + np.abs(dct_block)).mean(), 0, dct_block)
            dct_block = np.where(np.log(1 + np.abs(dct_block)) < time * np.log(1 + np.abs(dct_block)).mean(), 0,
                                 dct_block)
            # dct_block = dct_block * img_mask
            img_block = cv2.idct(dct_block)
            new_img[size * h: size * (h + 1), size * w: size * (w + 1)] = img_block
    new_img = new_img.reshape((height, width, 1))
    return np.uint8(new_img)
    # img_dct = np.log(1 + abs(img_dct))
    # img_dct = img_dct.reshape((height, width, 1))
    # return img_dct


# 分块图 DCT 变换
def block_img_dct(img_path, img=None, size=32, time=0.1):
    if img is None:
        img_u8 = Image.open(img_path)
        img_u8 = np.array(img_u8)
        img_u8 = cv2.resize(img_u8, (256, 256))
    else:
        img_u8 = img
    img_f32 = img_u8.astype(np.float32)
    R = img_f32[:, :, 0]
    G = img_f32[:, :, 1]
    B = img_f32[:, :, 2]
    r = block_dct(R, size, time)
    g = block_dct(G, size, time)
    b = block_dct(B, size, time)
    x = np.concatenate([r, g, b], axis=-1)
    # return img_u8, x
    return x


def block_img_dct1(img_path, img=None, size=32):
    if img is None:
        img = Image.open(img_path)
        img = np.array(img)
        img = cv2.resize(img, (256, 256))
        img_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_f32 = img_u8.astype(np.float64)
    i_img_f32 = block_dct(img_f32, size)
    x = np.concatenate([img, i_img_f32], axis=-1)
    return img_u8, x


if __name__ == '__main__':
    thr = 5
    block_num = 32
    # Deepfakes Face2Face FaceSwap NeuralTextures FaceShifter
    # clear_fft(img_path="/share/users/czt/FF/test/original_sequences/000/0000.png", time=thr)  # 1.0-1.3
    # clear_fft(img_path="/share/users/czt/FF/test/NeuralTextures/000/0000.png", time=thr)
    o1, c1 = whole_img_dct(img_path="/share/users/czt/FF/original_sequences/c23/images_2/000/0000.png", thr=thr)
    o2, c2 = whole_img_dct(img_path="/share/users/czt/FF/manipulated_sequences/Deepfakes/c23/images_2/000/0000.png",
                           thr=thr)
    # o1, c1 = block_img_dct(img_path="/share/users/czt/FF/original_sequences/c23/images_2/000/0000.png",
    #                        size=256 // block_num)
    # o2, c2 = block_img_dct(
    #     img_path="/share/users/czt/FF/manipulated_sequences/NeuralTextures/c23/images_2/000/0000.png",
    #     size=256 // block_num)
    plt.subplot(121), plt.imshow(o1, cmap="gray"), plt.title('Original True')
    plt.axis('off')
    plt.subplot(122), plt.imshow(o2, cmap="gray"), plt.title('Original False')
    plt.axis('off')
    plt.show()
    plt.subplot(121), plt.imshow(c1, cmap="gray"), plt.title('Change True')
    plt.axis('off')
    plt.subplot(122), plt.imshow(c2, cmap="gray"), plt.title('Change False')
    plt.axis('off')
    plt.show()
