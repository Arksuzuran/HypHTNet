import torch
import torch.nn as nn
import timm
from net.vit import ViT
from net.resnet import Resnet34, Resnet50
from net.xception import xception
import torch.nn.functional as functional
from net.crate import crate_small
from torchstat import stat
from net.vit_timm import vit_small_patch16_224
import math


def init_model(model, hyp_c=0.1, emb=128, clip_r=2.3, freeze=0, use_metric="no"):
    if model == "ViT":
        body = ViT(image_size=224, patch_size=16, dim=384, depth=12, mlp_dim=384, heads=12, dim_head=384 // 12)
        # state_dict = torch.load("/data/czt/.cache/torch/hub/checkpoints/deit_small_distilled_patch16_224-649709d9.pth")
        # body.load_state_dict(state_dict, strict=False)
    elif model == "resnet":
        body = Resnet34(embedding_size=384)
    elif model == "xception":
        body = xception(num_classes=2, pretrained=False)
    elif model == "crate":
        body = crate_small(num_classes=2, pretrained=True)
    else:
        body = timm.create_model(model, pretrained=True)
        # body = vit_small_patch16_224(pretrained=True, emb=384)
    if model == "xception":
        bdim = 2048
    elif model == "crate":
        bdim = 576
    elif model == 'vit_large_patch16_224':
        bdim = 1024
    elif model == 'vit_base_patch16_224':
        bdim = 768
    else:
        bdim = 384
    # todo
    ce_head = nn.Sequential(nn.Linear(bdim, 2))
    nn.init.constant_(ce_head[0].bias.data, 0)
    nn.init.orthogonal_(ce_head[0].weight.data)
    # ce_head = ArcfaceHead(embedding_size=bdim, num_classes=2, s=64., m1=0.2, m2=0.)

    e_head = None
    p_head = None
    d_head = None
    if use_metric == "e":
        e_head = nn.Sequential(nn.Linear(bdim, emb), nn.BatchNorm1d(emb))
        nn.init.constant_(e_head[0].bias.data, 0)
        nn.init.orthogonal_(e_head[0].weight.data)

    # TODO 双曲度量
    elif use_metric == "p":
        p_layer = ToPoincare(c=hyp_c, clip_r=clip_r)
        p_head = nn.Sequential(nn.Linear(bdim, emb), nn.BatchNorm1d(emb), p_layer)
        nn.init.constant_(p_head[0].bias.data, 0)
        nn.init.orthogonal_(p_head[0].weight.data)

    elif use_metric == "d":
        d_layer = ToProjection_hypersphere(c=hyp_c, clip_r=clip_r)
        d_head = nn.Sequential(nn.Linear(bdim, emb), nn.BatchNorm1d(emb), d_layer)
        nn.init.constant_(d_head[0].bias.data, 0)
        nn.init.orthogonal_(d_head[0].weight.data)

    rm_head(body)
    if freeze is not None and freeze != 0:
        freezer(body, freeze)
    model = HeadSwitch(body, ce_head, e_head, p_head, d_head, use_metric)
    # stat(model, (3, 224, 224))
    model.cuda().train()
    return model


class NormLayer(nn.Module):
    def forward(self, x):
        return functional.normalize(x, p=2, dim=1)


class HeadSwitch(nn.Module):
    def __init__(self, body, ce_head, e_head=None, p_head=None, d_head=None, use_metric="no"):
        super(HeadSwitch, self).__init__()
        self.body = body
        self.ce_head = ce_head
        self.e_head = e_head
        self.p_head = p_head
        self.d_head = d_head
        self.use_metric = use_metric

    def forward(self, x, freeze_body=False, label=None, patch=False):
        if freeze_body:
            with torch.no_grad():
                x = self.body(x)
        else:
            x = self.body(x)
        if patch:
            x_1 = x[:x.shape[0] // 2, :]
            x_2 = x[x.shape[0] // 2:, :]
            x_all = x_1 + x_2
            # diff_loss(x_1, x_2)
            orthogonality = torch.abs(
                torch.sum(functional.normalize(x_1, dim=-1) * functional.normalize(x_2, dim=-1), dim=-1))
            orthogonality = torch.sum(orthogonality) / x_all.shape[0]
            if self.use_metric == "no":
                if label is None:
                    x_ce = self.ce_head(x_all)
                else:
                    x_ce = self.ce_head(x_all, label)
                return x_ce, orthogonality
            elif self.use_metric == "p":
                x_p = self.p_head(x_all)
                x_ce = self.ce_head(x_p)
                return x_ce, x_p, orthogonality
        else:
            if self.use_metric == "no":
                if label is None:
                    x_ce = self.ce_head(x)
                else:
                    x_ce = self.ce_head(x, label)
                return x_ce
            if self.use_metric == "e":
                x_e = self.e_head(x)
                x_ce = self.ce_head(x_e)
                return x_ce, x_e
            elif self.use_metric == "p":
                x_ce = self.ce_head(x)
                x_p = self.p_head(x)
                # x_ce = self.ce_head(x_p)
                return x_ce, x_p
            elif self.use_metric == "d":
                x_d = self.d_head(x)
                x_ce = self.ce_head(x_d)
                return x_ce, x_d


class ArcfaceHead(nn.Module):
    def __init__(self, embedding_size=384, num_classes=2, s=64., m1=0.3, m2=0.1):
        super(ArcfaceHead, self).__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m1 = math.cos(m1)
        self.sin_m1 = math.sin(m1)
        self.th1 = math.cos(math.pi - m1)
        self.mm1 = math.sin(math.pi - m1) * m1

        self.cos_m2 = math.cos(m2)
        self.sin_m2 = math.sin(m2)
        self.th2 = math.cos(math.pi - m2)
        self.mm2 = math.sin(math.pi - m2) * m2

    def forward(self, inputs, label=None):
        # 归一化后的x与w相乘就是cos(角度)
        cosine = functional.linear(functional.normalize(inputs), functional.normalize(self.weight))
        if label is not None:
            bs = inputs.shape[0]
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
            cos_t = cosine[:bs // 2, :]
            cos_f = cosine[bs // 2:, :]
            sin_t = sine[:bs // 2, :]
            sin_f = sine[bs // 2:, :]
            phi_t = cos_t * self.cos_m1 - sin_t * self.sin_m1  # 等于cos(角度+m)
            # torch.where(a>0,a,b)      # 满足条件返回a, 不满足条件返回b
            phi_t = torch.where(cos_t.float() > self.th1, phi_t.float(), cos_t.float() - self.mm1)
            phi_f = cos_f * self.cos_m2 - sin_f * self.sin_m2
            phi_f = torch.where(cos_f.float() > self.th2, phi_f.float(), cos_f.float() - self.mm2)
            phi = torch.cat([phi_t, phi_f], dim=0)

            one_hot = torch.zeros(cosine.size()).type_as(phi).long()
            one_hot.scatter_(1, label.view(-1, 1).long(), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output *= self.s
        else:
            # cosine = functional.linear(inputs, self.weight)
            output = cosine
        return output


def diff_loss(input1, input2):
    # Zero mean
    input1_mean = torch.mean(input1, dim=0, keepdims=True)
    input2_mean = torch.mean(input2, dim=0, keepdims=True)
    input1 = input1 - input1_mean
    input2 = input2 - input2_mean

    input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

    input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

    diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))
    # diff_loss = torch.mean((input1_l2 * input2_l2).sum(dim=1).pow(2))

    return diff_loss


def freezer(model, num_block):
    def fr(m):
        for param in m.parameters():
            param.requires_grad = False

    fr(model.patch_embed)
    fr(model.pos_drop)
    for i in range(num_block):
        fr(model.blocks[i])


def rm_head(m):
    names = set(x[0] for x in m.named_children())
    target = {"head", "fc", "head_dist"}
    for x in names & target:
        m.add_module(x, nn.Identity())


class MultiSample:
    def __init__(self, transform, n=2):
        self.transform = transform
        self.num = n

    def __call__(self, x):
        return tuple(self.transform(x) for _ in range(self.num))


class ToPoincare(nn.Module):
    def __init__(self, c, clip_r=None):
        super(ToPoincare, self).__init__()
        self.register_parameter("xp", None)

        self.c = c

        self.clip_r = clip_r
        self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(
                torch.ones_like(x_norm),
                self.clip_r / x_norm
            )
            x = x * fac
        return self.grad_fix(project(expmap0(x, c=self.c), c=self.c))


class ToProjection_hypersphere(nn.Module):
    def __init__(self, c, clip_r=None):
        super(ToProjection_hypersphere, self).__init__()
        self.register_parameter("xp", None)
        self.c = c
        self.clip_r = clip_r

    def forward(self, x):
        if self.clip_r is not None:
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(
                torch.ones_like(x_norm),
                self.clip_r / x_norm
            )
            x = x * fac
        return project(dexp0(x, c=self.c), c=self.c)


def project(x, *, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)


def _project(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def expmap0(u, *, c=1.0):
    c = torch.as_tensor(c).type_as(u)
    return _expmap0(u, c)


def _expmap0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = tanh(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1


def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()


def dexp0(u, *, c=1.0):
    c = torch.as_tensor(c).type_as(u)
    return _dexp0(u, c)


def _dexp0(u, c):
    sqrt_c = c ** 0.5
    u_norm = torch.clamp_min(u.norm(dim=-1, p=2, keepdim=True), 1e-5)
    gamma_1 = torch.tan(sqrt_c * u_norm) * u / (sqrt_c * u_norm)
    return gamma_1
