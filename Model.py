from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as functional
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mlp_mult, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(dim * mlp_mult, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        dim_head = dim // heads
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, h, w, heads = *x.shape, self.heads

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h=heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


def Aggregate(dim, dim_out):
    return nn.Sequential(
        nn.Conv2d(dim, dim_out, 3, padding=1),
        LayerNorm(dim_out),
        nn.MaxPool2d(3, stride=2, padding=1)
    )


class Transformer(nn.Module):
    def __init__(self, dim, seq_len, depth, heads, mlp_mult, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pos_emb = nn.Parameter(torch.randn(seq_len))

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_mult, dropout=dropout))
            ]))

    def forward(self, x):
        *_, h, w = x.shape

        pos_emb = self.pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () () h w', h=h, w=w)
        x = x + pos_emb

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


def _project(x, c):
    norm = torch.clamp_min(x.norm(dim=-1, keepdim=True, p=2), 1e-5)
    maxnorm = (1 - 1e-3) / (c ** 0.5)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)


def project(x, *, c=1.0):
    c = torch.as_tensor(c).type_as(x)
    return _project(x, c)

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


class HTNet(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            num_classes,
            dim,
            heads,
            num_hierarchies,
            block_repeats,
            mlp_mult=4,
            channels=3,
            dim_head=64,
            dropout=0.,
            hyp_c=0.1,
            emb=128,
            clip_r=2.3
    ):
        """
        :param hyp_c:双曲空间曲率
        :param emb:双曲空间映射特征维数
        """
        super().__init__()
        assert (image_size % patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = channels * patch_size ** 2  #
        fmap_size = image_size // patch_size  #
        blocks = 2 ** (num_hierarchies - 1)  #

        seq_len = (fmap_size // blocks) ** 2  # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * heads, mults))
        layer_dims = list(map(lambda t: t * dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) h w', p1=patch_size, p2=patch_size),
            nn.Conv2d(patch_dim, layer_dims[0], 1),
        )

        block_repeats = cast_tuple(block_repeats, num_hierarchies)
        self.layers = nn.ModuleList([])
        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat
            self.layers.append(nn.ModuleList([
                Transformer(dim_in, seq_len, depth, heads, mlp_mult, dropout),
                Aggregate(dim_in, dim_out) if not is_last else nn.Identity()
            ]))

        # TODO: 分类器
        self.ce_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, num_classes)
        )
        nn.init.constant_(self.ce_head[2].bias.data, 0)
        nn.init.orthogonal_(self.ce_head[2].weight.data)

        # TODO: 双曲度量映射
        p_layer = ToPoincare(c=hyp_c, clip_r=clip_r)
        self.p_head = nn.Sequential(
            LayerNorm(last_dim),
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(last_dim, emb),
            nn.BatchNorm1d(emb),
            p_layer)
        nn.init.constant_(self.p_head[2].bias.data, 0)
        nn.init.orthogonal_(self.p_head[2].weight.data)
        # print("last_dim:", last_dim)

    def forward(self, img):
        """
        :return: p_head(x):降维映射到双曲空间后的特征 ce_head(x):分类结果
        """
        x = self.to_patch_embedding(img)
        b, c, h, w = x.shape
        num_hierarchies = len(self.layers)

        # 聚合操作 4合1逐级向上提取特征
        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), self.layers):
            block_size = 2 ** level
            x = rearrange(x, 'b c (b1 h) (b2 w) -> (b b1 b2) c h w', b1=block_size, b2=block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) c h w -> b c (b1 h) (b2 w)', b1=block_size, b2=block_size)
            x = aggregate(x)
        # print("聚合后的x:", x.shape)
        # TODO:映射到双曲空间 分类器
        x_p = self.p_head(x)
        ce = self.ce_head(x)
        return x_p, ce


# This function is to confuse three models
class Fusionmodel(nn.Module):
    def __init__(self):
        #  extend from original
        super(Fusionmodel, self).__init__()
        self.fc1 = nn.Linear(15, 3)
        self.bn1 = nn.BatchNorm1d(3)
        self.d1 = nn.Dropout(p=0.5)
        self.fc_2 = nn.Linear(6, 3)
        self.relu = nn.ReLU()
        # forward layers is to use these layers above

    def forward(self, whole_feature, l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature):
        fuse_four_features = torch.cat((l_eye_feature, l_lips_feature, nose_feature, r_eye_feature, r_lips_feature), 0)
        fuse_out = self.fc1(fuse_four_features)
        fuse_out = self.relu(fuse_out)
        fuse_out = self.d1(fuse_out)  # drop out
        fuse_whole_four_parts = torch.cat(
            (whole_feature, fuse_out), 0)
        fuse_whole_four_parts = self.relu(fuse_whole_four_parts)
        fuse_whole_four_parts = self.d1(fuse_whole_four_parts)
        out = self.fc_2(fuse_whole_four_parts)
        return out
