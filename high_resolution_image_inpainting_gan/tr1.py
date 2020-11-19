import torch
from torch import nn
from torch.nn import functional as F

# ##########################################################################################################
# This script is used to verify the attention part.
# ##########################################################################################################


def cal_patch(patch_num, mask, raw_size):
    pool = nn.MaxPool2d(raw_size // patch_num)  # patch_num=32
    patch_fb = pool(mask)  # out: [B, 1, 32, 32]
    return patch_fb


def compute_attention(feature, patch_fb):  # in: [B, C:32, 64, 64]
    b = feature.shape[0]
    feature = F.interpolate(feature, scale_factor=0.5, mode="bilinear")  # in: [B, C:32, 32, 32]
    p_fb = torch.reshape(patch_fb, [b, 32 * 32, 1])
    p_matrix = torch.bmm(p_fb, (1 - p_fb).permute([0, 2, 1]))
    f = feature.permute([0, 2, 3, 1]).reshape([b, 32 * 32, 32])
    c = cosine_Matrix(f, f) * p_matrix
    s = F.softmax(c, dim=2) * p_matrix
    return s


def attention_transfer(feature, attention):  # feature: [B, C, H, W]
    b_num, c, h, w = feature.shape
    f = extract_image_patches(feature, 32)
    f = torch.reshape(f, [b_num, f.shape[1] * f.shape[2], -1])
    f = torch.bmm(attention, f)
    f = torch.reshape(f, [b_num, 32, 32, h // 32, w // 32, c])
    # x = f.permute([0, 5, 3, 1, 4, 2])
    # x = torch.reshape(x, [b_num, c, h, w])
    y = f.permute([0, 5, 1, 3, 2, 4])
    y = torch.reshape(y, [b_num, c, h, w])
    # res = x - y
    return y


def extract_image_patches(img, patch_num):
    b, c, h, w = img.shape
    img = torch.reshape(img, [b, c, patch_num, h // patch_num, patch_num, w // patch_num])
    img = img.permute([0, 2, 4, 3, 5, 1])
    return img


def cosine_Matrix(_matrixA, _matrixB):
    _matrixA_matrixB = torch.bmm(_matrixA, _matrixB.permute([0, 2, 1]))
    _matrixA_norm = torch.sqrt((_matrixA * _matrixA).sum(axis=2)).unsqueeze(dim=2)
    _matrixB_norm = torch.sqrt((_matrixB * _matrixB).sum(axis=2)).unsqueeze(dim=2)
    return _matrixA_matrixB / torch.bmm(_matrixA_norm, _matrixB_norm.permute([0, 2, 1]))


L1Loss = nn.L1Loss()
tx = torch.rand(10, 32, 64, 64)
# mask = np.array([generate_stroke_mask([256,256]) for i in range(10)])
# mask = torch.from_numpy(mask.astype(np.float32)).permute(0, 3, 1, 2).contiguous()
# patch_fb=cal_patch(32,mask,256)
# att = compute_attention(input,patch_fb)
x = torch.eye(1024)  # 创建对角矩阵n*n
att = x.expand((10, 1024, 1024))  # 扩展维度到b维
out = attention_transfer(tx, att)
res = tx - out
print(L1Loss(tx, out))

# su = np.array([
#     [2, 8, 7, 1, 6, 5],
#     [9, 5, 4, 7, 3, 2],
#     [6, 1, 3, 8, 4, 9],
#     [8, 7, 9, 6, 5, 1],
#     [4, 2, 1, 3, 9, 8],
#     [3, 6, 5, 4, 2, 7]
# ])
# aa = torch.from_numpy(su.astype(np.float32))
# fff = torch.reshape(aa, [3,2,3,2])
# zzz = torch.reshape(aa, [2,3,2,3])
# ff=fff.permute(0,2,1,3)
# print(zzz)
