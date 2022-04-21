import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.dcn import ModulatedDeformConvPack, modulated_deform_conv


class ResidualBlock(nn.Module):
    def __init__(self, num_feat=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out


def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class DCNv2Pack(ModulatedDeformConvPack):
    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(
                f'Offset abs mean is {offset_absmean}, larger than 50.')

        return modulated_deform_conv(x, offset, mask, self.weight, self.bias,
                                     self.stride, self.padding, self.dilation,
                                     self.groups, self.deformable_groups)


class Alignment(nn.Module):
    def __init__(self, num_feat=64, num_deformable_group=8):
        super(Alignment, self).__init__()
        self.offset_conv1_1 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv2_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.offset_conv2_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.offset_conv3_1 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.offset_conv3_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.offset_conv2_3 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.offset_conv1_2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)

        self.dcn = DCNv2Pack(num_feat, num_feat, 3, padding=1, deformable_groups=num_deformable_group)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_feat, ref_feat):
        feat_cat = torch.cat([nbr_feat, ref_feat], dim=1)
        offset1_1 = self.lrelu(self.offset_conv1_1(feat_cat))
        offset2_1 = self.lrelu(self.offset_conv2_1(offset1_1))
        offset2_1 = self.lrelu(self.offset_conv2_2(offset2_1))
        offset3_1 = self.lrelu(self.offset_conv3_1(offset2_1))
        offset3_1 = self.lrelu(self.offset_conv3_2(offset3_1))

        offset2_2 = self.upsample(offset3_1)
        offset2_2 = torch.cat([offset2_1, offset2_2], dim=1)
        offset2_2 = self.lrelu(self.offset_conv2_3(offset2_2))
        offset1_2 = self.upsample(offset2_2)
        offset1_2 = torch.cat([offset1_1, offset1_2], dim=1)
        offset = self.lrelu(self.offset_conv1_2(offset1_2))
        feat = self.lrelu(self.dcn(nbr_feat, offset))
        return feat


class TSAggregation(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(TSAggregation, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        B, N, C, H, W = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, C, H, W))
        embedding = embedding.view(B, N, -1, H, W)  # (B, N, C, H, W)

        corr_l = []  # correlation list
        for i in range(N):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (B, H, W)
            corr_l.append(corr.unsqueeze(1))  # (B, 1, H, W)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (B, N, H, W)
        corr_prob = corr_prob.unsqueeze(2).expand(B, N, C, H, W)
        corr_prob = corr_prob.contiguous().view(B, -1, H, W)  # (B, N*C, H, W)
        aligned_feat = aligned_feat.view(B, -1, H, W) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        # after initialization, * 2 makes (attn * 2) to be close to 1.
        feat = feat * attn * 2 + attn_add
        return feat


class STE(torch.nn.Module):
    def __init__(self, num_in_ch=1, num_extract_block=8, num_feat=64):
        super(STE, self).__init__()
        
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.resblock = make_layer(ResidualBlock, num_extract_block, num_feat=num_feat)
        self.stride_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.fea_conv_l2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.stride_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 2, 1)
        self.fea_conv_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        feat_l1 = self.lrelu(self.conv_first(x))
        feat_l1 = self.resblock(feat_l1)
        feat_l2 = self.lrelu(self.stride_conv_l2(feat_l1))
        feat_l2 = self.lrelu(self.fea_conv_l2(feat_l2))
        feat_l3 = self.lrelu(self.stride_conv_l3(feat_l2))
        feat_l3 = self.lrelu(self.fea_conv_l3(feat_l3))
        return feat_l1, feat_l2, feat_l3


class TextureTransfer(nn.Module):
    def __init__(self):
        super(TextureTransfer, self).__init__()
        self.nb_patches = 8

    def bis(self, input, dim, index):
        # batch index select
        # input: [N, ?, ?, ...]
        # dim: scalar > 0
        # index: [N, idx]
        views = [input.size(0)] + [1 if i!=dim else -1 for i in range(1, len(input.size()))] # [N, 1, -1]
        expanse = list(input.size())
        expanse[0] = -1
        expanse[dim] = -1
        # print(expanse) # [-1, Ckk, -1]
        index = index.view(views).expand(expanse) # [N, Ckk, HW]
        return torch.gather(input, dim, index)

    def forward(self, lrsr_lv3, refsr_lv3, ref_lv1, ref_lv2, ref_lv3, train=True):
        m_batchsize, _, height, width = lrsr_lv3.size()
        if train == True:
            ### search
            lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
            refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
            refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)

            refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
            lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]

            R_lv3 = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold) #[N, Hr*Wr, H*W]
            R_lv3_star, R_lv3_star_arg = torch.max(R_lv3, dim=1) #[N, H*W]

            ### transfer
            ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
            ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
            ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)

            T_lv3_unfold = self.bis(ref_lv3_unfold, 2, R_lv3_star_arg)
            T_lv2_unfold = self.bis(ref_lv2_unfold, 2, R_lv3_star_arg)
            T_lv1_unfold = self.bis(ref_lv1_unfold, 2, R_lv3_star_arg)

            T_lv3 = F.fold(T_lv3_unfold, output_size=(height, width), kernel_size=(3,3), padding=1) / (3.*3.)
            T_lv2 = F.fold(T_lv2_unfold, output_size=(height*2, width*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
            T_lv1 = F.fold(T_lv1_unfold, output_size=(height*4, width*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)

            S = R_lv3_star.view(R_lv3_star.size(0), 1, height, width)

        else:
            ### search
            lrsr_lv3_unfold  = F.unfold(lrsr_lv3, kernel_size=(3, 3), padding=1)
            refsr_lv3_unfold = F.unfold(refsr_lv3, kernel_size=(3, 3), padding=1)
            del lrsr_lv3, refsr_lv3
            refsr_lv3_unfold = refsr_lv3_unfold.permute(0, 2, 1)
            
            refsr_lv3_unfold = F.normalize(refsr_lv3_unfold, dim=2) # [N, Hr*Wr, C*k*k]
            lrsr_lv3_unfold  = F.normalize(lrsr_lv3_unfold, dim=1) # [N, C*k*k, H*W]
            lrsr_lv3_unfold_patches = lrsr_lv3_unfold.view(m_batchsize, -1, self.nb_patches, height*width//self.nb_patches)

            R_lv3_patches_star = []
            R_lv3_patches_star_arg = []
            for i in range(self.nb_patches):
                R_lv3_patch = torch.bmm(refsr_lv3_unfold, lrsr_lv3_unfold_patches[:, :, i, :])
                R_lv3_patch_star, R_lv3_patch_star_arg = torch.max(R_lv3_patch, dim=1)
                R_lv3_patches_star.append(R_lv3_patch_star)
                R_lv3_patches_star_arg.append(R_lv3_patch_star_arg)
                del R_lv3_patch, R_lv3_patch_star, R_lv3_patch_star_arg

            del refsr_lv3_unfold, lrsr_lv3_unfold, lrsr_lv3_unfold_patches
            
            # for i in range(57600):
            #     if i != (torch.cat(R_lv3_patches_star_arg, dim=1).squeeze(0).cpu().numpy())[i]:
            #         print(i, (torch.cat(R_lv3_patches_star_arg, dim=1).squeeze(0).cpu().numpy())[i])

            R_lv3_star = torch.cat(R_lv3_patches_star, dim=1)
            S = R_lv3_star.view(R_lv3_star.size(0), 1, height, width)
            del R_lv3_patches_star, R_lv3_star
            
            ### transfer
            ref_lv3_unfold = F.unfold(ref_lv3, kernel_size=(3, 3), padding=1)
            ref_lv2_unfold = F.unfold(ref_lv2, kernel_size=(6, 6), padding=2, stride=2)
            ref_lv1_unfold = F.unfold(ref_lv1, kernel_size=(12, 12), padding=4, stride=4)
            del ref_lv1, ref_lv2, ref_lv3
            
            # 3
            T_lv3_unfold_patches = []
            for i in range(self.nb_patches):
                T_lv3_unfold_patch = self.bis(ref_lv3_unfold, 2, R_lv3_patches_star_arg[i])
                T_lv3_unfold_patches.append(T_lv3_unfold_patch)
                del T_lv3_unfold_patch
            
            del ref_lv3_unfold
            T_lv3_unfold = torch.cat(T_lv3_unfold_patches, dim=2) # [N, Ckk, HW]
            del T_lv3_unfold_patches
            T_lv3 = F.fold(T_lv3_unfold, output_size=(height, width), kernel_size=(3,3), padding=1) / (3.*3.)
            del T_lv3_unfold
            # 2
            T_lv2_unfold_patches = []
            for i in range(self.nb_patches):
                T_lv2_unfold_patch = self.bis(ref_lv2_unfold, 2, R_lv3_patches_star_arg[i])
                T_lv2_unfold_patches.append(T_lv2_unfold_patch)
                del T_lv2_unfold_patch
            
            del ref_lv2_unfold
            T_lv2_unfold = torch.cat(T_lv2_unfold_patches, dim=2)
            del T_lv2_unfold_patches
            T_lv2 = F.fold(T_lv2_unfold, output_size=(height*2, width*2), kernel_size=(6,6), padding=2, stride=2) / (3.*3.)
            del T_lv2_unfold
            # 1
            T_lv1_unfold_patches = []
            for i in range(self.nb_patches):
                T_lv1_unfold_patch = self.bis(ref_lv1_unfold, 2, R_lv3_patches_star_arg[i])
                T_lv1_unfold_patches.append(T_lv1_unfold_patch)
                del T_lv1_unfold_patch
            
            del ref_lv1_unfold, R_lv3_patches_star_arg
            T_lv1_unfold = torch.cat(T_lv1_unfold_patches, dim=2)
            del T_lv1_unfold_patches
            T_lv1 = F.fold(T_lv1_unfold, output_size=(height*4, width*4), kernel_size=(12,12), padding=4, stride=4) / (3.*3.)
            del T_lv1_unfold

        return S, T_lv3, T_lv2, T_lv1
        

class Fusion(nn.Module):
    def __init__(self, num_res_block=8, num_feat=64):
        super(Fusion, self).__init__()
        self.num_res_block = num_res_block
        self.num_feat = num_feat

        self.conv11_head = nn.Conv2d(self.num_feat*2, self.num_feat, 3, 1, 1)
        self.conv12_head = nn.Conv2d(self.num_feat*2, self.num_feat, 3, 1, 1)
        self.RB11 = make_layer(ResidualBlock, num_res_block, num_feat=num_feat)
        self.conv11_tail = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)

        self.conv12 = nn.Conv2d(self.num_feat, self.num_feat*4, 3, 1, 1)
        self.ps = nn.PixelShuffle(2)

        self.conv21_head = nn.Conv2d(self.num_feat*2, self.num_feat, 3, 1, 1)
        self.conv22_head = nn.Conv2d(self.num_feat*2, self.num_feat, 3, 1, 1)
        self.RB22 = make_layer(ResidualBlock, num_res_block, num_feat=num_feat)
        self.conv22_tail = nn.Conv2d(self.num_feat, self.num_feat, 3, 1, 1)

        self.conv_tail = nn.Conv2d(self.num_feat, 1, 3, 1, 1)
        
    def forward(self, x, S_f, S_b, F_lv2, F_lv1, B_lv2, B_lv1):
        x11_res = torch.cat((x, F_lv2), dim=1)
        x12_res = torch.cat((x, B_lv2), dim=1)
        x11_res = self.conv11_head(x11_res)
        x11_res = x11_res * F.interpolate(S_f, scale_factor=2, mode='bilinear', align_corners=False)
        x12_res = self.conv12_head(x12_res)
        x12_res = x12_res * F.interpolate(S_b, scale_factor=2, mode='bilinear', align_corners=False)
        x1 = x + x11_res + x12_res

        x1_res = self.RB11(x1)
        x1_res = self.conv11_tail(x1_res)
        x1 = x1 + x1_res

        x2 = self.conv12(x1)
        x2 = self.ps(x2)

        x21_res = torch.cat((x2, F_lv1), dim=1)
        x22_res = torch.cat((x2, B_lv1), dim=1)
        x21_res = self.conv21_head(x21_res)
        x21_res = x21_res * F.interpolate(S_f, scale_factor=4, mode='bilinear', align_corners=False)
        x22_res = self.conv22_head(x22_res)
        x22_res = x22_res * F.interpolate(S_b, scale_factor=4, mode='bilinear', align_corners=False)
        x2 = x2 + x21_res + x22_res

        x2_res = self.RB22(x2)
        x2_res = self.conv22_tail(x2_res)
        x2 = x2 + x2_res
        x = self.conv_tail(x2)

        return x


