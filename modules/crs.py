import torch
import torch.nn as nn
import torch.nn.functional as F
from .common import *


class CRS(nn.Module):
    def __init__(self, 
                 num_in_ch=1, 
                 num_out_ch=1, 
                 num_feat=64, 
                 num_frame=3, 
                 num_extract_block=8, 
                 num_deformable_group=8, 
                 num_reconstruct_block=8, 
                 num_fusion_block=8, 
                 center_frame_idx=1):
        super(CRS, self).__init__()
        if center_frame_idx is None:
            self.center_frame_idx = num_frame // 2
        else:
            self.center_frame_idx = center_frame_idx

        # extract features for each frame
        self.first_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.fea_ext = make_layer(ResidualBlock, num_extract_block, num_feat=num_feat)

        # align and tsa module
        self.alignment = Alignment(num_feat=num_feat, num_deformable_group=num_deformable_group)
        self.aggregation = TSAggregation(num_feat=num_feat, num_frame=num_frame, center_frame_idx=self.center_frame_idx)

        # reconstruction
        self.fea_recon = make_layer(ResidualBlock, num_reconstruct_block, num_feat=num_feat)

        self.ste = STE()
        self.tt = TextureTransfer()
        self.fusion = Fusion(num_res_block=num_fusion_block, num_feat=num_feat)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, intra_f, intra_b, train=True):
        B, N, C, H_ORG, W_ORG = x.size()
        x_center = x[:, self.center_frame_idx, :, :, :].contiguous()

        # if H_ORG % 4 != 0 or W_ORG % 4 != 0:
        #     H_PAD = int(4*np.ceil(H_ORG/4))
        #     W_PAD = int(4*np.ceil(W_ORG/4))
        #     input = x
        #     x = torch.zeros([B, N, C, H_PAD, W_PAD], dtype=torch.float32).cuda()
        #     x[:, :, :, :H_ORG, :W_ORG] = input

        B, N, C, H, W = x.size()
        # extract features for each frame
        feat = self.lrelu(self.first_conv(x.view(-1, C, H, W)))
        feat = self.fea_ext(feat)
        feat = feat.view(B, N, -1, H, W)

        # alignment
        cur_feat = feat[:, self.center_frame_idx, :, :, :].clone()
        aligned_feat_l = []
        for i in range(N):
            nbr_feat = feat[:, i, :, :, :].clone()
            aligned_feat_l.append(self.alignment(nbr_feat, cur_feat))

        aligned_feat = torch.stack(aligned_feat_l, dim=1)  # (B, N, C, H, W)

        aligned_feat = self.aggregation(aligned_feat)
        aligned_feat = self.fea_recon(aligned_feat)

        # if H_ORG % 4 != 0 or W_ORG % 4 != 0:
        #     aligned_feat = aligned_feat[:, :, :H_ORG, :W_ORG]

        base = F.interpolate(x_center, scale_factor=2, mode='bicubic', align_corners=False)
        intraf_lr = F.interpolate(intra_f, scale_factor=0.5, mode='bicubic', align_corners=False)
        intraf_sr = F.interpolate(intraf_lr, scale_factor=2, mode='bicubic', align_corners=False)
        intrab_lr = F.interpolate(intra_b, scale_factor=0.5, mode='bicubic', align_corners=False)
        intrab_sr = F.interpolate(intrab_lr, scale_factor=2, mode='bicubic', align_corners=False)

        # print(base.size(), intraf_sr.size(), intrab_sr.size())

        cur_feat_L1, cur_feat_L2, cur_feat_L3 = self.ste(base)
        intraf_sr_feat_L1, intraf_sr_feat_L2, intraf_sr_feat_L3 = self.ste(intraf_sr)
        intraf_feat_L1, intraf_feat_L2, intraf_feat_L3 = self.ste(intra_f)
        intrab_sr_feat_L1, intrab_sr_feat_L2, intrab_sr_feat_L3 = self.ste(intrab_sr)
        intrab_feat_L1, intrab_feat_L2, intrab_feat_L3 = self.ste(intra_b)

        # print(intraf_sr.size(), intrab_sr.size())

        if train == False:
            del cur_feat_L1, cur_feat_L2, intraf_sr_feat_L1, intraf_sr_feat_L2, intraf_lr, intraf_sr, intrab_sr_feat_L1, intrab_sr_feat_L2, intrab_lr, intrab_sr

        S_f, F_lv3, F_lv2, F_lv1 = self.tt(cur_feat_L3, intraf_sr_feat_L3, intraf_feat_L1, intraf_feat_L2, intraf_feat_L3, train=train)
        S_b, B_lv3, B_lv2, B_lv1 = self.tt(cur_feat_L3, intrab_sr_feat_L3, intrab_feat_L1, intrab_feat_L2, intrab_feat_L3, train=train)

        if train == False:
            del cur_feat_L3, intraf_sr_feat_L3, intraf_feat_L1, intraf_feat_L2, intraf_feat_L3, intrab_sr_feat_L3, intrab_feat_L1, intrab_feat_L2, intrab_feat_L3
        
        # print(aligned_feat.size(), F_lv2.size())
        res = self.fusion(aligned_feat, S_f, S_b, F_lv2, F_lv1, B_lv2, B_lv1)
        out = res + base
        return out