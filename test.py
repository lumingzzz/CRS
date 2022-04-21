import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import time
import struct, codecs
from modules import crs
from dataset.yuv_loader import load_yuv_seq
from torchvision import models
from ptflops import get_model_complexity_info
# from torchstat import stat
# from torchsummary import summary
# from thop import profile


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def cal_mse(im1, im2):
    mse = np.mean((im1 - im2) * (im1 - im2))
    return mse


def cal_psnr(im1, im2):
    mse = np.mean((im1 - im2) * (im1 - im2))
    psnr = 10. * np.log(1.0/mse) / np.log(10.)
    return psnr

def prepare_input(resolution):
    x1 = torch.FloatTensor(1, 3, 1, 360, 640)
    x2 = torch.FloatTensor(1, 1, 720, 1280)
    x3 = torch.FloatTensor(1, 1, 720, 1280)
    return dict(x=x1, intra_f=x2, intra_b=x3)


def params_count():
    with torch.cuda.device(0):
        net = crs.CRS(num_in_ch=1, 
                    num_out_ch=1, 
                    num_feat=64, 
                    num_frame=3, 
                    num_extract_block=8, 
                    num_deformable_group=8, 
                    num_reconstruct_block=8, 
                    num_fusion_block=8, 
                    center_frame_idx=1)
        # print(net)
        macs, params = get_model_complexity_info(net, (1,360,640), as_strings=True,
                                                input_constructor=prepare_input, print_per_layer_stat=True, verbose=True)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


# def params_count(model):
# #     """
# #     Compute the number of parameters.
# #     Args:
# #         model (model): model to count the number of parameters.
# #     """
#     return np.sum([p.numel() for p in model.parameters()]).item()


def lanczos_vvc_lras(seq, tmf_qp, width, height, frame_num, interval, visualize, new_yuv):
    tmf_dir = '/test_data/' + seq + '/crs/vvc/RA/qp_' + str(stf_qp) + '_' + str(tmf_qp) + '/'
    gt_dir = '/test_data/' + seq + '/'

    os.system('ffmpeg -s {}x{} -i {} -y -vf scale={}:{}:sws_flags=lanczos {}'.format(width//2, height//2, tmf_dir+'tmf.yuv', width, height, './temp.yuv'))
    tmf_seq_y, tmf_seq_u, tmf_seq_v = load_yuv_seq(seq_path='./temp.yuv', h=height, w=width, tot_frm=frame_num, bit=8, start_frm=0)
    gt_seq_y, gt_seq_u, gt_seq_v = load_yuv_seq(seq_path=gt_dir+seq+'.yuv', h=height, w=width, tot_frm=frame_num, bit=8, start_frm=0)

    sum_psnr = 0.0
    if visualize == True:
        file_new = open(new_yuv, "wb")
    for num in range(frame_num):
        if num%interval == 0:
            hr_y = stf_seq_y[num, :, :]
            # hr_u = stf_seq_u[num, :, :]
            # hr_v = stf_seq_v[num, :, :]
            gt_y = gt_seq_y[num, :, :]
            # gt_u = gt_seq_u[num, :, :]
            # gt_v = gt_seq_v[num, :, :]

            mse1 = cal_mse(hr_y/255.0, gt_y/255.0)
            # mse1 = (4*cal_mse(hr_y/255.0, gt_y/255.0) + cal_mse(hr_u/255.0, gt_u/255.0) + cal_mse(hr_v/255.0, gt_v/255.0))/6
            psnr1 = 10. * np.log(1.0/mse1) / np.log(10.)
            sum_psnr += psnr1
            # print('frame:', num, 'psnr:', psnr1)
            print(psnr1)

            if visualize == True:
                # 存储 y 帧
                _y =  stf_seq_y[num, :, :].flatten()
                for y in _y:
                    file_new.write(struct.pack("B", y))
                # 存储 u 帧
                _u = stf_seq_u[num, :, :].flatten()
                for u in _u:
                    file_new.write(struct.pack("B", u))
                # 存储 v 帧
                _v = stf_seq_v[num, :, :].flatten()
                for v in _v:
                    file_new.write(struct.pack("B", v))

        else:
            lr_y = tmf_seq_y[num, :, :]/255.0
            # lr_u = tmf_seq_u[num, :, :]/255.0
            # lr_v = tmf_seq_v[num, :, :]/255.0
            gt_y = gt_seq_y[num, :, :]/255.0
            # gt_u = gt_seq_u[num, :, :]/255.0
            # gt_v = gt_seq_v[num, :, :]/255.0
            
            mse1 = cal_mse(lr_y, gt_y)
            # mse1 = (4*cal_mse(lr_y, gt_y) + cal_mse(lr_u, gt_u) + cal_mse(lr_v, gt_v))/6
            psnr1 = 10. * np.log(1.0/mse1) / np.log(10.)
            sum_psnr += psnr1
            # print('frame:', num, 'psnr:', psnr1)
            print(psnr1)
            torch.cuda.empty_cache()

            if visualize == True:
                lr_y = (lr_y*255.0).astype(np.uint8)
                lr_u = (lr_u*255.0).astype(np.uint8)
                lr_v = (lr_v*255.0).astype(np.uint8)
                # 存储 y 帧
                _y =  lr_y.flatten()
                for y in _y:
                    file_new.write(struct.pack("B", y))
                # 存储 u 帧
                _u = lr_u.flatten()
                for u in _u:
                    file_new.write(struct.pack("B", u))
                # 存储 v 帧
                _v = lr_v.flatten()
                for v in _v:
                    file_new.write(struct.pack("B", v))

    if visualize == True:
        file_new.close()
    avg_psnr = sum_psnr / frame_num
    print('average psnr:', avg_psnr)


def crs_vvc_lras(seq, stf_qp, tmf_qp, width, height, frame_num, interval, visualize, new_yuv):
    stf_dir = '/test_data/' + seq + '/vvc/RA/qp' + str(stf_qp) + '/'
    tmf_dir = '/test_data/' + seq + '/dcs/vvc/RA/qp_' + str(stf_qp) + '_' + str(tmf_qp) + '/'
    gt_dir = '/test_data/' + seq + '/'

    stf_seq_y, stf_seq_u, stf_seq_v = load_yuv_seq(seq_path=stf_dir+seq+'.yuv', h=height, w=width, tot_frm=frame_num, bit=8, start_frm=0)
    tmf_seq_y, tmf_seq_u, tmf_seq_v = load_yuv_seq(seq_path=tmf_dir+'tmf.yuv', h=height//2, w=width//2, tot_frm=frame_num, bit=8, start_frm=0)
    gt_seq_y, gt_seq_u, gt_seq_v = load_yuv_seq(seq_path=gt_dir+seq+'.yuv', h=height, w=width, tot_frm=frame_num, bit=8, start_frm=0)

    sum_psnr = 0.0
    if visualize == True:
        file_new = open(new_yuv, "wb")
    for num in range(frame_num):
        if num%interval == 0:
            hr_y = stf_seq_y[num, :, :]
            # hr_u = stf_seq_u[num, :, :]
            # hr_v = stf_seq_v[num, :, :]
            gt_y = gt_seq_y[num, :, :]
            # gt_u = gt_seq_u[num, :, :]
            # gt_v = gt_seq_v[num, :, :]

            mse1 = cal_mse(hr_y/255.0, gt_y/255.0)
            # mse1 = (4*cal_mse(hr_y/255.0, gt_y/255.0) + cal_mse(hr_u/255.0, gt_u/255.0) + cal_mse(hr_v/255.0, gt_v/255.0))/6
            psnr1 = 10. * np.log(1.0/mse1) / np.log(10.)
            sum_psnr += psnr1
            # print('frame:', num, 'psnr:', psnr1)
            print(psnr1)

            if visualize == True:
                # 存储 y 帧
                _y =  stf_seq_y[num, :, :].flatten()
                for y in _y:
                    file_new.write(struct.pack("B", y))
                # 存储 u 帧
                _u = stf_seq_u[num, :, :].flatten()
                for u in _u:
                    file_new.write(struct.pack("B", u))
                # 存储 v 帧
                _v = stf_seq_v[num, :, :].flatten()
                for v in _v:
                    file_new.write(struct.pack("B", v))

        elif num%interval == 1:
            lr_y = tmf_seq_y[num, :, :]/255.0
            # lr_u = tmf_seq_u[num, :, :]/255.0
            # lr_v = tmf_seq_v[num, :, :]/255.0
            ref1_y = stf_seq_y[(num//interval)*interval, :, :]/255.0
            ref2_y = tmf_seq_y[num+1, :, :]/255.0
            stf_y_f = stf_seq_y[(num//interval)*interval, :, :]/255.0
            if (num//interval+1)*interval < frame_num:
                stf_y_b = stf_seq_y[(num//interval+1)*interval, :, :]/255.0
            else:
                stf_y_b = stf_seq_y[(num//interval)*interval, :, :]/255.0
            gt_y = gt_seq_y[num, :, :]/255.0
            # gt_u = gt_seq_u[num, :, :]/255.0
            # gt_v = gt_seq_v[num, :, :]/255.0

            lr_y = torch.FloatTensor(lr_y).unsqueeze(0).unsqueeze(0)
            # lr_u = torch.FloatTensor(lr_u).unsqueeze(0).unsqueeze(0)
            # lr_v = torch.FloatTensor(lr_v).unsqueeze(0).unsqueeze(0)
            # lr_y = F.interpolate(lr_y,scale_factor=0.5,mode='bicubic')
            # lr_y = F.interpolate(lr_y,scale_factor=2,mode='bicubic')
            # lr_u = F.interpolate(lr_u,scale_factor=2,mode='bicubic')
            # lr_v = F.interpolate(lr_v,scale_factor=2,mode='bicubic')
            ref1_y = torch.FloatTensor(ref1_y).unsqueeze(0).unsqueeze(0)
            ref1_y = F.interpolate(ref1_y,scale_factor=0.5,mode='bicubic')
            ref2_y = torch.FloatTensor(ref2_y).unsqueeze(0).unsqueeze(0)
            stf_y_f = torch.FloatTensor(stf_y_f).unsqueeze(0).unsqueeze(0)
            stf_y_b = torch.FloatTensor(stf_y_b).unsqueeze(0).unsqueeze(0)

            ref1_y = ref1_y.cuda()
            ref2_y = ref2_y.cuda()
            lr_y = lr_y.cuda()
            stf_y_f = stf_y_f.cuda()
            stf_y_b = stf_y_b.cuda()

            with torch.no_grad():
                frames_y = torch.stack([ref1_y, lr_y, ref2_y], dim=1)
                output_y = crs(frames_y, stf_y_f, stf_y_b, train=False)
                # output_y = lr_y

            output_y = output_y.clamp(0,1.0)

            output_y = output_y.squeeze(0).squeeze(0).cpu().numpy()
            # lr_u = lr_u.squeeze(0).squeeze(0).numpy()
            # lr_v = lr_v.squeeze(0).squeeze(0).numpy()
            
            mse1 = cal_mse(output_y, gt_y)
            # mse1 = (4*cal_mse(lr_y, gt_y) + cal_mse(lr_u, gt_u) + cal_mse(lr_v, gt_v))/6
            psnr1 = 10. * np.log(1.0/mse1) / np.log(10.)
            sum_psnr += psnr1
            # print('frame:', num, 'psnr:', psnr1)
            print(psnr1)
            torch.cuda.empty_cache()

            if visualize == True:
                lr_y = (lr_y*255.0).astype(np.uint8)
                lr_u = (lr_u*255.0).astype(np.uint8)
                lr_v = (lr_v*255.0).astype(np.uint8)
                # 存储 y 帧
                _y =  lr_y.flatten()
                for y in _y:
                    file_new.write(struct.pack("B", y))
                # 存储 u 帧
                _u = lr_u.flatten()
                for u in _u:
                    file_new.write(struct.pack("B", u))
                # 存储 v 帧
                _v = lr_v.flatten()
                for v in _v:
                    file_new.write(struct.pack("B", v))

        elif num%interval == interval-1:
            lr_y = tmf_seq_y[num, :, :]/255.0
            # lr_u = tmf_seq_u[num, :, :]/255.0
            # lr_v = tmf_seq_v[num, :, :]/255.0
            ref1_y = tmf_seq_y[num-1, :, :]/255.0
            ref2_y = stf_seq_y[(num//interval+1)*interval, :, :]/255.0
            stf_y_f = stf_seq_y[(num//interval)*interval, :, :]/255.0
            if (num//interval+1)*interval < frame_num:
                stf_y_b = stf_seq_y[(num//interval+1)*interval, :, :]/255.0
            else:
                stf_y_b = stf_seq_y[(num//interval)*interval, :, :]/255.0
            gt_y = gt_seq_y[num, :, :]/255.0
            # gt_u = gt_seq_u[num, :, :]/255.0
            # gt_v = gt_seq_v[num, :, :]/255.0

            lr_y = torch.FloatTensor(lr_y).unsqueeze(0).unsqueeze(0)
            # lr_u = torch.FloatTensor(lr_u).unsqueeze(0).unsqueeze(0)
            # lr_v = torch.FloatTensor(lr_v).unsqueeze(0).unsqueeze(0)
            # lr_y = F.interpolate(lr_y,scale_factor=0.5,mode='bicubic')
            # lr_y = F.interpolate(lr_y,scale_factor=2,mode='bicubic')
            # lr_u = F.interpolate(lr_u,scale_factor=2,mode='bicubic')
            # lr_v = F.interpolate(lr_v,scale_factor=2,mode='bicubic')
            ref1_y = torch.FloatTensor(ref1_y).unsqueeze(0).unsqueeze(0)
            ref2_y = torch.FloatTensor(ref2_y).unsqueeze(0).unsqueeze(0)
            ref2_y = F.interpolate(ref2_y,scale_factor=0.5,mode='bicubic')
            stf_y_f = torch.FloatTensor(stf_y_f).unsqueeze(0).unsqueeze(0)
            stf_y_b = torch.FloatTensor(stf_y_b).unsqueeze(0).unsqueeze(0)

            ref1_y = ref1_y.cuda()
            ref2_y = ref2_y.cuda()
            lr_y = lr_y.cuda()
            stf_y_f = stf_y_f.cuda()
            stf_y_b = stf_y_b.cuda()

            with torch.no_grad():
                frames_y = torch.stack([ref1_y, lr_y, ref2_y], dim=1)
                output_y = crs(frames_y, stf_y_f, stf_y_b, train=False)
                # output_y = lr_y

            output_y = output_y.clamp(0,1.0)

            output_y = output_y.squeeze(0).squeeze(0).cpu().numpy()
            # lr_u = lr_u.squeeze(0).squeeze(0).numpy()
            # lr_v = lr_v.squeeze(0).squeeze(0).numpy()
            
            mse1 = cal_mse(output_y, gt_y)
            # mse1 = (4*cal_mse(lr_y, gt_y) + cal_mse(lr_u, gt_u) + cal_mse(lr_v, gt_v))/6
            psnr1 = 10. * np.log(1.0/mse1) / np.log(10.)
            sum_psnr += psnr1
            # print('frame:', num, 'psnr:', psnr1)
            print(psnr1)
            torch.cuda.empty_cache()

            if visualize == True:
                lr_y = (lr_y*255.0).astype(np.uint8)
                lr_u = (lr_u*255.0).astype(np.uint8)
                lr_v = (lr_v*255.0).astype(np.uint8)
                # 存储 y 帧
                _y =  lr_y.flatten()
                for y in _y:
                    file_new.write(struct.pack("B", y))
                # 存储 u 帧
                _u = lr_u.flatten()
                for u in _u:
                    file_new.write(struct.pack("B", u))
                # 存储 v 帧
                _v = lr_v.flatten()
                for v in _v:
                    file_new.write(struct.pack("B", v))

        else:
            lr_y = tmf_seq_y[num, :, :]/255.0
            # lr_u = tmf_seq_u[num, :, :]/255.0
            # lr_v = tmf_seq_v[num, :, :]/255.0
            ref1_y = tmf_seq_y[num-1, :, :]/255.0
            if num+1 < frame_num:
                ref2_y = tmf_seq_y[num+1, :, :]/255.0
            else:
                ref2_y = tmf_seq_y[num-1, :, :]/255.0
            stf_y_f = stf_seq_y[(num//interval)*interval, :, :]/255.0
            if (num//interval+1)*interval < frame_num:
                stf_y_b = stf_seq_y[(num//interval+1)*interval, :, :]/255.0
            else:
                stf_y_b = stf_seq_y[(num//interval)*interval, :, :]/255.0
            gt_y = gt_seq_y[num, :, :]/255.0
            # gt_u = gt_seq_u[num, :, :]/255.0
            # gt_v = gt_seq_v[num, :, :]/255.0

            lr_y = torch.FloatTensor(lr_y).unsqueeze(0).unsqueeze(0)
            # lr_u = torch.FloatTensor(lr_u).unsqueeze(0).unsqueeze(0)
            # lr_v = torch.FloatTensor(lr_v).unsqueeze(0).unsqueeze(0)
            # lr_y = F.interpolate(lr_y,scale_factor=0.5,mode='bicubic')
            # lr_y = F.interpolate(lr_y,scale_factor=2,mode='bicubic')
            # lr_u = F.interpolate(lr_u,scale_factor=2,mode='bicubic')
            # lr_v = F.interpolate(lr_v,scale_factor=2,mode='bicubic')
            ref1_y = torch.FloatTensor(ref1_y).unsqueeze(0).unsqueeze(0)
            ref2_y = torch.FloatTensor(ref2_y).unsqueeze(0).unsqueeze(0)
            stf_y_f = torch.FloatTensor(stf_y_f).unsqueeze(0).unsqueeze(0)
            stf_y_b = torch.FloatTensor(stf_y_b).unsqueeze(0).unsqueeze(0)

            ref1_y = ref1_y.cuda()
            ref2_y = ref2_y.cuda()
            lr_y = lr_y.cuda()
            stf_y_f = stf_y_f.cuda()
            stf_y_b = stf_y_b.cuda()

            with torch.no_grad():
                frames_y = torch.stack([ref1_y, lr_y, ref2_y], dim=1)
                output_y = crs(frames_y, stf_y_f, stf_y_b, train=False)
                # output_y = lr_y

            output_y = output_y.clamp(0,1.0)

            output_y = output_y.squeeze(0).squeeze(0).cpu().numpy()
            # lr_u = lr_u.squeeze(0).squeeze(0).numpy()
            # lr_v = lr_v.squeeze(0).squeeze(0).numpy()
            
            mse1 = cal_mse(output_y, gt_y)
            # mse1 = (4*cal_mse(lr_y, gt_y) + cal_mse(lr_u, gt_u) + cal_mse(lr_v, gt_v))/6
            psnr1 = 10. * np.log(1.0/mse1) / np.log(10.)
            sum_psnr += psnr1
            # print('frame:', num, 'psnr:', psnr1)
            print(psnr1)
            torch.cuda.empty_cache()

            if visualize == True:
                lr_y = (lr_y*255.0).astype(np.uint8)
                lr_u = (lr_u*255.0).astype(np.uint8)
                lr_v = (lr_v*255.0).astype(np.uint8)
                # 存储 y 帧
                _y =  lr_y.flatten()
                for y in _y:
                    file_new.write(struct.pack("B", y))
                # 存储 u 帧
                _u = lr_u.flatten()
                for u in _u:
                    file_new.write(struct.pack("B", u))
                # 存储 v 帧
                _v = lr_v.flatten()
                for v in _v:
                    file_new.write(struct.pack("B", v))

    if visualize == True:
        file_new.close()
    avg_psnr = sum_psnr / frame_num
    print('average psnr:', avg_psnr)


if __name__=='__main__':
    # lanczos eval
    # lanczos_vvc_lras(seq='KristenAndSara_1280x720_60', tmf_qp=27, width=1280, height=720, frame_num=600, interval=32, visualize=False, new_yuv='')

    # # crs eval
    # save_checkpoint = torch.load('/workspace/lm/CRS/checkpoints/RA/qp_32_27/epoch_99_loss_0.0112790.pkl')
    # crs = crs.CRS(num_in_ch=1, 
    #               num_out_ch=1, 
    #               num_feat=64, 
    #               num_frame=3, 
    #               num_extract_block=8, 
    #               num_deformable_group=8, 
    #               num_reconstruct_block=8, 
    #               num_fusion_block=8, 
    #               center_frame_idx=1).cuda()

    # crs.load_state_dict(save_checkpoint)
    # crs_vvc_lras(seq='KristenAndSara_1280x720_60', stf_qp=32, tmf_qp=27, width=1280, height=720, frame_num=600, interval=32, visualize=False, new_yuv='')

    params_count()


