import os
import struct
import codecs


def yuv_extract(org_yuv, new_yuv, yuv_num, interval, width, height):
    # 计算各帧尺寸
    yuv_size = int(width * height * 1.5)
    y_size = int(width * height)
    u_size = int(width * height / 4)
    v_size = int(width * height / 4)

    # 打开输入文件，得到码流
    file = open(org_yuv, "rb")
    video = list(file.read())
    file_new = open(new_yuv, "wb")

    for i in range(yuv_num):
        # if i % interval == 0 or i == yuv_num-1:
        if i % interval == 0:
            # 存储 y 帧
            _y = video[i * yuv_size : i * yuv_size + y_size]
            for y in _y:
                file_new.write(struct.pack("B", y))

            # 存储 u 帧
            _u = video[i * yuv_size + y_size : i * yuv_size + y_size + u_size]
            for u in _u:
                file_new.write(struct.pack("B", u))

            # 存储 v 帧
            _v = video[i * yuv_size + y_size + u_size : (i + 1) * yuv_size]
            for v in _v:
                file_new.write(struct.pack("B", v))

    file_new.close()
    file.close()


def yuv_per_frame_buffer(yuv_seq, yuv_buffer, interval, width, height):
    if not os.path.exists(yuv_buffer):
        os.makedirs(yuv_buffer)

    # 计算各帧尺寸
    yuv_size = int(width * height * 1.5)
    y_size = int(width * height)
    u_size = int(width * height / 4)
    v_size = int(width * height / 4)

    # 计算帧数
    yuv_num = int(os.path.getsize(yuv_seq) / yuv_size)

    # 打开输入文件，得到码流
    file = open(yuv_seq, "rb")
    video = list(file.read())

    for i in range(yuv_num):
        file_new = open(yuv_buffer+'poc_'+str(interval*i)+'.yuv', "wb")
        # 存储 y 帧
        _y = video[i * yuv_size : i * yuv_size + y_size]
        for y in _y:
            file_new.write(struct.pack("B", y))
        # 存储 u 帧
        _u = video[i * yuv_size + y_size : i * yuv_size + y_size + u_size]
        for u in _u:
            file_new.write(struct.pack("B", u))
        # 存储 v 帧
        _v = video[i * yuv_size + y_size + u_size : (i + 1) * yuv_size]
        for v in _v:
            file_new.write(struct.pack("B", v))
        file_new.close()
    
    file.close()


def x265_vvc_ldp_infi(seq, stf_qp, tmf_qp, frame_num, width, height):
    yuv_org = '/test_data/' + seq + '/' + seq + '.yuv'
    yuv_l2 = '/test_data/' + seq + '/' + seq + '_l2.yuv'

    # dcs
    generate_path = '/test_data/' + seq + '/dcs/x265/LDP/qp_' + str(stf_qp) + '_' + str(tmf_qp) + '/'
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    # stf coding
    os.system('./x265-Release_3.3/source/build/x265 --input {} --fps 25 --input-res {}x{} --psnr --csv {} --csv-log-level 2 --qp {} --bframes 0 -r {} -o {}'.format(yuv_org, width, height, generate_path+'stf.log', stf_qp, generate_path+'stf.yuv', generate_path+'stf.h265'))

    # # tmf coding
    # os.system('ffmpeg -s {}x{} -i {} -y -vf scale={}:{} {}'.format(width, height, generate_path+'stf.yuv', width//2, height//2, generate_path+'stf_ds.yuv'))
    # os.system('ffmpeg -s {}x{} -i {} -y -vf pad=iw:2*ih {}'.format(width//2, height//2, generate_path+'stf_ds.yuv', generate_path+'stf_ds_w.yuv'))
    # os.system('ffmpeg -s {}x{} -i {} -y -vf crop={}:{}:0:0 {}'.format(width//2, height, generate_path+'stf_ds_w.yuv', width//2, height//2+4, generate_path+'stf_ds_pad.yuv'))
    # yuv_per_frame_buffer(yuv_seq=generate_path+'stf_ds_pad.yuv', yuv_buffer='/test_data/'+seq+'/poc_buffer/', interval=1, width=width//2, height=height//2+4)
    # for i in os.listdir('/test_data/'+seq+'/poc_buffer'):
    #     if int(i[4:-4]) % 250 != 0:
    #         os.remove('/test_data/'+seq+'/poc_buffer/'+i)

    # os.system('./x265_3.3_modified/source/build/x265 --input {} --fps 25 --input-res {}x{} --psnr --csv {} --csv-log-level 2 --qp {} --bframes 0 -r {} -o {}'.format(yuv_l2, width//2, height//2, generate_path+'tmf.log', tmf_qp, generate_path+'tmf.yuv', generate_path+'tmf.h265'))


def dcs_vvc_ldp_infi(seq, stf_qp, tmf_qp, frame_num, width, height):
    yuv_org = '/test_data/' + seq + '/' + seq + '.yuv'
    yuv_l2 = '/test_data/' + seq + '/dcs/' + seq + '_l2.yuv'

    # dcs
    generate_path = '/test_data/' + seq + '/dcs/vvc/LDP/qp_' + str(stf_qp) + '_' + str(tmf_qp) + '/'
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    # # stf coding
    # os.system('/workspace/VVCSoftware_VTM-VTM-10.0/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -fr 1 -c /workspace/VVCSoftware_VTM-VTM-10.0/cfg/encoder_lowdelay_P_vtm.cfg --OutputBitDepth=8 -q {} -f {} -ip {} -b {} -o {} > {}'.format(yuv_org, width, height, stf_qp, intra_num, interval, generate_path+'stf.bin', generate_path+'stf.yuv', generate_path+'stf.txt'))

    # tmf coding
    os.system('cp -f ./utils/Picture1.cpp ./VVCSoftware_VTM-VTM-10.0-dev/source/Lib/CommonLib/Picture.cpp')
    os.system('cd ./VVCSoftware_VTM-VTM-10.0-dev/build/ && make -j')
    os.system('cd /workspace')
    # os.system('mkdir ./VVCSoftware_VTM-VTM-10.0-dev/build')
    # os.system('cd ./codes/VVCSoftware_VTM-VTM-10.0-dev/build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j')
    os.system('rm -rf /test_data/poc_buffer/*')
    os.system('ffmpeg -s {}x{} -y -i {} -vf scale={}:{} {}'.format(width, height, '/test_data/'+seq+'/vvc/LDP/qp'+str(stf_qp)+'/'+seq+'.yuv', width//2, height//2, '/test_data/poc_buffer/'+seq+'.yuv'))
    os.system('./VVCSoftware_VTM-VTM-10.0-dev/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -c ./VVCSoftware_VTM-VTM-10.0-dev/cfg/encoder_lowdelay_P_vtm.cfg --OutputBitDepth=8 --ConformanceWindowMode=1 -fr 24 -q 51 -f {} -b ./str.bin -o ./recons.yuv'.format('/test_data/poc_buffer/'+seq+'.yuv', width//2, height//2, frame_num))
    os.remove('/test_data/poc_buffer/'+seq+'.yuv')
    for i in os.listdir('/test_data/poc_buffer/'):
        if int(i[:-4]) != 0:
            os.remove('/test_data/poc_buffer/'+i)

    os.system('cp -f ./utils/Picture0.cpp ./VVCSoftware_VTM-VTM-10.0-dev/source/Lib/CommonLib/Picture.cpp')
    os.system('cd ./VVCSoftware_VTM-VTM-10.0-dev/build/ && make -j')
    os.system('cd /workspace')
    os.system('ffmpeg -s {}x{} -i {} -y -vf scale={}:{} {}'.format(width, height, yuv_org, width//2, height//2, yuv_l2))
    os.system('./VVCSoftware_VTM-VTM-10.0-dev/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -fr 24 -c ./VVCSoftware_VTM-VTM-10.0-dev/cfg/encoder_lowdelay_P_vtm.cfg --OutputBitDepth=8 --ConformanceWindowMode=1 -q {} -f {} -b {} -o {} > {}'.format(yuv_l2, width//2, height//2, tmf_qp, frame_num, generate_path+'tmf.bin', generate_path+'tmf.yuv', generate_path+'tmf.txt'))


def crs_vvc_ra_generate(seq, stf_qp, tmf_qp, frame_num, width, height, interval):
    yuv_org = '/workspace/test_data/' + seq + '/' + seq + '.yuv'
    yuv_l2 = '/test_data/' + seq + '/dcs/' + seq + '_l2.yuv'

    # crs
    generate_path = '/test_data/' + seq + '/dcs/vvc/RA/qp_' + str(stf_qp) + '_' + str(tmf_qp) + '/'
    if not os.path.exists(generate_path):
        os.makedirs(generate_path)

    # # stf coding
    # os.system('/workspace/VVCSoftware_VTM-VTM-10.0/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -fr 1 -c /workspace/VVCSoftware_VTM-VTM-10.0/cfg/encoder_intra_vtm.cfg --OutputBitDepth=8 -q {} -f {} -ts {} -b {} -o {} > {}'.format(yuv_org, width, height, qp, frame_num, interval, generate_path+'stf.bin', generate_path+'stf.yuv', generate_path+'stf.txt'))

    # tmf coding
    os.system('cp utils/Picture1.cpp /workspace/VVCSoftware_VTM-VTM-10.0-dev/source/Lib/CommonLib/Picture.cpp')
    os.system('cd /workspace/VVCSoftware_VTM-VTM-10.0-dev/build/ && make -j')
    os.system('cd /workspace/')
    os.system('rm -rf /test_data/poc_buffer/*')
    os.system('ffmpeg -s {}x{} -y -i {} -vf scale={}:{}:sws_flags=lanczos {}'.format(width, height, '/test_data/'+seq+'/vvc/RA/qp'+str(stf_qp)+'/'+seq+'.yuv', width//2, height//2, '/test_data/poc_buffer/'+seq+'.yuv'))
    os.system('/workspace/VVCSoftware_VTM-VTM-10.0-dev/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -c /workspace/VVCSoftware_VTM-VTM-10.0-dev/cfg/encoder_randomaccess_vtm.cfg --OutputBitDepth=8 --ConformanceWindowMode=1 -fr 25 -q 63 -f {} -b str.bin -o recons.yuv'.format('/test_data/poc_buffer/'+seq+'.yuv', width//2, height//2, frame_num))
    os.remove('/test_data/poc_buffer/'+seq+'.yuv')
    for i in os.listdir('/test_data/poc_buffer/'):
        if int(i[:-4])%interval != 0:
            os.remove('/test_data/poc_buffer/'+i)

    os.system('cp utils/Picture0.cpp /workspace/VVCSoftware_VTM-VTM-10.0-dev/source/Lib/CommonLib/Picture.cpp')
    os.system('cd /workspace/VVCSoftware_VTM-VTM-10.0-dev/build/ && make -j')
    os.system('cd /workspace/')
    os.system('ffmpeg -s {}x{} -i {} -y -vf scale={}:{}:sws_flags=lanczos {}'.format(width, height, yuv_org, width//2, height//2, yuv_l2))
    os.system('/workspace/VVCSoftware_VTM-VTM-10.0-dev/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -fr 24 -c /workspace/VVCSoftware_VTM-VTM-10.0-dev/cfg/encoder_randomaccess_vtm.cfg --OutputBitDepth=8 --ConformanceWindowMode=1 -q {} -f {} -b {} -o {} > {}'.format(yuv_l2, width//2, height//2, tmf_qp, frame_num, generate_path+'tmf.bin', generate_path+'tmf.yuv', generate_path+'tmf.txt'))


def vvc_ldp_infi(seq, vvc_qp, frame_num, width, height):
    yuv_org = '/test_data/' + seq + '/' + seq + '.yuv'
    # vvc
    vvc_path = '/test_data/' + seq + '/vvc/LDP/qp' + str(vvc_qp) + '/'
    if not os.path.exists(vvc_path):
        os.makedirs(vvc_path)
    os.system('/workspace/VVCSoftware_VTM-VTM-10.0/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -fr 24 -c /workspace/VVCSoftware_VTM-VTM-10.0/cfg/encoder_lowdelay_P_vtm.cfg --OutputBitDepth=8 -f {} -q {} -b {} -o {} > {}'.format(yuv_org, width, height, frame_num, vvc_qp, vvc_path+seq+'.bin', vvc_path+seq+'.yuv', vvc_path+'log.txt'))


def vvc_ldp_2s(seq, vvc_qp, frame_num, width, height, interval):
    yuv_org = '/test_data/' + seq + '/' + seq + '.yuv'
    # vvc
    vvc_path = '/test_data/' + seq + '/vvc/LDP/qp' + str(vvc_qp) + '/'
    if not os.path.exists(vvc_path):
        os.makedirs(vvc_path)
    os.system('/workspace/VVCSoftware_VTM-VTM-10.0/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -fr 24 -c /workspace/VVCSoftware_VTM-VTM-10.0/cfg/encoder_lowdelay_P_vtm.cfg --OutputBitDepth=8 -f {} -q {} -b {} -o {} > {}'.format(yuv_org, width, height, frame_num, vvc_qp, vvc_path+seq+'.bin', vvc_path+seq+'.yuv', vvc_path+'log.txt'))


def vvc_ra_generate(class_num, seq, vvc_qp, frame_num, width, height):
    yuv_org = '/workspace/shared/vvc_ctc_seq/' + class_num + '/' + seq + '.yuv'
    # vvc
    vvc_path = '/workspace/shared/vvc_ctc_seq/vvc/' + class_num + '/qp' + str(vvc_qp) + '/' + seq + '/'
    if not os.path.exists(vvc_path):
        os.makedirs(vvc_path)
    os.system('/workspace/lm/CRS/codes/dataset/VVCSoftware_VTM-VTM-10.0/bin/EncoderAppStatic -i {} -wdt {} -hgt {} -fr 24 -c /workspace/lm/CRS/codes/dataset/VVCSoftware_VTM-VTM-10.0/cfg/encoder_randomaccess_vtm.cfg --OutputBitDepth=8 -f {} -q {} -b {} -o {} > {}'.format(yuv_org, width, height, frame_num, vvc_qp, vvc_path+seq+'.bin', vvc_path+seq+'.yuv', vvc_path+'log.txt'))


if __name__ == '__main__':
    vvc_ra_generate(class_num='ClassA', seq='Tango2_3840x2160_60', vvc_qp=32, frame_num=294, width=3840, height=2160)
    vvc_ra_generate(class_num='ClassA', seq='Tango2_3840x2160_60', vvc_qp=37, frame_num=294, width=3840, height=2160)
    vvc_ra_generate(class_num='ClassA', seq='Tango2_3840x2160_60', vvc_qp=42, frame_num=294, width=3840, height=2160)
    vvc_ra_generate(class_num='ClassA', seq='Tango2_3840x2160_60', vvc_qp=47, frame_num=294, width=3840, height=2160)


    # f = codecs.open('utils/test_seq_info.txt', mode='r', encoding='utf-8')
    # line = f.readline()
    # while line:
    #     seq_info = line.split()
    #     seq_name = seq_info[0]
    #     seq_width = int(seq_info[1])
    #     seq_height = int(seq_info[2])
    #     seq_frame = int(seq_info[3])
    #     print(seq_name)

    #     qp_set = [27]
    #     for seq_qp in qp_set:
    #         dcs_vvc_generate(seq=seq_name, qp=int(seq_qp), frame_num=seq_frame, width=seq_width, height=seq_height, interval=32)
        
    #     line = f.readline()
    # f.close()


    # f = codecs.open('utils/test_seq_info_E.txt', mode='r', encoding='utf-8')
    # line = f.readline()
    # while line:
    #     seq_info = line.split()
    #     seq_name = seq_info[0]
    #     seq_width = int(seq_info[1])
    #     seq_height = int(seq_info[2])
    #     seq_frame = int(seq_info[3])
    #     print(seq_name)

    #     qp_set = [42]
    #     for seq_qp in qp_set:
    #         vvc_generate(seq=seq_name, vvc_qp=seq_qp, frame_num=seq_frame, width=seq_width, height=seq_height)
        
    #     line = f.readline()
    # f.close()

    # dcs_vvc_generate(seq='vidyo4_720p_60', tmf_qp=27, frame_num=600, width=1280, height=720)
    # for qp in [52]:
    #     vvc_generate(seq='ParkScene_1920x1080_24', vvc_qp=qp, frame_num=240, width=1920, height=1080)

    # vvc_ldp_infi(seq='Cactus_1920x1080_50', vvc_qp=27, frame_num=500, width=1920, height=1080)
    # vvc_ldp_infi(seq='Cactus_1920x1080_50', vvc_qp=32, frame_num=500, width=1920, height=1080)
    # vvc_ldp_infi(seq='Cactus_1920x1080_50', vvc_qp=37, frame_num=500, width=1920, height=1080)
    # vvc_ldp_infi(seq='Cactus_1920x1080_50', vvc_qp=42, frame_num=500, width=1920, height=1080)
    # vvc_ldp_infi(seq='Cactus_1920x1080_50', vvc_qp=47, frame_num=500, width=1920, height=1080)
    # vvc_ldp_infi(seq='Cactus_1920x1080_50', vvc_qp=52, frame_num=500, width=1920, height=1080)

    # vvc_ldp_infi(seq='ParkScene_1920x1080_24', vvc_qp=27, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='ParkScene_1920x1080_24', vvc_qp=32, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='ParkScene_1920x1080_24', vvc_qp=37, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='ParkScene_1920x1080_24', vvc_qp=42, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='ParkScene_1920x1080_24', vvc_qp=47, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='ParkScene_1920x1080_24', vvc_qp=52, frame_num=240, width=1920, height=1080)

    # vvc_ldp_infi(seq='Kimono1_1920x1080_24', vvc_qp=27, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='Kimono1_1920x1080_24', vvc_qp=32, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='Kimono1_1920x1080_24', vvc_qp=37, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='Kimono1_1920x1080_24', vvc_qp=42, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='Kimono1_1920x1080_24', vvc_qp=47, frame_num=240, width=1920, height=1080)
    # vvc_ldp_infi(seq='Kimono1_1920x1080_24', vvc_qp=52, frame_num=240, width=1920, height=1080)

    # dcs_vvc_ldp_infi(seq='Kimono1_1920x1080_24', stf_qp=27, tmf_qp=27, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='Kimono1_1920x1080_24', stf_qp=32, tmf_qp=32, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='Kimono1_1920x1080_24', stf_qp=37, tmf_qp=37, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='Kimono1_1920x1080_24', stf_qp=42, tmf_qp=42, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='Kimono1_1920x1080_24', stf_qp=47, tmf_qp=47, frame_num=240, width=1920, height=1080)

    # dcs_vvc_ldp_infi(seq='ParkScene_1920x1080_24', stf_qp=27, tmf_qp=27, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='ParkScene_1920x1080_24', stf_qp=32, tmf_qp=32, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='ParkScene_1920x1080_24', stf_qp=37, tmf_qp=37, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='ParkScene_1920x1080_24', stf_qp=42, tmf_qp=42, frame_num=240, width=1920, height=1080)
    # dcs_vvc_ldp_infi(seq='ParkScene_1920x1080_24', stf_qp=47, tmf_qp=47, frame_num=240, width=1920, height=1080)

    # dcs_vvc_ldp_infi(seq='FourPeople_1280x720_60', stf_qp=27, tmf_qp=27, frame_num=600, width=1280, height=720)
    # dcs_vvc_ldp_infi(seq='FourPeople_1280x720_60', stf_qp=32, tmf_qp=32, frame_num=600, width=1280, height=720)
    # dcs_vvc_ldp_infi(seq='FourPeople_1280x720_60', stf_qp=37, tmf_qp=37, frame_num=600, width=1280, height=720)
    # dcs_vvc_ldp_infi(seq='FourPeople_1280x720_60', stf_qp=42, tmf_qp=42, frame_num=600, width=1280, height=720)
    # dcs_vvc_ldp_infi(seq='FourPeople_1280x720_60', stf_qp=47, tmf_qp=47, frame_num=600, width=1280, height=720)
    # dcs_vvc_ldp_generate(seq='Johnny_1280x720_60', stf_qp=27, tmf_qp=27, intra_num=1, inter_num=1, width=1280, height=720, interval=-1)

    # dcs_vvc_ra_generate(seq='KristenAndSara_1280x720_60', stf_qp=27, tmf_qp=22, frame_num=600, width=1280, height=720, interval=32)
    # dcs_vvc_ra_generate(seq='KristenAndSara_1280x720_60', stf_qp=37, tmf_qp=32, frame_num=600, width=1280, height=720, interval=32)
    # dcs_vvc_ra_generate(seq='KristenAndSara_1280x720_60', stf_qp=42, tmf_qp=37, frame_num=600, width=1280, height=720, interval=32)
    # dcs_vvc_ra_generate(seq='KristenAndSara_1280x720_60', stf_qp=47, tmf_qp=42, frame_num=600, width=1280, height=720, interval=32)
    # dcs_vvc_ra_generate(seq='FourPeople_1280x720_60', stf_qp=32, tmf_qp=27, frame_num=600, width=1280, height=720, interval=32)
    # dcs_vvc_ra_generate(seq='FourPeople_1280x720_60', stf_qp=37, tmf_qp=32, frame_num=600, width=1280, height=720, interval=32)








