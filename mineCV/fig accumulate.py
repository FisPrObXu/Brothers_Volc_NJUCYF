import copy
import os, cv2
import numpy as np
# 通道索引 channel_list = [1, 2, 3]
def split_channel_img(root_dir, channel_list):
    '''
    得到一个标准扫查切面中一个通道的所有图像
    :param root_dir: 一个切面图像的目录
	:param channel: 通道列表
    :return:一个扫查切面中所有通道的图像信息，以字典的形式保存{通道数：图片名}
    '''
    img_channel_info = {}
    img_names = os.listdir(root_dir)

    img_names = [n for n in img_names if n.endswith('.png')]
	# 筛选出通道为channel的图像
    for c in channel_list:
        img_names_channel = [n for n in img_names if int(n.split('-')[1])==c]
        # print(img_names_channel)
        # 对文件名排序
        img_names_channel.sort(key=lambda x: int(x[:-4].split('-')[2]))
        img_channel_info[c] = img_names_channel
    return img_channel_info
	
def get_channel_img(img_channel_info, num):
    '''

    :param img_channel_info: 扫查三道所有图像
    :param num: 每道图片数
    :return: 三道拼接后的图像张量
    '''
    sec = []
    im3d = np.zeros(shape=(num, 480, 556), dtype='uint8')
    for channel, filelist in img_channel_info.items():
        print(channel, filelist)
        count = 0
        for file in filelist:
            filepath = os.path.join(section_dir, file)

            im2d = cv2.imread(filepath)
            im3d[count] = im2d[:, :, 0]# 三个通道的值一样，只取一个通道
            count += 1
        im3d_t = im3d.transpose(2, 1, 0)
        sec.append(im3d_t.copy())# 注意内存地址变化
    res = np.concatenate(sec).transpose(2, 1, 0)
    return res

if __name__=='__main__':
    section_dir = 'D:\A1_BREAST_DATA\LLAT'
    channel_list = [1, 2, 3]
    img_channel_info = split_channel_img(section_dir, channel_list)
    num = len(img_channel_info[1])
    im_sec_3d = get_channel_img(img_channel_info, num)
    # for i in range(im_sec_3d.shape[0]):
    #     cv2.imwrite('./save/{}.jpg'.format(i), im_sec_3d[i, :, :])
    cv2.imshow('img', im_sec_3d[30, :, :])
    cv2.waitKey(0)
