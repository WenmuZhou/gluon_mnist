# -*- coding: utf-8 -*-
# @Time    : 2018/8/23 11:35
# @Author  : zhoujun
import os
from tqdm import tqdm
import cv2
from typing import List


def get_file_list(folder_path: str, p_postfix: List[str] = ['.jpg'], sub_dir: bool = True) -> list:
    """
    获取所给文件目录里的指定后缀的文件,读取文件列表目前使用的是 os.walk 和 os.listdir ，这两个目前比 pathlib 快很多
    :param filder_path: 文件夹名称
    :param p_postfix: 文件后缀,如果为 [.*]将返回全部文件
    :param sub_dir: 是否搜索子文件夹
    :return: 获取到的指定类型的文件列表
    """
    assert os.path.exists(folder_path) and os.path.isdir(folder_path)
    file_list = []
    if sub_dir:
        for rootdir, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(rootdir, file)
                if os.path.splitext(file_path)[-1] in p_postfix or '.*' in p_postfix:
                    file_list.append(file_path)
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if (os.path.splitext(file_path)[-1] in p_postfix or '.*' in p_postfix) and os.path.isfile(file_path):
                file_list.append(file_path)
    return file_list


def get_label(dir_path, save_path, thre_value=0.9):
    '''
    根据dir_path目录下的图片来生生成标签文本
    :param dir_path: 图片所在目录
    :param save_path: 标签文本保存位置
    :param thre_value: 每一类中多少作为训练集
    :return:　NONE
    '''
    train_file = open(save_path, 'w')
    dirs = os.listdir(dir_path)
    print(dirs)
    for cur_dir in dirs:
        cur_dir1 = os.path.join(dir_path, cur_dir)
        files = os.listdir(cur_dir1)
        for i in range(len(files)):
            train_file.write(cur_dir1 + '/' + files[i] + ' ' + cur_dir + '\n')
    train_file.close()


if __name__ == '__main__':
    dir_path = '/data1/zj/data/mnist/test/'
    save_path = '/data1/zj/data/mnist/test.txt'
    get_label(dir_path=dir_path, save_path=save_path)

# img_list = get_file_list('/data1/zj/data/mnist/train', ['.bmp'])
# pbar = tqdm(total=len(img_list))
# for img_path in img_list:
#     pbar.update(1)
#     save_path = img_path[:-4] + '.jpg'
#     if os.path.exists(save_path):
#         os.remove(img_path)
#         continue
#     img = cv2.imread(img_path, 1)
#     cv2.imwrite(save_path, img)
#     os.remove(img_path)
# pbar.close()
