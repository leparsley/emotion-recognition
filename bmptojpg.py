# _*_coding:utf-8
import os

pic_path = "D:/emotion/new_fear"

i = 1
a = range(0, 2000)
name_list = list(a)

def rename():
    piclist = os.listdir(pic_path)
    total_num = len(piclist)
    for pic in piclist:
        if pic.endswith(".bmp"):
            old_path = os.path.join(os.path.abspath(pic_path), pic)
            new_path = os.path.join(os.path.abspath(pic_path), 'fear_' + str(name_list[i]) + '.jpg')
            os.renames(old_path, new_path)
            i = i + 1



rename()