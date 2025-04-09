import os
import os.path

i = 0
a = range(0, 8000)
name_list = list(a)
lst_files = os.walk('D:/emotion_resnet/a/sadness')

for dirpath, dirname, filename in lst_files:
    for file in filename:
        name = 'sadness.'+str(name_list[i]) + '.jpg'
        i += 1
        print(name)
        # 选中的文件的目录加文件名
        src = os.path.join(dirpath, file)
        # 修改之后的目录加文件名
        dst = os.path.join(dirpath, name)
        os.rename(src, dst)