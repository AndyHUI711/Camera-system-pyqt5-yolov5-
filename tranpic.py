'''

将指定源文件夹内的所有指定类型的图片,
转为指定格式的文件,
再另存到指定的目标文件夹,
不该变源文件夹下的文件内容.

'''

source_folder = "C:/Users/xuzha/Downloads/testtube"
destination_folder = 'C:/Users/xuzha/Downloads/testtube-yolo'
source_format = ['jpg', 'bmp', 'png']
destination_format = 'jpg'

import os
import time
from PIL import Image


def pictures2jpg():
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
        print("目录: " + destination_folder + " 创建成功！")

    files = os.listdir(source_folder)  # 返回一个列表，每个元素是一个字符串，字符串的内容是文件名包括扩展名
    totalNum = len(files)
    cnt = 1
    scale = 50
    # start = time.perf_counter()
    start = time.time()
    for file in files:
        if file.split(".")[-1] in source_format:
            sourcePic = Image.open(source_folder + '/' + file)
            sourcePic.save(destination_folder + '/' + (os.path.splitext(file)[0]) + '.' + destination_format)

        a = '*' * int(cnt / totalNum * scale)
        b = '.' * (scale - int(cnt / totalNum * scale))
        c = (cnt / totalNum) * 100
        # t = time.process_time() - start
        t = time.time() - start
        printInfo = "\r{:^6.1f}%[{}->{}]{:.2f}s".format(c, a, b, t)
        print(printInfo, end="")
        cnt += 1


if __name__ == "__main__":
    pictures2jpg()


