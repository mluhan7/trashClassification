import os
import random
'''
将data文件中的数据集打包成标签（分为一级，二级，三级目录
'''
def creat_filelist(input_path, classes):
    # 创建三级目录
    dir_image1 = []  # 二级目录
    file_train = []
    file_test = []
    # file_valid = []
    for index, name in enumerate(classes):
        file_list = []  # 三级目录
        index_str = str(index)
        dir_image1_temp = input_path + '/' + name
        for dir2 in os.listdir(dir_image1_temp):  # 此处是遍历第三级目录,也就是将其中一个文件夹的图片信息传入
            dir_image2_temp = dir_image1_temp + '/' + dir2 + '\t' + index_str
            file_list.append(dir_image2_temp)
        file_list = random.sample(file_list, len(file_list))
        x, y = divide_filelist(file_list)
        file_train = file_train + x
        file_test = file_test + y
        # file_valid = file_valid + z
    return dir_image1, file_train, file_test
def creat_txtfile(output_path, file_list):
    with open(output_path, 'w', encoding='utf-8') as f:
        for list in file_list:
            print(list)
            f.write(str(list) + '\n')
def make_list():
    dir_image0 = '垃圾图片库'
    dir_image1 = os.listdir(dir_image0)
    classes = dir_image1
    dir_list, file_train, file_test = creat_filelist(dir_image0, classes)
    output_path_train = 'E:/项目文件/python/机器学习/trash/model/my_train.txt'
    output_path_test = 'E:/项目文件/python/机器学习/trash/model/my_test.txt'
    # 再重新一次打乱
    file_train = random.sample(file_train, len(file_train))
    file_test = random.sample(file_test, len(file_test))
    # file_valid = random.sample(file_valid, len(file_valid))
    # 写入txt文件中
    creat_txtfile(output_path_train, file_train)
    creat_txtfile(output_path_test, file_test)


    # creat_txtfile(output_path_valid, file_valid)
'''
将保存在txt中的数据集划分为训练测试验证集
'''
def divide_filelist(file_list):
    file_train = []
    file_test = []
    # file_valid = []
    for dir in file_list:  # 此处自己修改参数
        if len(file_train) < len(file_list) * 0.9:
            file_train.append(dir)
        elif len(file_test) < len(file_list) * 0.1:
            file_test.append(dir)
        # else:
        #     file_valid.append(dir)
    return file_train, file_test
if __name__ == '__main__':
    make_list()