import os
def file_name(file_dir):
    list_Name = []
    for dir in os.listdir(file_dir):
        list_Name.append(dir)
    return list_Name
fileNames = open('E:/项目文件/python/机器学习/trash/model/dir_label.txt', 'w')
file_dir = 'E:/项目文件/python/机器学习/trash/垃圾图片库'     # 待取名称文件夹的绝对路径
list_name = file_name(file_dir)   # listname 就是需要的标签样本
print(list_name)
count = 0
for i in list_name:
    # 找到同类的标签
    flag = str(i).split('_', 1)[0]
    if(flag == "其他垃圾"):
        flag1 = 0
        data = str(i) + '\t' + str(count) + '\t' + str(flag1) + '\n'
        count = count + 1
        fileNames.write(data)  # 开始写入文件
    elif(flag == "厨余垃圾"):
        flag2 = 1
        data = str(i) + '\t' + str(count) + '\t' + str(flag2) + '\n'
        count = count + 1
        fileNames.write(data)  # 开始写入文件
    elif (flag == "可回收物"):
        flag3 = 2
        data = str(i) + '\t' + str(count) + '\t' + str(flag3) + '\n'
        count = count + 1
        fileNames.write(data)  # 开始写入文件
    elif (flag == "有害垃圾"):
        flag4 = 3
        data = str(i) + '\t' + str(count) + '\t' + str(flag4) + '\n'
        count = count + 1
        fileNames.write(data)  # 开始写入文件
fileNames.close()  # 最后关掉文件

