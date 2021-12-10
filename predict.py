from dataset import Garbage_Loader
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models
import torch.nn as nn
import torch
import os
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# %matplotlib inline
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
# def accuracy(output, target, topk=(1,)):
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#         class_to = pred[0].cpu().numpy()
#         res = []
#         for k in topk:
#             correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res, class_to
#
# def Test(val_loader, model, criterion, epoch, phase="TEST"):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     # switch to evaluate mode
#     model.eval()
#     with torch.no_grad():
#         end = time.time()
#         for i, (input, target) in enumerate(tqdm(test_loader)):
#             input = input.cuda()
#             target = target.cuda()
#             output = model(input)
#             loss = criterion(output, target)
#             # measure accuracy and record loss
#             [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
#             losses.update(loss.item(), input.size(0))
#             top1.update(prec1[0], input.size(0))
#             top5.update(prec5[0], input.size(0))
#
#             # measure elapsed time
#             batch_time.update(time.time() - end)
#             end = time.time()
#             if i % 10 == 0:
#                 print('Test-{0}: [{1}/{2}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
#                     phase, i, len(val_loader),
#                     batch_time=batch_time,
#                     loss=losses,
#                     top1=top1))
#
#         print(' * {} Prec@1 {top1.avg:.3f}'
#               .format(phase, top1=top1))
#
# if __name__ == "__main__":
#     lr_init = 0.001
#     lr_stepsize = 20
#     weight_decay = 0.001
#     test_list = 'model/test.txt'
#     test_data = Garbage_Loader(test_list, train_flag=False)
#     test_loader = DataLoader(dataset=test_data, num_workers=8, pin_memory=True, batch_size=1)
#     model = models.resnet50(pretrained=False)
#     fc_inputs = model.fc.in_features
#     model.fc = nn.Linear(fc_inputs, 214)
#     model = model.cuda()
#     # 加载训练好的模型
#     checkpoint = torch.load('gpu_model_best_checkpoint_resnet50.pth.tar')
#     model.load_state_dict(checkpoint['state_dict'])
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
#     for epoch in range(1):
#         scheduler.step()
#         Test(test_loader, model, criterion, epoch, phase="TEST")


def softmax(x):
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, 0)
    return softmax_x


with open('model/dir_label.txt', 'r', encoding='utf-8') as f:
    labels = f.readlines()
    labels = list(map(lambda x: x.strip().split('\t'), labels))
    print(labels) #['其他垃圾_PE塑料袋', '0', '0']

if __name__ == "__main__":
    test_list = 'model/test.txt'
    test_data = Garbage_Loader(test_list, train_flag=False)
    test_loader = DataLoader(dataset=test_data, num_workers=1, pin_memory=True, batch_size=1)
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, 214)
    model = model.cuda()
    # 加载训练好的模型
    checkpoint = torch.load('gpu_model_best_checkpoint_resnet50.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    for i, (image, label) in enumerate(test_loader):
        src = image.numpy()
        src = src.reshape(3, 224, 224)
        src = np.transpose(src, (1, 2, 0))
        image = image.cuda()
        label = label.cuda()
        pred = model(image)
        pred = pred.data.cpu().numpy()[0]
        score = softmax(pred)
        pred_id = np.argmax(score)
        if(pred_id==203):
            plt.imshow(src)
            print('预测结果：', labels[pred_id][0])
            plt.show()

