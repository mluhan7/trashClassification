from dataset import Garbage_Loader
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch
import shutil
from tqdm import tqdm
import time
print("训练开始！！！")
#计算时间消耗
start = time.time()
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        class_to = pred[0].cpu().numpy()
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, class_to
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     """
#         根据 is_best 存模型，一般保存 valid acc 最好的模型
#     """
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'cpu_model_best_' + filename)
def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, (input, target) in enumerate(tqdm(train_loader)):
        data_time.update(time.time() - end)
        output = model(input)
        loss = criterion(output, target)
        # measure accuracy and record loss
        [prec1, prec5], class_to = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

# def validate(val_loader, model, criterion, epoch, phase="VAL"):
#     batch_time = AverageMeter()
#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()
#     # switch to evaluate mode
#     model.eval()
#
#     with torch.no_grad():
#         end = time.time()
#         for i, (input, target) in enumerate(tqdm(val_loader)):
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
#
#             if i % 10 == 0:
#                 print('Test-{0}: [{1}/{2}]\t'
#                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
#                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
#                       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     phase, i, len(val_loader),
#                     batch_time=batch_time,
#                     loss=losses,
#                     top1=top1, top5=top5))
#
#         print(' * {} Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
#               .format(phase, top1=top1, top5=top5))
#     return top1.avg, top5.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == "__main__":
    # -------------------------------------------- step 1/4 : 加载数据 ---------------------------
    train_dir_list = './model/my_train.txt'
    batch_size = 128
    epochs = 1
    num_classes = 214
    train_data = Garbage_Loader(train_dir_list, train_flag=True)
    train_loader = DataLoader(dataset=train_data, pin_memory=True, batch_size=batch_size, shuffle=True)
    train_data_size = len(train_data)
    print('训练集数量：%d' % train_data_size)
    # ------------------------------------ step 2/4 : 定义网络 ------------------------------------
    model = models.resnet50(pretrained=True)
    fc_inputs = model.fc.in_features
    model.fc = nn.Linear(fc_inputs, num_classes)
    # ------------------------------------ step 3/4 : 定义损失函数和优化器等 -------------------------
    lr_init = 0.001
    lr_stepsize = 20
    weight_decay = 0.001
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_stepsize, gamma=0.1)
    # ------------------------------------ step 4/4 : 训练 -----------------------------------------
    # best_prec1 = 0
    for epoch in range(epochs):
        scheduler.step()
        train(train_loader, model, criterion, optimizer, epoch)
        # 在验证集上测试效果
        # valid_prec1, valid_prec5 = validate(valid_loader, model, criterion, epoch, phase="VAL")
        # is_best = valid_prec1 > best_prec1
        # best_prec1 = max(valid_prec1, best_prec1)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': 'resnet50',
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best,
        #     filename='cpu_checkpoint_resnet50.pth.tar')
    end = time.time()
    min = (end - start) / 60
    hour = min / 60
    print("训练结束！！！")
    print("训练时间:%.2f分钟" % min)
    print("训练时间:%.2f小时" % hour)