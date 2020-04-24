"""
@author: Signatrix GmbH
Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)
"""
import argparse
import os
import shutil
import time
from pthflops import count_ops
from torchsummary import summary

import torch
import torch.nn as nn
import torch.nn.parallel
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.utils.tensorboard import SummaryWriter

from src.regnet import RegNetY
from src.dataset import Imagenet
from src.config import TRAIN_IMAGE_SIZE


def get_args():
    parser = argparse.ArgumentParser(
        description="Implementation of paradigm described in paper: Designing Network Design Spaces published by Facebook AI Research (FAIR)")

    parser.add_argument("-d", "--data_path", type=str, default="data", help="the root folder of dataset")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="number of total epochs to run")
    parser.add_argument("-b", "--batch_size", default=32, type=int)
    parser.add_argument("-l", "--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("-m", "--momentum", default=0.9, type=float, help="momentum")
    parser.add_argument("-w", "--weight_decay", default=5e-4, type=float, help="weight decay")
    parser.add_argument("--log_path", type=str, default="tensorboard/signatrix_regnet_imagenet")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    # These default parameters are for RegnetY 200MF
    parser.add_argument("--bottleneck_ratio", default=1, type=int)
    parser.add_argument("--group_width", default=8, type=int)
    parser.add_argument("--initial_width", default=24, type=int)
    parser.add_argument("--slope", default=36, type=float)
    parser.add_argument("--quantized_param", default=2.5, type=float)
    parser.add_argument("--network_depth", default=13, type=int)
    parser.add_argument("--stride", default=2, type=int)
    parser.add_argument("--se_ratio", default=4, type=int)

    args = parser.parse_args()
    return args

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main(opt):
    num_gpus = 1
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    training_params = {"batch_size": opt.batch_size * num_gpus,
                       "shuffle": True,
                       "drop_last": True,
                       "num_workers": 12}

    test_params = {"batch_size": opt.batch_size//10,
                   "shuffle": False,
                   "drop_last": False,
                   "num_workers": 12}

    training_set = Imagenet(root_dir=opt.data_path, mode="train")
    training_generator = DataLoader(training_set, **training_params)

    test_set = Imagenet(root_dir=opt.data_path, mode="val")
    test_generator = DataLoader(test_set, **test_params)

    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)

    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)

    writer = SummaryWriter(opt.log_path)
    model = RegNetY(opt.initial_width, opt.slope, opt.quantized_param, opt.network_depth, opt.bottleneck_ratio,
                    opt.group_width, opt.stride, opt.se_ratio)

    dummy_input = torch.randn((1, 3, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE))
    writer.add_graph(model, dummy_input)
    # Calculate model FLOPS and number of parameters
    count_ops(model, dummy_input, verbose=False)
    summary(model, (3, TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE), device="cpu")

    if torch.cuda.is_available():
        model = nn.DataParallel(model)
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay, nesterov=True)
    best_acc1 = 0
    model.train()

    for epoch in range(opt.epochs):
        adjust_learning_rate(optimizer, epoch, opt.lr)
        train(training_generator, model, criterion, optimizer, epoch, writer)
        acc1 = validate(test_generator, model, criterion, epoch, writer)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "optimizer": optimizer.state_dict(),
        }, is_best, opt.saved_path)


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    model.train()
    end = time.time()
    num_iter_per_epoch = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.detach().item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        writer.add_scalar('Train/Loss', losses.avg, epoch * num_iter_per_epoch + i)
        writer.add_scalar('Train/Top1_acc', top1.avg, epoch * num_iter_per_epoch + i)
        writer.add_scalar('Train/Top5_acc', top5.avg, epoch * num_iter_per_epoch + i)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix="Test: ")

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                progress.display(i)

        print(" * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}"
              .format(top1=top1, top5=top5))
        writer.add_scalar('Test/Loss', losses.avg, epoch)
        writer.add_scalar('Test/Top1_acc', top1.avg, epoch)
        writer.add_scalar('Test/Top5_acc', top5.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, saved_path, filename="checkpoint.pth.tar"):
    file_path = os.path.join(saved_path, filename)
    torch.save(state, file_path)
    if is_best:
        shutil.copyfile(file_path, os.path.join(saved_path, "best_checkpoint.pth.tar"))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == "__main__":
    opt = get_args()
    main(opt)
