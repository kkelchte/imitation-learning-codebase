#!/usr/bin/env python
import os
import shutil
import argparse
import sys

import torch
import time
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data
import torchvision.models as models
import torchvision.transforms as transforms

from src.ai.architectures import *  # Do not remove
from src.ai.base_net import ArchitectureConfig
from src.core.utils import get_data_dir

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    default="/esat/visicsrodata/datasets/ilsvrc2012",
                    help='path to dataset, test dir: /esat/opal/kkelchte/experimental_data/datasets/dummy_ilsvrc')
parser.add_argument('-bs', '--batch_size', default=32)
parser.add_argument('-lr', '--learning_rate', default=0.01)
parser.add_argument('-n', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-a', '--architecture',
                    default="auto_encoder_deeply_supervised",
                    help='architecture to train, make sure that architecture contains ImageNet besides the normal Net.')
parser.add_argument('-o', '--output_path',
                    default="test_imagenet_pretrain")
parser.add_argument("-rm", action='store_true', help="remove current output dir before start")


def train(train_loader, model, criterion, optimizer, epoch) -> float:
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            progress.display(i)
    return acc1


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.01 * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main():
    args = parser.parse_args()
    print(args)
    if not args.output_path.startswith('/'):
        f'{get_data_dir(os.environ["HOME"])}/{args.output_path}'

    if args.rm:
        shutil.rmtree(args.output_path, ignore_errors=True)

    #  model = models.__dict__['resnet18']()

    model = eval(args.architecture).ImageNet(ArchitectureConfig().create(config_dict={
        'architecture': '',
        'batch_normalisation': True,
        'output_path': args.output_path
    }))

    traindir = os.path.join(args.data, 'ILSVRC2012_img_train')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                momentum=0.9,
                                weight_decay=5e-4)
    best_acc1 = 0
    for epoch in range(args.epochs):
        # train for one epoch
        acc1 = train(train_loader, model, criterion, optimizer, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        torch.save({
            'epoch': epoch + 1,
            'arch': args.architecture,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(args.output_path, 'torch_checkpoints', f'model_{epoch}.ckpt'))
        if is_best:
            shutil.copyfile(os.path.join(args.output_path, 'torch_checkpoints', f'model_{epoch}.ckpt'),
                            os.path.join(args.output_path, 'torch_checkpoints', f'model_best.ckpt'))


if __name__ == "__main__":
    main()
