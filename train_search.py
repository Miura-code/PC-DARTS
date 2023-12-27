""" Architecture Searchを行い、最適なセル構造を探索する """

import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from torch.autograd import Variable
from model_search import Network
from architect import Architect

import utils


def arguments():
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument('--data', type=str, default='/home/miura/lab/data', help='location of the data corpus')
  parser.add_argument('--set', type=str, default='cifar10', help='location of the data corpus')
  parser.add_argument('--batch_size', type=int, default=256, help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
  parser.add_argument('--learning_rate_min', type=float, default=0.0, help='min learning rate')
  parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
  parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
  parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
  parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
  parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
  parser.add_argument('--layers', type=int, default=8, help='total number of layers')
  parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
  parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
  parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
  parser.add_argument('--save', type=str, default='EXP', help='experiment name')
  parser.add_argument('--seed', type=int, default=2, help='random seed')
  parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
  parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
  parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
  parser.add_argument('--arch_learning_rate', type=float, default=6e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
  args = parser.parse_args()

  args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
  # utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  utils.create_exp_dir(args.save, scripts_to_save=None)

  print("{0:-<15} args {0:-<15}\n".format(""))
  for arg in vars(args):
      print("{:>15}->{}".format(arg, getattr(args, arg)))
    
  return args

def main():
  args = arguments()
  
  CIFAR_CLASSES = 10
  if args.set=='cifar100':
    CIFAR_CLASSES = 100

  utils.log_setting(args.save)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  utils.set_seed_gpu(args.seed, args.gpu)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  if args.set=='cifar100':
      train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
  else:
      train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  # 訓練用と検証用にデータを分割
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)

  # 学習率のスケジューラを設定
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  # Architecture Search
  best_acc_top1 = 0.0
  best_acc_top5 = 0.0
  for epoch in tqdm(range(args.epochs)):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    logging.info('genotype = %s', genotype)

    # training
    train_acc, train_obj = train(args, train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    if args.epochs-epoch<=1:
      valid_acc_top1, valid_acc_top5, valid_obj = infer(args, valid_queue, model, criterion)
      is_best = False
      if valid_acc_top5 > best_acc_top5:
          best_acc_top5 = valid_acc_top5
      if valid_acc_top1 > best_acc_top1:
          best_acc_top1 = valid_acc_top1
          is_best = True
      logging.info('valid_acc_top1 %f, valid_acc_top5 %f, best_acc_top1 %f, best_acc_top1 %f', valid_acc_top1, valid_acc_top5, best_acc_top1, best_acc_top5)

    # モデルを保存
    utils.save(model, os.path.join(args.save, 'weights.pt'))
    utils.save_genotype(genotype, os.path.join(args.save, 'genotype.pt'))
    logging.info('model saved at %s' % args.save)

def train(args, train_queue, valid_queue, model, architect, criterion, optimizer, lr,epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)
    input = Variable(input, requires_grad=False).cuda()
    target = Variable(target, requires_grad=False).cuda(non_blocking=True)

    # get a random minibatch from the search queue with replacement
    # input_search, target_search = next(iter(valid_queue))
    try:
     input_search, target_search = next(valid_queue_iter)
    except:
     valid_queue_iter = iter(valid_queue)
     input_search, target_search = next(valid_queue_iter)
    input_search = Variable(input_search, requires_grad=False).cuda()
    target_search = Variable(target_search, requires_grad=False).cuda(non_blocking=True)

    # 15epoch以降にArchitecture parameterを最適化（Warm-up）
    if epoch>=15:
      architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)
    
    # Networkの重みを最適化
    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03dsteps %eobjs.avg %ftop1.avg %ftop5.avg', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg

def infer(args, valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    #input = input.cuda()
    #target = target.cuda(non_blocking=True)
    input = Variable(input, volatile=True).cuda()
    target = Variable(target, volatile=True).cuda(non_blocking=True)
    with torch.no_grad():
      logits = model(input)
      loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03dsteps %eobjs.avg %ftop1.avg %ftop5.avg', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg
  
if __name__ == '__main__':
  main() 

