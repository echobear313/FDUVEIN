import argparse
import tqdm
import torch.nn.functional as F
from torch import nn
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--device_ids', help='devices id to be used', type=int, nargs='*')
parser.add_argument('--fold', help='train test split', type=int, default=0)
parser.add_argument('--batch_size', help='batch_size', type=int, default=16)
parser.add_argument('--encoder', help='used backbone', type=str, default='se_resnext50_32x4d')
parser.add_argument('--weight', help='imagenet or instagram', type=str, default='imagenet')
parser.add_argument('--ce_loss_weights', type=float, nargs='*', default=[0.0, 0.0, 0.5, 1.0])
parser.add_argument('--mse_loss_weights', type=float, nargs='*', default=[1.0, 0.0, 0.25, 0.0])
parser.add_argument('--l1_loss_weights', type=float, nargs='*', default=[1.0, 0.0, 0.25, 0.0])
parser.add_argument('--grad_loss_weight', type=float, default=30.)
parser.add_argument('--gan_loss_weight', type=float, default=0.1)
parser.add_argument('--max_epoch', type=int, default=40)
parser.add_argument('--d_in_scales', type=int, nargs='*', default=[1, 4, 8])
parser.add_argument('--name', type=str, default='')
args = parser.parse_args()
print(args)
torch.cuda.set_device(args.device_ids[0])
args.batch_size = args.batch_size * len(args.device_ids)

import torch
import dataset

dataloaders = {
    'train': dataset.get_loader(args.encoder, args.weight, batch_size=args.batch_size, train=True, fold=args.fold),
    'val': dataset.get_loader(args.encoder, args.weight, batch_size=1, train=False, fold=args.fold)
}

import models

model = models.PSPNet(args.encoder, args.weight)
model = nn.DataParallel(model, device_ids=args.device_ids)
model = model.cuda()
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001,
)

discriminator = models.Discriminator(in_channels=3, in_scales=args.d_in_scales)

discriminator = models.Discriminator.turn_on_spectral_norm(discriminator)

discriminator = nn.DataParallel(discriminator, device_ids=args.device_ids)
discriminator = discriminator.cuda()
D_optimizer = torch.optim.Adam(
    discriminator.parameters(), lr=0.0001,
)

from utils import MultiscaleCounter, loss_metric

iou_counter = MultiscaleCounter()
max_score = 0

grad_loss_metric = models.GradLoss()
gan_loss_metric = models.LSGAN(discriminator)

for epoch in range(args.max_epoch):
    for phase in ['train', 'val']:
        is_train = phase == 'train'
        model.train(is_train)
        for input_s1, label_s1 in tqdm.tqdm(dataloaders[phase], desc='{} {}/{}'.format(phase, epoch, args.max_epoch)):
            input_s1, label_s1 = input_s1.cuda(), label_s1.cuda()
            mask_s1, mask_s2, mask_s4, mask_s8 = model(input_s1)
            label_s2, label_s4, label_s8 = F.interpolate(label_s1, scale_factor=1 / 2, mode='nearest'), \
                                           F.interpolate(label_s1, scale_factor=1 / 4, mode='nearest'), \
                                           F.interpolate(label_s1, scale_factor=1 / 8, mode='nearest')
            input_s2, input_s4, input_s8 = F.interpolate(input_s1, scale_factor=1 / 2, mode='bilinear', align_corners=False), \
                                           F.interpolate(input_s1, scale_factor=1 / 4, mode='bilinear', align_corners=False), \
                                           F.interpolate(input_s1, scale_factor=1 / 8, mode='bilinear', align_corners=False)

            gan_loss = 0.
            if is_train:
                real_samples = input_s1 * label_s1, input_s2 * label_s2, input_s4 * label_s4, input_s8 * label_s8
                fake_samples = input_s1 * mask_s1, input_s2 * mask_s2, input_s4 * mask_s4, input_s8 * mask_s8
                ########## train D ###################
                d_loss = gan_loss_metric.dis_loss(real_samples, list(map(lambda x: x.detach(), fake_samples)))
                D_optimizer.zero_grad()
                d_loss.backward()
                D_optimizer.step()

                ########## train G ###################
                gan_loss = gan_loss_metric.gen_loss(real_samples, fake_samples)

                # print('d_loss {:.4f}, g_loss {:.4f}'.format(d_loss.item(), gan_loss.item()))

            masks = (mask_s1, mask_s2, mask_s4, mask_s8)
            inputs = (input_s1, input_s2, input_s4, input_s8)
            labels = (label_s1, label_s2, label_s4, label_s8)
            mask_loss = loss_metric(masks, labels,
                                       ce=args.ce_loss_weights, mse=args.mse_loss_weights,
                                       l1=args.l1_loss_weights)

            grad_loss = grad_loss_metric(mask_s1, label_s1)
            loss = mask_loss + \
                   grad_loss * args.grad_loss_weight + \
                   gan_loss * args.gan_loss_weight
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            iou_counter.update(masks, labels)
        torch.cuda.empty_cache()

        print('{}-{}'.format(phase, epoch))
        print('{}'.format(str(iou_counter)))
        iou_score = iou_counter.reset()
        # do something (save model, change lr, etc.)
        if max_score < iou_score and not is_train:
            max_score = iou_score
            torch.save(model.state_dict(), 'checkpoints/best_model_{}_ours{}.pth'.format(args.fold, '_%s' % args.name))
            print('Model saved!')
