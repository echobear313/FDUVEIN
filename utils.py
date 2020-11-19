import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from hausdorff import hausdorff_distance
import copy


def load_network(state_dict):
    if isinstance(state_dict, str):
        state_dict = torch.load(state_dict, map_location='cpu')
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == 'module.':
            k = k[7:]
        new_state_dict[k] = v
    return new_state_dict


def loss_metric(masks, labels, ce, mse, l1):
    # s1, s2, s4, s8
    mask_loss = ce[0] * F.binary_cross_entropy(masks[0], labels[0]) + \
                mse[0] * F.mse_loss(masks[0], labels[0]) + \
                l1[0] * F.l1_loss(masks[0], labels[0])
    for i in range(1, 4):
        mask_loss += ce[i] * F.binary_cross_entropy(masks[i], labels[i]) + \
                    mse[i] * F.mse_loss(masks[i], labels[i]) + \
                    l1[i] * F.l1_loss(masks[i], labels[i])
    return mask_loss

class SinglescaleCounter(object):
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold = threshold
        self.eps = eps
        self.counter = 0
        self.score = {}

    def compute_score(self, pr, gt):
        # only for binary
        pr = pr > self.threshold
        tp = torch.sum(pr * gt, dim=[1, 2, 3])
        tn = torch.sum((pr == 0) * (gt == 0), dim=[1, 2, 3])
        fp = torch.sum((pr == 1) * (gt == 0), dim=[1, 2, 3])
        fn = torch.sum((pr == 0) * (gt == 1), dim=[1, 2, 3])
        iou_score = torch.mean((tp + self.eps) / (fp + tp + fn + self.eps))
        dice_score = torch.mean((2 * tp + self.eps) / (fp + 2 * tp + fn + self.eps))
        sensitivity = torch.mean((tp + self.eps) / (tp + fn + self.eps))
        specificity = torch.mean((tn + self.eps) / (tn + fp + self.eps))

        pr = pr.float().cpu().numpy()
        gt = gt.float().cpu().numpy()
        hausdorff_score = 0.
        for i in range(pr.shape[0]):
            hausdorff_score += hausdorff_distance(pr[i, 0], gt[i, 0], distance="euclidean")
        return {
            'iou': iou_score,
            'dice': dice_score,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'hausdorff': (hausdorff_score * 0.95) / pr.shape[0]
        }

    def update(self, input, target):
        with torch.no_grad():
            scores = self.compute_score(input, target)
            for k in scores:
                if k not in self.score:
                    self.score[k] = 0.
                self.score[k] += scores[k]
            self.counter += 1

    def reset(self):
        score = copy.deepcopy(self.score)
        for k in self.score:
            self.score[k] = 0.
            score[k] /= self.counter
        self.counter = 0
        return score

class MultiscaleCounter(object):
    def __init__(self, threshold=0.5, eps=1e-7):
        self.threshold = threshold
        self.eps = eps
        self.scale = [1, 2, 4, 8]
        self.counter = 0
        self.score = {}
        for scale in self.scale:
            self.score[scale] = {}

    def compute_score(self, pr, gt):
        # only for binary
        pr = pr > self.threshold
        tp = torch.sum(pr * gt, dim=[1, 2, 3])
        tn = torch.sum((pr == 0) * (gt == 0), dim=[1, 2, 3])
        fp = torch.sum((pr == 1) * (gt == 0), dim=[1, 2, 3])
        fn = torch.sum((pr == 0) * (gt == 1), dim=[1, 2, 3])
        iou_score = torch.mean((tp + self.eps) / (fp + tp + fn + self.eps))
        dice_score = torch.mean((2 * tp + self.eps) / (fp + 2 * tp + fn + self.eps))
        sensitivity = torch.mean((tp + self.eps) / (tp + fn + self.eps))
        specificity = torch.mean((tn + self.eps) / (tn + fp + self.eps))

        pr = pr.float().cpu().numpy()
        gt = gt.float().cpu().numpy()
        hausdorff_score = 0.
        for i in range(pr.shape[0]):
            hausdorff_score += hausdorff_distance(pr[i, 0], gt[i, 0], distance="euclidean")
        return {
            'iou': iou_score,
            'dice': dice_score,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'hausdorff': (hausdorff_score * 0.95) / pr.shape[0]
        }

    def update(self, input, target):
        with torch.no_grad():
            for i, scale in enumerate(self.scale):
                scores = self.compute_score(input[i], target[i])
                for k in scores:
                    if k not in self.score[scale]:
                        self.score[scale][k] = 0.
                    self.score[scale][k] += scores[k]
            self.counter += 1

    def __str__(self):
        output_str = ''
        for scale in self.scale:
            output_str += 's{}:'.format(scale)
            for k in self.score[scale]:
                output_str += ' {}: {:4f}'.format(k, self.score[scale][k] / self.counter)
            output_str += '\n'
        return output_str

    def reset(self):
        iou_score = self.score[1]['iou'] / self.counter
        for scale in self.scale:
            for k in self.score[scale]:
                self.score[scale][k] = 0.
            self.counter = 0
        return iou_score


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()
