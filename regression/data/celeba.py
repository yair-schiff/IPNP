import argparse
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from regression.utils.paths import datasets_path


class CelebA(object):
    def __init__(self, train=True, resize=0):
        requested_file = ('train' if train else 'eval') + (f'_resized_{resize}' if resize else '') + '.pt'
        print('Requested file:', osp.join(datasets_path, 'celeba', requested_file))
        if not osp.exists(osp.join(datasets_path, 'celeba', requested_file)):
            self.setup(resize=resize)
        self.data, self.targets = torch.load(osp.join(datasets_path, 'celeba', requested_file))
        self.data = self.data.float() / 255.0

        if train:
            self.data, self.targets = self.data, self.targets
        else:
            self.data, self.targets = self.data, self.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    @staticmethod
    def setup(resize=0):
        # load train/val/test split
        splitdict = {}
        with open(osp.join(datasets_path, 'celeba', 'Eval', 'list_eval_partition.txt'), 'r') as f:
            for line in f:
                fn, split = line.split()
                splitdict[fn] = int(split)

        # load identities
        iddict = {}
        with open(osp.join(datasets_path, 'celeba', 'Anno', 'identity_CelebA.txt'), 'r') as f:
            for line in f:
                fn, label = line.split()
                iddict[fn] = int(label)

        train_imgs = []
        train_labels = []
        eval_imgs = []
        eval_labels = []
        path = osp.join(datasets_path, 'celeba', 'Img', 'img_align_celeba')
        imgfilenames = os.listdir(path)
        for fn in tqdm(imgfilenames):
            # TODO: Making the resize a flag
            img = Image.open(osp.join(path, fn)).resize((resize, resize)) if resize else Image.open(osp.join(path, fn))
            if splitdict[fn] == 2:
                eval_imgs.append(torch.LongTensor(np.array(img).transpose((2, 0, 1))))
                eval_labels.append(iddict[fn])
            else:
                train_imgs.append(torch.LongTensor(np.array(img).transpose((2, 0, 1))))
                train_labels.append(iddict[fn])

        print(f'{len(train_imgs)} train, {len(eval_imgs)} eval')

        train_imgs = torch.stack(train_imgs)
        train_labels = torch.LongTensor(train_labels)
        torch.save([train_imgs, train_labels], osp.join(datasets_path, 'celeba',
                                                        'train' + (f'_resized_{resize}' if resize else '') + '.pt'))

        eval_imgs = torch.stack(eval_imgs)
        eval_labels = torch.LongTensor(eval_labels)
        torch.save([eval_imgs, eval_labels], osp.join(datasets_path, 'celeba',
                                                      'eval' + (f'_resized_{resize}' if resize else '') + '.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resize', type=int, default=64)
    args = parser.parse_args()
    print('Files will be saved to', osp.join(datasets_path, 'celeba'))
    _ = CelebA(train=True, resize=args.resize)
    _ = CelebA(train=False, resize=args.resize)
