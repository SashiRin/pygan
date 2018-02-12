from __future__ import print_function
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

from . import *

if __name__ == '__main__':
    from models.models import model
else:
    from .models.models import model


def main(opt):
    if opt.image:
        data_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                               transforms.Lambda(lambda x: x.view(-1))
                           ])),
            batch_size=opt.batch_size, shuffle=False)
    else:
        data_loader = load_data(opt)

    if opt.image:
        z_dim, x_dim, class_num = 100, 784, 10
    else:
        z_dim, x_dim, class_num = opt.z_dim, data_loader.dataset.x_dim, data_loader.dataset.class_num
    GAN = model[opt.gan_type](
        train_loader=data_loader,
        batch_size=opt.batch_size,
        x_size=x_dim,
        z_size=z_dim,
        y_size=1,
        lrG=opt.lrG,
        lrD=opt.lrD,
        epoch_num=opt.epoch_num,
        class_num=class_num)
    GAN.train(opt.epoch_num)
    gen_data = pd.DataFrame(GAN.generate(opt.gen_num), columns=opt.data_root.columns)
    gen_data = data_loader.dataset.destandardizeDataFrame(gen_data)
    gen_data = data_loader.dataset.dataRound(gen_data)
    GAN.save('{}/generator_weight'.format(opt.path), '{}/discriminator_weight'.format(opt.path))
    gen_data.to_csv('{}/gen_data.csv'.format(opt.path), index=False)
    return GAN, gen_data

def load_data(opt):
    dataset = DataFrameDataset(opt.data_root, opt.y_label)
    dataset.standardizeDataFrame()
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_type', default='GAN', help=' GAN | CGAN | WGAN | WCGAN ')
    parser.add_argument('--gen_num', type=int, default=64)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--y_label', default=None, help='input Y label')
    parser.add_argument('--z_dim', type=int, default=100, help='input Z dimension')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch_num', type=int, default=100, help='input epoch number')
    parser.add_argument('--lrD', type=float, default=0.00005)
    parser.add_argument('--lrG', type=float, default=0.00005)
    parser.add_argument('--input_norm', action='store_true')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--path', type=str, default='./result')
    opt = parser.parse_args()

    main(opt)
