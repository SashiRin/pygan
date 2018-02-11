from __future__ import print_function
import argparse
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch

if __name__ == '__main__':
	from models.models import model
else:
	from .models.models import model


def main(opt):
    # testing
    if opt.image:
        data_loader = torch.utils.data.DataLoader(
            dset.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                               transforms.Lambda(lambda x: x.view(-1))
                           ])),
            batch_size=opt.batch_size, shuffle=False)
    # testing
    else:
        data_loader = load_data(opt)

    # testing
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
        class_num=class_num,
        clamp_lower=opt.clamp_lower,
        clamp_upper=opt.clamp_upper,
        image=opt.image)
    GAN.train(opt.epoch_num)
    gen_data = GAN.generate(opt.gen_num)
    gen_data = data_loader.dataset.dataDestandarize(gen_data)
    data_loader.dataset.dataRound(gen_data)
    GAN.save('{}/generator_weight'.format(opt.path), '{}/discriminator_weight'.format(opt.path))
    gen_data.to_csv('{}/gen_data.csv'.format(opt.path), index=False)
    return GAN, gen_data

def load_data(opt):
    if opt.data_type == 'FRAME':
        dataset = DataFrameDataset(opt.data_root, opt.y_label)
    elif opt.data_type == 'SET':
        dataset = dset.ImageFolder(root=opt.data_root,
                                   transform=transforms.Compose([
                                       transforms.Scale(opt.image_size),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))
    else:
        # error handling
        dataset = None
    # dataset.dataNorm()
    dataset.dataStandardize()
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan_type', default='GAN', help=' GAN | CGAN | WGAN | WCGAN ')
    parser.add_argument('--gen_num', type=int, default=64)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--data_type', help=' FRAME | SET ')
    parser.add_argument('--y_label', default=None, help='input Y label')
    parser.add_argument('--z_dim', type=int, default=100, help='input Z dimension')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epoch_num', type=int, default=100, help='input epoch number')
    parser.add_argument('--lrD', type=float, default=0.00005)
    parser.add_argument('--lrG', type=float, default=0.00005)
    parser.add_argument('--input_norm', action='store_true')
    parser.add_argument('--batch_norm', action='store_true')
    parser.add_argument('--image', action='store_true')
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--path', type=str, default='./result')
    opt = parser.parse_args()

    main(opt)
