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
    """
    데이터를 주어진 옵션에 따라 자동으로 생성합니다.
    옵션은 types.SimpleNamespace 를 통해 정의할 수 있습니다.

        >>> import types
        >>> opt = types.SimpleNamespace()
        >>> opt.data_root = pd.read_csv('../data/train_set.csv')
        >>> opt.y_label = 'Grant.Status'
        >>> opt.z_dim = 100
        >>> opt.lrD = 0.00005
        >>> opt.lrG = 0.00005
        >>> opt.gen_num = 10
        >>> opt.gan_type = 'CGAN'
        >>> opt.epoch_num = 10
        >>> opt.path = './result'
        >>> opt.batch_size = 128

    :param opt:
        * opt.data_root: 사용할 데이터 프레임
        * opt.y_label: 학습의 정답으로 사용할 레이블의 이름
        * opt.z_dim: Generator의 입력으로 사용될 Noise의 차원
        * opt.lrD, opt.lrG: Discriminator, Generator Learning Rate
        * opt.gen_num: 생성할 데이터 수
        * opt.gan_type: 사용할 GAN의 종류 ( GAN | CGAN | WGAN | WCGAN )
        * opt.epoch_num: 학습의 epoch 수
        * opt.path: 학습된 Generator와 Discriminator의 Weight과 생성된 데이터가 저장될 위치
        * opt.batch_size: 학습에 사용될 batch 크기
    :return:
        * (GAN, gen_data)
        * GAN: 학습된 GAN, GAN.G 와 같이 학습된 Generator를 사용할 수 있다.
        * gen_data: 생성된 데이터, 데이터 프레임 타입.
    """
    # if opt.image:
    #     data_loader = torch.utils.data.DataLoader(
    #         datasets.MNIST('../data', train=True, download=True,
    #                        transform=transforms.Compose([
    #                            transforms.ToTensor(),
    #                            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    #                            transforms.Lambda(lambda x: x.view(-1))
    #                        ])),
    #         batch_size=opt.batch_size, shuffle=False)
    # else:
    #     data_loader = load_data(opt)
    data_loader = load_data(opt)
    # if opt.image:
    #     z_dim, x_dim, class_num = 100, 784, 10
    # else:
    #     z_dim, x_dim, class_num = opt.z_dim, data_loader.dataset.x_dim, data_loader.dataset.class_num
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
    gen_data = GAN.generate(opt.gen_num)
    if opt.gan_type == 'GAN' or opt.gan_type == 'WGAN':
        gen_data = pd.DataFrame(gen_data, columns=opt.data_root.columns.drop(opt.y_label))
    elif opt.gan_type == 'CGAN' or opt.gan_type == 'WCGAN':
        gen_data = pd.DataFrame(gen_data, columns=opt.data_root.columns)
    gen_data = data_loader.dataset.destandardizeDataFrame(gen_data)
    gen_data = data_loader.dataset.dataRound(gen_data)
    GAN.save('{}/generator_weight'.format(opt.path), '{}/discriminator_weight'.format(opt.path))
    gen_data.to_csv('{}/gen_data.csv'.format(opt.path), index=False)
    return GAN, gen_data

def load_data(opt):
    opt.dataframe = opt.data_root
    dataset = DataFrameDataset(**(opt.__dict__))
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
