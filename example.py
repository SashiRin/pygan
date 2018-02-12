import pygan
from pygan import augmentator
import types
import pandas as pd

opt = types.SimpleNamespace()
opt.image = False
opt.data_root = pd.read_csv('../testing/data.csv')
opt.y_label = 'Grant.Status'
opt.z_dim = 100
opt.lrD = 0.00005
opt.lrG = 0.00005
opt.gen_num = 1000
opt.gan_type = 'WGAN'
opt.epoch_num = 50
opt.path = './result'
opt.batch_norm = False
opt.input_norm = False
opt.batch_size = 128

augmentator.main(opt)

