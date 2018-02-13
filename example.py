import pygan
from pygan import augmentator
import types
import pandas as pd

opt = types.SimpleNamespace()
opt.data_root = pd.read_csv('../testing/data.csv')
opt.y_label = 'Grant.Status'
opt.z_dim = 100
opt.lrD = 0.00005
opt.lrG = 0.00005
opt.gen_num = 10
opt.gan_type = 'CGAN'
opt.epoch_num = 10
opt.path = './result'
opt.batch_size = 128
opt.all_yes = True
augmentator.main(opt)

