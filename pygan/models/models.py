from .gan import GAN
from .cgan import CGAN
from .wgan import WGAN
from .wcgan import WCGAN

model = {
    'GAN': GAN,
    'CGAN': CGAN,
    'WGAN': WGAN,
    'WCGAN': WCGAN
}
