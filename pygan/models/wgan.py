from .gan import GAN
from . import *
from .utils import *
from .generators import *
from .discriminators import *

class WGAN(GAN):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.train_loader = kwargs['train_loader']
        self.G = BaseGenerator(**kwargs)
        self.D = BaseDiscriminator(**kwargs)
        self.z_size = kwargs['z_size']
        self.class_num = kwargs['class_num']
        self.fixed_z = torch.rand(10, self.z_size)

        if tcuda.is_available():
            self.G, self.D = self.G.cuda(), self.D.cuda()

        self.BCE_loss = nn.BCELoss()

        self.G_optimizer = optim.Adam(self.G.parameters(), lr=kwargs['lrG'], betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=kwargs['lrD'], betas=(0.5, 0.999))


    def train(self, epoch_num=10):
        gen_noise_tensor = torch.FloatTensor(self.batch_size, self.z_size)
        gp_alpha_tensor = torch.FloatTensor(self.batch_size, 1, 1, 1)
        if tcuda.is_available():
            gen_noise_tensor = gen_noise_tensor.cuda()
            gp_alpha_tensor = gp_alpha_tensor.cuda()
        gen_noise_tensor = Variable(gen_noise_tensor, requires_grad=False)
        for epoch in trange(epoch_num, desc='Epoch'):
            pbar2 = tqdm(total=len(self.train_loader))
            generator_losses = []
            discriminator_losses = []

            for x, y in self.train_loader:
                batch_size = x.size()[0]

                if tcuda.is_available():
                    x = x.cuda()
                x = Variable(x, requires_grad=False)
                enable_gradients(self.G)
                disable_gradients(self.D)
                self.G.zero_grad()
                loss = wgan_generator_loss(gen_noise_tensor, self.G, self.D)
                loss.backward()
                self.G_optimizer.step()
                generator_losses.append(loss.data[0])

                self.D.zero_grad()
                enable_gradients(self.D)
                disable_gradients(self.G)
                self.D.zero_grad()
                loss = wgan_gp_discriminator_loss(gen_noise_tensor, x, self.G,
                                                  self.D, 10., gp_alpha_tensor)
                loss.backward()
                self.D_optimizer.step()
                discriminator_losses.append(loss.data[0])
                pbar2.update(batch_size)

            pbar2.close()
            tqdm.write('Training [{:>5}:{:>5}] D Loss {:.6f}, G Loss {:.6f}'.format(
                epoch + 1, epoch_num,
                torch.mean(torch.FloatTensor(discriminator_losses)),
                torch.mean(torch.FloatTensor(generator_losses))))


    def generate(self, gen_num=10):
        z = torch.rand(gen_num, self.z_size)
        if tcuda.is_available():
            z = z.cuda()
        z = Variable(z)
        self.G.eval()
        results = self.G(z)
        self.G.train()
        return results.data.numpy(),


    def save(self, generator_path, discriminator_path):
        super().save(generator_path, discriminator_path)
