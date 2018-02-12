from . import *
from .gan import GAN
from .generators import *
from .discriminators import *

class CGAN(GAN):
    def __init__(self, **kwargs):
        self.batch_size = kwargs['batch_size']
        self.train_loader = kwargs['train_loader']
        self.G = ConditionalBNGenerator(**kwargs)
        self.D = ConditionalBNDiscriminator(**kwargs)
        self.z_size = kwargs['z_size']
        self.class_num = kwargs['class_num']
        self.fixed_z = torch.rand(10, self.z_size)

        if tcuda.is_available():
            self.G, self.D = self.G.cuda(), self.D.cuda()

        # todo --> customizable
        self.BCE_loss = nn.BCELoss()

        # todo --> customizable
        # todo --> weight decay setting
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=kwargs['lrG'], betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=kwargs['lrD'], betas=(0.5, 0.999))


    def train(self, epoch_num=10):
        self.G.weight_init(mean=0, std=0.02)
        self.D.weight_init(mean=0, std=0.02)
        for epoch in trange(epoch_num, desc='Epoch'):
            pbar2 = tqdm(total=len(self.train_loader))

            generator_losses = []
            discriminator_losses = []
            for x, y in self.train_loader:
                self.D.zero_grad()
                batch_size = x.size()[0]

                y_real = torch.ones(batch_size)
                y_fake = torch.zeros(batch_size)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y.view(batch_size, 1), 1)
                if tcuda.is_available():
                    x, y_real, y_fake, y_label = x.cuda(), y_real.cuda(), y_fake.cuda(), y_label.cuda()
                x, y_real, y_fake, y_label = Variable(x), Variable(y_real), Variable(y_fake), Variable(y_label)

                y_pred = self.D(x, y_label).squeeze()
                real_loss = self.BCE_loss(y_pred, y_real)
                z = torch.rand((batch_size, self.z_size))
                y = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y.view(batch_size, 1), 1)
                if tcuda.is_available():
                    z, y_label = z.cuda(), y_label.cuda()
                z, y_label = Variable(z), Variable(y_label)
                y_pred = self.D(self.G(z, y_label), y_label).squeeze()
                fake_loss = self.BCE_loss(y_pred, y_fake)

                D_train_loss = real_loss + fake_loss
                D_train_loss.backward()
                discriminator_losses.append(D_train_loss.data[0])

                self.D_optimizer.step()

                self.G.zero_grad()
                z = torch.rand((batch_size, self.z_size))
                y = (torch.rand(batch_size, 1) * self.class_num).type(torch.LongTensor)
                y_label = torch.zeros(batch_size, self.class_num)
                y_label.scatter_(1, y.view(batch_size, 1), 1)

                if tcuda.is_available():
                    z, y_label = z.cuda(), y_label.cuda()
                z, y_label = Variable(z), Variable(y_label)
                y_pred = self.D(self.G(z, y_label), y_label).squeeze()
                G_train_loss = self.BCE_loss(y_pred, y_real)
                G_train_loss.backward()
                generator_losses.append(G_train_loss.data[0])

                self.G_optimizer.step()

                pbar2.update(batch_size)

            pbar2.close()
            tqdm.write('Training [{:>5}:{:>5}] D Loss {:.6f}, G Loss {:.6f}'.format(
                epoch + 1, epoch_num,
                torch.mean(torch.FloatTensor(discriminator_losses)),
                torch.mean(torch.FloatTensor(generator_losses))))


    def generate(self, gen_num=10):
        z = torch.rand(gen_num, self.z_size)
        c_ = torch.zeros(gen_num // self.class_num, 1)
        for i in range(1, self.class_num):
            temp = torch.zeros(gen_num // self.class_num, 1) + i
            c_ = torch.cat([c_, temp], 0)
        c = torch.zeros(gen_num, self.class_num)
        c.scatter_(1, c_.type(torch.LongTensor), 1)
        if tcuda.is_available():
            z, c = z.cuda(), c.cuda()
        z, c = Variable(z), Variable(c)
        self.G.eval()
        results = self.G(z, c)
        resultsd = torch.cat([results.data, c_], 1)
        return resultsd.numpy()


    def save(self, generator_path, discriminator_path):
        super().save(generator_path, discriminator_path)
