from torch.autograd import grad, Variable

def wgan_generator_loss(gen_noise, gen_net, disc_net):
    """
    Generator loss for Wasserstein GAN (same for WGAN-GP)

    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
    Outputs:
        loss (PyTorch scalar): Generator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out = disc_net(gen_data)
    # get loss
    loss = -disc_out.mean()
    return loss


def wcgan_generator_loss(gen_noise, labels, gen_net, disc_net):
    gen_noise.data.normal_()
    gen_data = gen_net(gen_noise, labels)
    disc_out = disc_net(gen_data, labels)
    loss = -disc_out.mean()
    return loss


def wgan_gp_discriminator_loss(gen_noise, real_data, gen_net, disc_net, lmbda,
                               gp_alpha):
    """
    Discriminator loss with gradient penalty for Wasserstein GAN (WGAN-GP)

    Inputs:
        gen_noise (PyTorch Tensor): Noise to feed through generator
        real_data (PyTorch Tensor): Noise to feed through generator
        gen_net (PyTorch Module): Network to generate images from noise
        disc_net (PyTorch Module): Network to determine whether images are real
                                   or fake
        lmbda (float): Hyperparameter for weighting gradient penalty
        gp_alpha (PyTorch Tensor): Values to use to randomly interpolate
                                   between real and fake data for GP penalty
    Outputs:
        loss (PyTorch scalar): Discriminator Loss
    """
    # draw noise
    gen_noise.data.normal_()
    # get generated data
    gen_data = gen_net(gen_noise)
    # feed data through discriminator
    disc_out_gen = disc_net(gen_data)
    disc_out_real = disc_net(real_data)
    # get loss (w/o GP)
    loss = disc_out_gen.mean() - disc_out_real.mean()
    # draw interpolation values
    gp_alpha.uniform_()
    # interpolate between real and generated data
    interpolates = gp_alpha * real_data.data + (1 - gp_alpha) * gen_data.data
    interpolates = Variable(interpolates, requires_grad=True)
    # feed interpolates through discriminator
    disc_out_interp = disc_net(interpolates)
    # get gradients of discriminator output with respect to input
    gradients = grad(outputs=disc_out_interp.sum(), inputs=interpolates,
                     create_graph=True)[0]
    # calculate gradient penalty
    grad_pen = ((gradients.view(gradients.size(0), -1).norm(
        2, dim=1) - 1)**2).mean()
    # add gradient penalty to loss
    loss += lmbda * grad_pen
    return loss

def wcgan_gp_discriminator_loss(gen_noise, gen_labels, real_data, real_labels, gen_net, disc_net, lmbda,
                               gp_alpha):
    gen_noise.data.normal_()
    gen_data = gen_net(gen_noise, gen_labels)
    disc_out_gen = disc_net(gen_data, gen_labels)
    disc_out_real = disc_net(real_data, real_labels)
    loss = disc_out_gen.mean() - disc_out_real.mean()
    gp_alpha.uniform_()
    interpolates = gp_alpha * real_data.data + (1 - gp_alpha) * gen_data.data
    labels_interpolates = gp_alpha * real_labels.data + (1 - gp_alpha) * gen_labels.data
    interpolates = Variable(interpolates, requires_grad=True)
    labels_interpolates = Variable(labels_interpolates, requires_grad=False)
    disc_out_interp = disc_net(interpolates, labels_interpolates)
    gradients = grad(outputs=disc_out_interp.sum(), inputs=interpolates,
                     create_graph=True)[0]
    grad_pen = ((gradients.view(gradients.size(0), -1).norm(
        2, dim=1) - 1)**2).mean()
    loss += lmbda * grad_pen
    return loss


def enable_gradients(net):
    for p in net.parameters():
        p.requires_grad = True


def disable_gradients(net):
    for p in net.parameters():
        p.requires_grad = False
