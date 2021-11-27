import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_channel_gan(gan, batch_size, n_epochs):

    gan.generator.train()
    gan.discriminator.train()
    optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    real_labels = torch.ones((batch_size, 1), device=gan.dev)
    fake_labels = torch.zeros((batch_size, 1), device=gan.dev)

    fig = plt.figure(figsize=[32, 8])
    (ax1, ax2, ax3) = fig.subplots(1, 3)
    gan.plot_gan_noise(ax1)
    gan.plot_gan_dist(ax3)

    for epoch in range(n_epochs):

        loss_g = 0
        loss_d = 0

        # train discriminator
        optimizer_d.zero_grad()
        channel_input = gan.gen_channel_input(batch_size)
        real_samples = gan.channel(channel_input)
        pred_real = gan.discriminator(real_samples)
        loss_d_real = criterion(pred_real, real_labels)
        
        noise = gan.gen_noise(batch_size)
        with torch.no_grad():
            # generator_input = torch.cat((channel_input, noise), dim=1)
            fake_samples = gan.generator(channel_input, noise)
        pred_fake_samples = gan.discriminator(fake_samples)
        loss_d_fake = criterion(pred_fake_samples, fake_labels)

        loss_d = (loss_d_real + loss_d_fake) / 2
        loss_d.backward()
        optimizer_d.step()

        # train generator
        optimizer_g.zero_grad()
        noise = gan.gen_noise(batch_size)
        # generator_input = torch.cat((channel_input, noise), dim=1)
        generated_samples = gan.generator(channel_input, noise)
        pred_generated_samples = gan.discriminator(generated_samples)
        loss_g = criterion(pred_generated_samples, real_labels)

        loss_g.backward()
        optimizer_g.step()

        loss_g = loss_g.item() / batch_size
        loss_d = loss_d.item() / batch_size

        if (epoch + 1) % 100 == 0:
            # print(d_loss.item())
            # print(g_loss.item())
            print('GAN #{}/{} | loss_g = {:.6f} | loss_d = {:.6f}'.format(epoch + 1, n_epochs, loss_g, loss_d))
            gan.plot_gan_gen(ax2)
            plt.pause(0.01)
            ax2.clear()
