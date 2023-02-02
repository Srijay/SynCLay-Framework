import torch
import torch.nn as nn


from layers import GlobalAvgPool, build_cnn, ResnetBlock, get_norm_layer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def tissue_image_generator(input_nc, output_nc, ngf, n_downsample_global=3, n_blocks_global=9, norm='instance'):
    norm_layer = get_norm_layer(norm_type=norm)
    netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    # netG.cuda()
    netG.apply(weights_init)
    return netG


##############################################################################
# Generator
##############################################################################
class GlobalGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), activation]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        prediction = self.model(input)
        return prediction


def gan_generator(dim):

    # define the standalone generator model
    layers = []
    ngf = 1

    layers.append(nn.ConvTranspose2d(dim, ngf * 8, 4, 1, 0, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 8))
    layers.append(nn.ReLU(True))

    layers.append(nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 4))
    layers.append(nn.ReLU(True))

    layers.append(nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf * 2))
    layers.append(nn.ReLU(True))

    #32X32
    layers.append(nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(ngf))
    layers.append(nn.ReLU(True))

    #64X64
    out_channels = 1
    layers.append(nn.ConvTranspose2d(ngf, out_channels, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(True))

    #128X128
    out_channels = 1
    layers.append(nn.ConvTranspose2d(1, out_channels, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(True))

    #256X256
    out_channels = 1
    layers.append(nn.ConvTranspose2d(1, out_channels, 4, 2, 1, bias=False))
    layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class Pix2PixGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(Pix2PixGenerator, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

        # self.down1 = UNetDown(in_channels, 16, normalize=False)
        # self.down2 = UNetDown(16, 32)
        # self.down3 = UNetDown(32, 64)
        # self.down4 = UNetDown(64, 128, dropout=0.5)
        # self.down5 = UNetDown(128, 256, dropout=0.5)
        # self.down6 = UNetDown(256, 512, dropout=0.5)
        # self.down7 = UNetDown(512, 512, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
        #
        # self.up1 = UNetUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 256, dropout=0.5)
        # self.up4 = UNetUp(512, 128, dropout=0.5)
        # self.up5 = UNetUp(256, 64)
        # self.up6 = UNetUp(128, 32)
        # self.up7 = UNetUp(64, 16)
        #
        # self.final = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(32, out_channels, 4, padding=1),
        #     nn.Tanh(),
        # )


    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        return self.final(u7)


def pix2pix_generator(in_channels=3):
    netG = Pix2PixGenerator(in_channels=in_channels)
    # assert (torch.cuda.is_available())
    # netG.cuda()
    netG.apply(weights_init)
    return netG