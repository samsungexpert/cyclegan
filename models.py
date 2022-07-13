
import torch
import torch.nn as nn


# https://velog.io/@tjdtnsu/PyTorch-%EC%8B%A4%EC%A0%84-CycleGAN-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%ED%98%84%ED%95%98%EA%B8%B0

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.InstanceNorm2d(256),
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()


        model = []

        # 1, c7s1-64
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=7),
            nn.InstanceNorm2d(64),
        ]

        # 2, dk
        model += [self.__conv_block(64, 128),
                  self.__conv_block(128, 256)]

        # 3, Rk
        model += [ResidualBlock()] * num_blocks

        # 4, uk
        model += [
            self.__conv_block(256, 128, upsample=True),
            self.__conv_block(128, 64, upsample=True),
        ]

        # 5, c7s1-3
        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Tanh()]

        # 6
        self.model = nn.Sequential(*model)

    # 7
    def forward(self, x):
        return self.model(x)


    def __conv_block(self, in_features, out_features, upsample=False):
        if upsample:
            # 8
            conv = nn.ConvTranspose2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1)
        else:
            conv = nn.Conv2d(
                    in_channels=in_features,
                    out_channels=out_features,
                    kernel_size=3,
                    stride=2,
                    padding=1)

        # 9
        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        )




class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        # 10
        self.model = nn.Sequential(
            self.__conv_layer(3, 64, norm=False),
            self.__conv_layer(64, 128),
            self.__conv_layer(128, 256),
            self.__conv_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    # 11
    def __conv_layer(self, in_features, out_features, stride=2, norm=True):
        layer = [nn.Conv2d(in_features, out_features, 4, stride, 1)]

        if norm:
            layer.append(nn.InstanceNorm2d(out_features))

        layer.append(nn.LeakyReLU(0.2))

        layer = nn.Sequential(*layer)

        return layer

    def forward(self, x):
        return self.model(x)





if __name__ == "__main__":
    x = torch.rand((1, 3, 256, 256))
    generator = Generator(6)
    discriminator = Discriminator()

    print("G(x) shape:", generator(x).shape)
    print("D(x) shape:", discriminator(x).shape)