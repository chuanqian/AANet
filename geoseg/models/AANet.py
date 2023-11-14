import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath
import timm
import math


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias, dilation=dilation, stride=stride,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=in_channels,
                      bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6(),
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=in_channels,
                      bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2, groups=in_channels,
                      bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MCAM(nn.Module):
    def __init__(self, decode_channel, class_channel):
        super(MCAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        b = 1
        gamma = 2
        t = int(abs((math.log(decode_channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        r = math.log(k - 1, 2)
        k1 = int(2 ** r + 1)
        k2 = int(2 ** (r + 1) + 1)
        k3 = int(2 ** (r + 2) + 1)
        k4 = int(2 ** (r + 3) + 1)
        self.conv0 = nn.Conv1d(1, 1, kernel_size=k1, padding=(k1 - 1) // 2)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k2, padding=(k2 - 1) // 2)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k3, padding=(k3 - 1) // 2)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=k4, padding=(k4 - 1) // 2)
        self.sigmoid = nn.Sigmoid()
        self.cSE = nn.Sequential(nn.Conv2d(decode_channel, decode_channel // 4, 1),
                                 nn.ReLU(inplace=True), nn.Conv2d(decode_channel // 4, class_channel, 1), nn.Sigmoid())
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * decode_channel, decode_channel)
        self.fc1 = nn.Sequential(ConvBN(decode_channel, class_channel, kernel_size=1), nn.Softmax(dim=1))

    def forward(self, x):
        y = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        ym = self.max_pool(x).squeeze(-1).transpose(1, 2)
        y0 = self.conv0(y).transpose(1, 2).unsqueeze(-1)
        y1 = self.conv1(y).transpose(1, 2).unsqueeze(-1)
        y2 = self.conv2(y).transpose(1, 2).unsqueeze(-1)
        y3 = self.conv3(y).transpose(1, 2).unsqueeze(-1)
        ym0 = self.conv0(ym).transpose(1, 2).unsqueeze(-1)
        ym1 = self.conv1(ym).transpose(1, 2).unsqueeze(-1)
        ym2 = self.conv2(ym).transpose(1, 2).unsqueeze(-1)
        ym3 = self.conv3(ym).transpose(1, 2).unsqueeze(-1)
        y_full = torch.cat([y0, y1, y2, y3, ym0, ym1, ym2, ym3], dim=1).squeeze(-1).transpose(1, 2)
        y = self.fc(self.conv(y_full).transpose(1, 2).squeeze(-1))
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        y1 = self.cSE(y)
        print("y1: ", y1.shape)
        class_feat = self.fc1(y).squeeze(-1) + y1.squeeze(-1)
        print(class_feat.shape)
        class_matrix = torch.bmm(y.squeeze(3), class_feat.transpose(2, 1)).unsqueeze(3)
        class_matrix = nn.functional.softmax(class_matrix, dim=1)
        class_matrix = nn.functional.softmax(class_matrix, dim=2)
        return class_matrix


class MSAM(nn.Module):
    def __init__(self, decode_channel, size):
        super(MSAM, self).__init__()
        self.h_pools = nn.AvgPool2d(kernel_size=(size, 1))
        self.w_pools = nn.AvgPool2d(kernel_size=(1, size))
        self.conv = ConvBNReLU(decode_channel, decode_channel, kernel_size=3)
        self.bn = nn.BatchNorm2d(decode_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x_m = x
        b, c, h, w = x_m.shape
        x_h = self.h_pools(x_m).view(b * c, 1, w)
        x_w = self.w_pools(x_m).view(b * c, h, 1)
        qk = torch.bmm(x_w, x_h).view(b, c, h * w)
        v = self.conv(x_m.view(b, c, h, w) * nn.functional.softmax(qk, -1).view(b, c, h, w) + x_m)
        return v


class CIAFM(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, dropout_rate=0.0):
        super(CIAFM, self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.f_query = ConvBNReLU(in_channels, key_channels, kernel_size=3, stride=1)
        self.f_key = ConvBNReLU(in_channels, key_channels, kernel_size=1)
        self.f_value = ConvBNReLU(in_channels, key_channels, kernel_size=1)
        self.f_up = ConvBNReLU(key_channels, in_channels, kernel_size=1)
        self.fuse = ConvBNReLU(in_channels, in_channels, kernel_size=3, stride=1)
        self.conv3x3 = nn.Sequential(ConvBNReLU(in_channels, out_channels, 3, 1))
        self.conv1x1 = nn.Sequential(
            ConvBNReLU(2 * in_channels, out_channels, 1), nn.Dropout2d(dropout_rate)
        )

    def forward(self, x, proxy):
        x_shape = x.shape
        query = self.f_query(x)
        query = torch.reshape(query, (x_shape[0], self.key_channels, -1))
        query = query.transpose(1, 2)
        key = self.f_key(proxy)
        key = torch.reshape(key, (x_shape[0], self.key_channels, -1))
        value = self.f_value(proxy)
        value = torch.reshape(value, (x_shape[0], self.key_channels, -1))
        value = value.transpose(1, 2)
        min_map = torch.matmul(query, key)
        min_map = (self.key_channels ** -0.5) * min_map
        min_map = nn.Softmax(dim=1)(min_map)
        context = torch.matmul(min_map, value)
        context = context.transpose(2, 1)
        context = torch.reshape(
            context, (x_shape[0], self.key_channels, x_shape[2], x_shape[3])
        )
        context = self.f_up(context)
        context = self.fuse(context + x)
        space_pixels = self.conv3x3(x)
        out_feats = torch.cat([context, space_pixels], dim=1)
        out_feats = self.conv1x1(out_feats)
        return out_feats


class Attention(nn.Module):
    def __init__(self, dim, num_classes, size, dropout_rate=0.2):
        super(Attention, self).__init__()
        self.channel_attn = MCAM(dim, num_classes)
        self.space_attn = MSAM(dim, size)
        self.af = CIAFM(dim, dim // 2, dim, dropout_rate)

    def forward(self, x, skip):
        x = skip + x
        c_attn = self.channel_attn(x)
        s_attn = self.space_attn(x)
        out = self.af(s_attn, c_attn)
        return out


class Block(nn.Module):
    def __init__(self, dim, classes_dim, size, mlp_ratio=4.0, drop=0.2, upSample=False, drop_path=0.0,
                 act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_classes=classes_dim, size=size, dropout_rate=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)
        self.upSample = upSample
        if self.upSample:
            self.conv = ConvBNReLU(dim, dim, kernel_size=3)

    def forward(self, x, skip):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(skip)))
        out = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.upSample:
            out = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
            out = self.conv(out)
        return out


class Decoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes, size=(128, 64, 32, 16), dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.conv1 = ConvBN(encoder_channels[0], decoder_channels, kernel_size=3)
        self.conv2 = ConvBN(encoder_channels[1], decoder_channels, kernel_size=3)
        self.conv3 = ConvBN(encoder_channels[2], decoder_channels, kernel_size=3)
        self.conv4 = ConvBN(encoder_channels[3], decoder_channels, kernel_size=3)
        self.b4 = Block(dim=64, classes_dim=num_classes, size=size[3], drop=dropout_rate, upSample=True)
        self.b3 = Block(dim=64, classes_dim=num_classes, size=size[2], drop=dropout_rate, upSample=True)
        self.b2 = Block(dim=64, classes_dim=num_classes, size=size[1], drop=dropout_rate, upSample=True)
        self.b1 = Block(dim=64, classes_dim=num_classes, size=size[0], drop=dropout_rate)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decoder_channels, decoder_channels),
                                               nn.Dropout2d(p=dropout_rate, inplace=True),
                                               Conv(decoder_channels, num_classes, kernel_size=1))
        self.init_weight()

    def forward(self, res1, res2, res3, res4):
        stage1 = self.conv1(res1)
        stage2 = self.conv2(res2)
        stage3 = self.conv3(res3)
        stage4 = self.conv4(res4)
        b4 = self.b4(stage4, stage4)
        b3 = self.b3(b4, stage3)
        b2 = self.b2(b3, stage2)
        b1 = self.b1(b2, stage1)
        output = self.segmentation_head(b1)

        return output

    def init_weight(self):
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AANet(nn.Module):
    def __init__(self, num_classes, decoder_channels, size=(128, 64, 32, 16), backbone_name="swsl_resnet18",
                 pretrained=True):
        super(AANet, self).__init__()
        self.backbone = timm.create_model(backbone_name, features_only=True, out_indices=(1, 2, 3, 4),
                                          pretrained=pretrained)
        encoder_channels = self.backbone.feature_info.channels()
        self.decoder = Decoder(encoder_channels, decoder_channels, num_classes=num_classes, size=size, dropout_rate=0.5)

    def forward(self, x):
        res1, res2, res3, res4 = self.backbone(x)
        output = self.decoder(res1, res2, res3, res4)
        output = F.interpolate(output, size=x.shape[2:], mode="bilinear", align_corners=True)
        return output


if __name__ == "__main__":
    model = AANet(num_classes=6, decoder_channels=64)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.5fM" % (total / 1e6))
    x = torch.randn([4, 3, 512, 512])
    model.train()
    out = model(x)
    print(out.shape)
