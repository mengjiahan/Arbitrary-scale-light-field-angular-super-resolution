import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from einops import rearrange

# for HCI
class Net_CrossAttention(nn.Module):
    def __init__(self, angular_in, angular_out):
        super(Net_CrossAttention, self).__init__()
        ngroup, nblock, channel = 1, 6, 64
        self.channel = channel
        self.angRes = angular_in
        self.angRes_out = angular_out
        layer_num = 8
        self.MHSA_params = {}
        self.MHSA_params['num_heads'] = 8
        self.MHSA_params['dropout'] = 0.
        self.FeaExtract = nn.Sequential(
            nn.Conv3d(1, channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False),
        )
        self.DeepFeaExt = CascadedBlocks(layer_num, channel, angular_in)
        self.epiFeatureRebuild = EpiFeatureRebuild(self.angRes, self.angRes_out, channel, feat_unfold=False)
        self.DownSample = nn.Sequential(
            nn.Conv3d(channel, channel // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel // 4 , channel // 4 // 4, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(channel // 4 // 4, 1, kernel_size=1, stride=1,
                      padding=0, bias=False),
        )
        # self.UpSample = nn.Sequential(
        #     nn.Conv3d(channel * angular_in * angular_in, channel // 4 * angular_in * angular_in, kernel_size=1,
        #               stride=1, padding=0, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(channel // 4 * angular_in * angular_in, channel // 4 * angular_out * angular_out, kernel_size=1,
        #               stride=1, padding=0, bias=False),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv3d(channel // 4 * angular_out * angular_out, angular_out * angular_out, kernel_size=1, stride=1,
        #               padding=0, bias=False),
        # )
        self.altblock = self.make_layer(layer_num=layer_num)

    def make_layer(self, layer_num):
        layers = []
        # layers.append(C42_Trans_serial(self.angRes, self.channels, self.MHSA_params, layer_num))
        for i in range(layer_num):
            layers.append(EPIX_Trans(self.angRes, self.channel, self.MHSA_params))
            layers.append(SA_Epi_Trans(self.angRes, self.channel, self.MHSA_params))
            # layers.append(CascadedBlocks(1, self.channels, self.angRes))
            # layers.append(C42_Trans_parallel(self.angRes, self.channels, self.MHSA_params))
        layers.append(
            nn.Conv3d(self.channel, self.channel, kernel_size=(1, 3, 3), padding=(0, 1, 1), dilation=1, bias=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x_mv = LFsplit(x, self.angRes)
        buffer = self.FeaExtract(x_mv)
        buffer = self.altblock(self.DeepFeaExt(buffer))
        buffer = rearrange(buffer, 'b c (u v) h w -> b c u v h w', u=self.angRes, v=self.angRes)
        buffer = self.epiFeatureRebuild(buffer)
        buffer = rearrange(buffer, 'b c u v h w -> b c (u v) h w')
        b, c, n, h, w = buffer.shape

        # buffer = buffer.contiguous().view(b, c * n, 1, h, w)
        buffer = self.DownSample(buffer).view(b, 1, self.angRes_out * self.angRes_out, h, w)  # n == angRes * angRes
        out = FormOutput(buffer)
        return out


class EpiFeatureRebuild(nn.Module):
    def __init__(self, angRes_in, angRes_out, channels, feat_unfold=True, local_ensemble=False, cell_decode=False):
        super().__init__()
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.cell_decode = cell_decode

        imnet_in_dim = channels
        if self.feat_unfold:
            imnet_in_dim *= 9
        imnet_in_dim += 2  # attach coord
        if self.cell_decode:
            imnet_in_dim += 2
        self.imnet = MLP(in_dim=imnet_in_dim, out_dim=channels, hidden_list=[256, 256, 256, 256])

    def query_feature(self, Feature, coord, cell=None):
        feat = Feature

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]
            areas[0] = areas[3]
            areas[3] = t

            t = areas[1]
            areas[1] = areas[2]
            areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def query_Epi(self, epi):
        buh, c, v, w = epi.shape

        # 2 x W --> 7 x W
        coord = make_coord([self.angRes_out, w]).cuda() \
            .unsqueeze(0).expand(epi.shape[0], w * self.angRes_out, 2)
        output_epi = self.query_feature(epi, coord, cell=None).permute(0, 2, 1) \
            .view(epi.shape[0], -1, self.angRes_out, w)

        # buh, c, angRes_out, w
        return output_epi

    def forward(self, x):
        batch_size, channle, u, v, h, w = x.shape

        # 2 x 2 x H x W --> 2 x 7 x H x W
        horizontal_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b u h) c v w'))
        x = rearrange(horizontal_x, '(b u h) c v w -> b c u v h w', b=batch_size, h=h)
        # 2 x 7 x H x W --> 7 x 7 x H x W
        vertical_x = self.query_Epi(rearrange(x, 'b c u v h w -> (b v w) c u h'))
        output = rearrange(vertical_x, '(b v w) c u h -> b c u v h w', b=batch_size, w=w)

        return output


class MLP(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_list):
        super().__init__()
        layers = []
        lastv = in_dim
        for hidden in hidden_list:
            layers.append(nn.Linear(lastv, hidden))
            layers.append(nn.ReLU())
            lastv = hidden
        layers.append(nn.Linear(lastv, out_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        shape = x.shape[:-1]
        x = self.layers(x.view(-1, x.shape[-1]))
        return x.view(*shape, -1)


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


class InitFeaExtract(nn.Module):
    def __init__(self, channel):
        super(InitFeaExtract, self).__init__()
        self.FEconv = nn.Sequential(
            nn.Conv2d(1, channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        b, n, _, h, w = x.shape
        x = x.contiguous().view(b * n, -1, h, w)
        buffer = self.FEconv(x)
        _, c, h, w = buffer.shape
        buffer = buffer.unsqueeze(1).contiguous().view(b, -1, c, h,
                                                       w)  # .permute(0,2,1,3,4)  # buffer_sv:  B, N, C, H, W

        return buffer


class Upsample(nn.Module):
    def __init__(self, channel, angular_in, angular_out):
        super(Upsample, self).__init__()
        self.an = angular_in
        self.an_out = angular_out
        self.angconv = nn.Sequential(
        )
        self.upsp = nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=angular_in, stride=angular_in, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel * angular_out * angular_out, kernel_size=1, padding=0),
            nn.PixelShuffle(angular_out),
            nn.Conv2d(channel, 1, kernel_size=3, padding=1))

    def forward(self, x):
        b, n, c, h, w = x.shape
        x = x.contiguous().view(b, n, c, h * w)
        x = torch.transpose(x, 1, 3)
        x = x.contiguous().view(b * h * w, c, self.an, self.an)
        up_in = self.angconv(x)

        out = self.upsp(up_in)

        out = out.view(b, h * w, -1, self.an_out * self.an_out)
        out = torch.transpose(out, 1, 3)
        out = out.contiguous().view(b, self.an_out * self.an_out, -1, h, w)  # [N*81,c,h,w]
        return out


class SA_Epi_CrossAttention_Trans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(SA_Epi_CrossAttention_Trans, self).__init__()
        self.emb_dim = emb_dim
        self.sa_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.epi_linear_in = nn.Linear(channels//2, emb_dim, bias=False)
        self.sa_norm = nn.LayerNorm(emb_dim)
        self.epi_norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels//2, bias=False)

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        sa_token = token[:, :, :c//2]
        epi_token = token[:, :, c//2:]

        epi_token_short_cut = epi_token

        sa_token = self.sa_linear_in(sa_token)
        epi_token = self.epi_linear_in(epi_token)

        sa_token_norm = self.sa_norm(sa_token)
        epi_token_norm = self.epi_norm(epi_token)
        sa_token = self.attention(query=sa_token_norm,
                                   key=epi_token_norm,
                                   value=sa_token,
                                   need_weights=False)[0] + sa_token

        sa_token = self.feed_forward(sa_token) + sa_token
        sa_token = self.linear_out(sa_token)

        buffer = torch.cat((sa_token, epi_token_short_cut), 2)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = buffer.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer


class SA_Epi_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(SA_Epi_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = SA_Epi_CrossAttention_Trans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)  # (b,c,v,u,w,h)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut


        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut
        # shortcut = buffer

        return buffer


class EpiXTrans(nn.Module):
    def __init__(self, channels, emb_dim, MHSA_params):
        super(EpiXTrans, self).__init__()
        self.emb_dim = emb_dim
        self.linear_in = nn.Linear(channels, emb_dim, bias=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim,
                                               MHSA_params['num_heads'],
                                               MHSA_params['dropout'],
                                               bias=False)
        nn.init.kaiming_uniform_(self.attention.in_proj_weight, a=math.sqrt(5))
        self.attention.out_proj.bias = None
        self.attention.in_proj_bias = None
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim * 2, bias=False),
            nn.ReLU(True),
            nn.Dropout(MHSA_params['dropout']),
            nn.Linear(emb_dim * 2, emb_dim, bias=False),
            nn.Dropout(MHSA_params['dropout'])
        )
        self.linear_out = nn.Linear(emb_dim, channels, bias=False)

    ######### very important!!!
    def gen_mask(self, h: int, w: int, maxdisp: int = 18):  # when 30 Scenes Reflective Occlusion
        # def gen_mask(self, h: int, w: int, maxdisp: int=18):  # when HCI data
        attn_mask = torch.zeros([h, w, h, w])
        # k_h_left = k_h // 2
        # k_h_right = k_h - k_h_left
        # k_w_left = k_w // 2
        # k_w_right = k_w - k_w_left
        [ii, jj] = torch.meshgrid(torch.arange(h), torch.arange(w))

        for i in range(h):
            for j in range(w):
                temp = torch.zeros(h, w)
                temp[(ii - i).abs() * maxdisp >= (jj - j).abs()] = 1
                # temp[max(0, i - k_h_left):min(h, i + k_h_right), max(0, j - k_w_left):min(w, j + k_w_right)] = 1
                attn_mask[i, j, :, :] = temp

        # attn_mask = rearrange(attn_mask, 'a b c d -> (a b) (c d)')
        attn_mask = attn_mask.reshape(h * w, h * w)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))

        return attn_mask

    def forward(self, buffer):
        # [_, _, n, v, w] = buffer.size()
        # b, c, u, h, v, w = buffer.shape
        b, c, u, v, h, w = buffer.shape
        # attn_mask = self.gen_mask(v, w, self.mask_field[0], self.mask_field[1]).to(buffer.device)
        attn_mask = self.gen_mask(v, w, ).to(buffer.device)

        # epi_token = rearrange(buffer, 'b c n v w -> (v w) (b n) c')
        epi_token = buffer.permute(3, 5, 0, 2, 4, 1).reshape(v * w, b * u * h, c)
        epi_token = self.linear_in(epi_token)

        epi_token_norm = self.norm(epi_token)
        epi_token = self.attention(query=epi_token_norm,
                                   key=epi_token_norm,
                                   value=epi_token,
                                   attn_mask=attn_mask,
                                   need_weights=False)[0] + epi_token

        epi_token = self.feed_forward(epi_token) + epi_token
        epi_token = self.linear_out(epi_token)
        # buffer = rearrange(epi_token, '(v w) (b n) c -> b c n v w', v=v, w=w, n=n)
        buffer = epi_token.reshape(v, w, b, u, h, c).permute(2, 5, 3, 0, 4, 1).reshape(b, c, u, v, h, w)

        return buffer


class EPIX_Trans(nn.Module):
    def __init__(self, angRes, channels, MHSA_params):
        super(EPIX_Trans, self).__init__()
        self.angRes = angRes

        self.epi_trans = EpiXTrans(channels, channels * 2, MHSA_params)
        self.conv_1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        )

    def forward(self, x):
        # [_, _, _, h, w] = x.size()
        b, c, n, h, w = x.size()

        u, v = self.angRes, self.angRes

        shortcut = x

        # EPI uh
        buffer = x.reshape(b, c, u, v, h, w).permute(0, 1, 3, 2, 5, 4)  # (b,c,v,u,w,h)
        buffer = self.conv_1(self.epi_trans(buffer).permute(0, 1, 3, 2, 5, 4).reshape(b, c, n, h, w)) + shortcut


        # EPI vw
        buffer = buffer.reshape(b, c, u, v, h, w)
        buffer = self.conv_1(self.epi_trans(buffer).reshape(b, c, n, h, w)) + shortcut
        # shortcut = buffer

        return buffer


class C42_Conv(nn.Module):
    def __init__(self, ch, angRes):
        super(C42_Conv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch, E_ch, D_ch = ch, ch, ch // 2, ch // 2
        self.angRes = angRes
        self.spaconv = SpatialConv(ch)
        self.angconv = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, E_ch)
        self.dpiconv = EPiConv(ch, angRes, D_ch)
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels=S_ch + A_ch + E_ch + E_ch + D_ch + D_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))

    def forward(self, x):
        # b, n, c, h, w = x.shape
        b, c, n, h, w = x.shape
        an = int(math.sqrt(n))
        s_out = self.spaconv(x)
        a_out = self.angconv(x)

        epih_in = x.contiguous().view(b, c, an, an, h, w)  # b,c,u,v,h,w
        epih_out = self.epiconv(epih_in)

        # epiv_in = epih_in.permute(0,2,1,3,5,4)
        epiv_in = epih_in.permute(0, 1, 3, 2, 5, 4)  # b,c,v,u,w,h
        epiv_out = self.epiconv(epiv_in).reshape(b, -1, an, an, w, h).permute(0, 1, 3, 2, 5, 4).reshape(b, -1, n, h, w)

        dpih_in = epih_in.permute(0, 1, 3, 2, 4, 5)  # b,c,v,u,h,w
        dpih_out = self.dpiconv(dpih_in).reshape(b, -1, an, an, w, h).permute(0, 1, 3, 2, 4, 5).reshape(b, -1, n, h, w)

        dpiv_in = epih_in.permute(0, 1, 2, 3, 5, 4)  # b,c,u,v,w,h
        dpiv_out = self.dpiconv(dpiv_in).reshape(b, -1, an, an, w, h).permute(0, 1, 2, 3, 5, 4).reshape(b, -1, n, h, w)

        out = torch.cat((s_out, a_out, epih_out, epiv_out, dpih_out, dpiv_out), 1)
        out = self.fuse(out)

        return out + x  # out.contiguous().view(b,n,c,h,w) + x


class SA_Conv(nn.Module):
    def __init__(self, ch, angRes):
        super(SA_Conv, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        S_ch, A_ch, E_ch = ch, ch, ch // 2
        self.angRes = angRes
        self.spaconv = SpatialConv(ch)
        self.angconv = AngularConv(ch, angRes, A_ch)
        self.epiconv = EPiConv(ch, angRes, E_ch)
        self.SA_fuse = nn.Sequential(
            nn.Conv3d(in_channels=S_ch + A_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))
        self.Epi_fuse = nn.Sequential(
            nn.Conv3d(in_channels= E_ch + E_ch, out_channels=ch, kernel_size=1, stride=1,
                      padding=0, dilation=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(ch, ch // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), dilation=1))

    def forward(self, x):
        # b, n, c, h, w = x.shape
        b, c, n, h, w = x.shape
        an = int(math.sqrt(n))
        s_out = self.spaconv(x)
        a_out = self.angconv(x)

        sa_out = self.SA_fuse(torch.cat((s_out, a_out), 1))

        epih_in = x.contiguous().view(b, c, an, an, h, w)  # b,c,u,v,h,w
        epih_out = self.epiconv(epih_in)

        # epiv_in = epih_in.permute(0,2,1,3,5,4)
        epiv_in = epih_in.permute(0, 1, 3, 2, 5, 4)  # b,c,v,u,w,h
        epiv_out = self.epiconv(epiv_in).reshape(b, -1, an, an, w, h).permute(0, 1, 3, 2, 5, 4).reshape(b, -1, n, h, w)

        epi_out = self.Epi_fuse(torch.cat((epih_out, epiv_out), 1))

        out = torch.cat((sa_out, epi_out), 1)

        return out + x  # out.contiguous().view(b,n,c,h,w) + x


class SpatialConv(nn.Module):
    def __init__(self, ch):
        super(SpatialConv, self).__init__()
        self.spaconv_s = nn.Sequential(
            nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv3d(in_channels=ch, out_channels=ch, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1),
                      dilation=(1, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, fm):
        return self.spaconv_s(fm)


class AngularConv(nn.Module):
    def __init__(self, ch, angRes, AngChannel):
        super(AngularConv, self).__init__()
        self.angconv = nn.Sequential(
            nn.Conv3d(ch * angRes * angRes, AngChannel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(AngChannel, AngChannel * angRes * angRes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.PixelShuffle(angRes)
        )
        # self.an = angRes

    def forward(self, fm):
        b, c, n, h, w = fm.shape
        a_in = fm.contiguous().view(b, c * n, 1, h, w)
        out = self.angconv(a_in).view(b, -1, n, h, w)  # n == angRes * angRes
        return out


class EPiConv(nn.Module):
    def __init__(self, ch, angRes, EPIChannel):
        super(EPiConv, self).__init__()
        self.epi_ch = EPIChannel
        self.epiconv = nn.Sequential(
            nn.Conv3d(ch, EPIChannel, kernel_size=(1, angRes, angRes // 2 * 2 + 1), stride=1,
                      padding=(0, 0, angRes // 2), bias=False),
            nn.LeakyReLU(0.1, True),
            nn.Conv3d(EPIChannel, angRes * EPIChannel, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0),
                      bias=False),  # ksize maybe (1,1,angRes//2*2+1) ?
            nn.LeakyReLU(0.1, True),
            # PixelShuffle1D(angRes),
        )
        # self.an = angRes

    def forward(self, fm):
        b, c, u, v, h, w = fm.shape

        epih_in = fm.permute(0, 1, 2, 4, 3, 5).reshape(b, c, u * h, v, w)
        epih_out = self.epiconv(epih_in)  # (b,self.epi_ch*v, u*h, 1, w)
        epih_out = epih_out.reshape(b, self.epi_ch, v, u, h, w).permute(0, 1, 3, 2, 4, 5).reshape(b, self.epi_ch, u * v,
                                                                                                  h, w)
        return epih_out


class PixelShuffle1D(nn.Module):
    def __init__(self, factor):
        super(PixelShuffle1D, self).__init__()
        self.factor = factor

    def forward(self, x):
        b, fc, h, w = x.shape
        c = fc // self.factor

        return x.view(b, c, h * self.factor, w)


class CascadedBlocks(nn.Module):
    '''
    Hierarchical feature fusion
    '''

    def __init__(self, n_blocks, channel, angRes):
        super(CascadedBlocks, self).__init__()
        self.n_blocks = n_blocks
        body = []
        for i in range(n_blocks):
            body.append(SA_Conv(channel, angRes))
        self.body = nn.Sequential(*body)
        # self.conv = nn.Conv2d(channel, channel, kernel_size = (3,3), stride = 1, padding = 1, dilation=1)
        self.conv = nn.Conv3d(channel, channel, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=1)

    def forward(self, x):
        buffer = x
        for i in range(self.n_blocks):
            buffer = self.body[i](buffer)
        buffer = self.conv(buffer) + x
        return buffer


# class CascadeC42Group(nn.Module):
#     def __init__(self, n_group, n_block, channels, angRes):
#         super(CascadeC42Group, self).__init__()
#         self.n_group = n_group
#         Groups = []
#         for i in range(n_group):
#             Groups.append(CascadedBlocks(n_block, channels, angRes))
#         self.Group = nn.Sequential(*Groups)
#         self.conv = nn.Conv3d(channels, channels, kernel_size = (1,3,3), stride = (1,1,1), padding = (0,1,1), dilation=1)
#
#     def forward(self, x):
#         buffer = x
#         for i in range(self.n_group):
#             buffer = self.Group[i](buffer)
#         buffer = self.conv(buffer)
#         return buffer + x

def LFsplit(data, angRes):
    b, _, H, W = data.shape
    h = int(H / angRes)
    w = int(W / angRes)
    data_sv = []
    for u in range(angRes):
        for v in range(angRes):
            data_sv.append(data[:, :, u * h:(u + 1) * h, v * w:(v + 1) * w])

    data_st = torch.stack(data_sv, dim=1)
    return data_st.permute(0, 2, 1, 3, 4)


def FormOutput(x_sv):
    x_sv = x_sv.permute(0, 2, 1, 3, 4)
    b, n, c, h, w = x_sv.shape
    angRes = int(math.sqrt(n + 1))
    out = []
    kk = 0
    for u in range(angRes):
        buffer = []
        for v in range(angRes):
            buffer.append(x_sv[:, kk, :, :, :])
            kk = kk + 1
        buffer = torch.cat(buffer, 3)
        out.append(buffer)
    out = torch.cat(out, 2)

    return out


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    net = Net_CrossAttention(2, 15).cuda()
    # from thop import profile
    # ##### get input index ######
    # input = torch.randn(1, 1, 128, 128).cuda()
    # total = sum([param.nelement() for param in net.parameters()])
    # flops, params = profile(net, inputs=(input,))
    # print('   Number of parameters: %.4fM' % (total / 1e6))
    # print('   Number of FLOPs: %.4fG' % (flops / 1e9))

    input = torch.randn(1, 1, 128, 128).cuda()
    # input = torch.randn(1, 64, 2, 16, 16)
    # conv = nn.Conv3d(64, 16, kernel_size=1, stride=1, padding=0, bias=False)
    # out = conv(input)
    out = net(input)
    print(out.shape)