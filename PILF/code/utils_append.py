import os
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader
from skimage import metrics
from scipy.interpolate import interp2d

class TrainSetLoader(Dataset):
    def __init__(self, dataset_dir):
        super(TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(dataset_dir)
        item_num = len(self.file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data, label = augmentation(data, label)
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num


class new_TrainSetLoader(Dataset):
    def __init__(self, dataset_dir, angRes_in, angRes_out):
        super(new_TrainSetLoader, self).__init__()
        self.dataset_dir = dataset_dir
        self.angRes_in = angRes_in
        self.angRes_out = angRes_out
        self.file_list = os.listdir(dataset_dir)
        item_num = len(self.file_list)
        self.item_num = item_num

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data, label = new_augmentation(data, label, self.angRes_in, self.angRes_out)
            data = ToTensor()(data.copy())
            label = ToTensor()(label.copy())
        return data, label

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    dataset_dir = args.testset_dir
    data_list = os.listdir(dataset_dir)

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name, Lr_Info=None)
        length_of_tests += len(test_Dataset)
        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=0, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name, Lr_Info=None):
        super(TestSetDataLoader, self).__init__()
        self.dataset_dir = args.testset_dir + data_name
        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = tmp_list[index]

        self.file_list.extend(tmp_list)
        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = self.dataset_dir + '/' + self.file_list[index]
        with h5py.File(file_name, 'r') as hf:
            data = np.array(hf.get('data'))
            label = np.array(hf.get('label'))
            data, label = np.transpose(data, (1, 0)), np.transpose(label, (1, 0))
            data, label = ToTensor()(data.copy()), ToTensor()(label.copy())

        return data, label

    def __len__(self):
        return self.item_num

class TestSetLoader(Dataset):
    def __init__(self, cfg, data_name = 'ALL', Lr_Info=None):
        super(TestSetLoader, self).__init__()
        self.angRes = cfg.angRes
        self.dataset_dir = cfg.data_for_test + str(cfg.angRes) + 'x' + str(cfg.angRes) + '_' + str(cfg.scale_factor) + 'xSR/'
        data_list = [data_name]

        self.Lr_Info = self.angRes

        self.file_list = []
        for data_name in data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            Lr_SAI_y = np.array(hf.get('data_SAI_y'))
            Sr_SAI_cbcr = np.array(hf.get('data_SAI_cbcr'))
            Hr_SAI_ycbcr = np.array(hf.get('label_SAI_ycbcr'))
            Lr_SAI_y = np.transpose(Lr_SAI_y, (1, 0))
            Hr_SAI_ycbcr = np.transpose(Hr_SAI_ycbcr, (0, 2, 1)).transpose(1, 2, 0)
            Sr_SAI_cbcr  = np.transpose(Sr_SAI_cbcr,  (0, 2, 1)).transpose(1, 2, 0)

        Lr_SAI_y = ToTensor()(Lr_SAI_y.copy())
        Hr_SAI_ycbcr = ToTensor()(Hr_SAI_ycbcr.copy())
        Sr_SAI_cbcr = ToTensor()(Sr_SAI_cbcr.copy())

        return Lr_SAI_y, Hr_SAI_ycbcr, Sr_SAI_cbcr, self.Lr_Info

    def __len__(self):
        return self.item_num


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5: # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)

    return data, label


def new_augmentation(data, label, angRes_in, angRes_out):
    if random.random() < 0.5:  # resize data
        data, label = resize_fuc(data, label, angRes_in, angRes_out)
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)

    return data, label


def resize_fuc(data, label, angRes_in, angRes_out):
    # factor = 2
    factor = 4
    patchsize = data.shape[0] // angRes_in
    new_data = np.zeros_like(data)
    new_label = np.zeros_like(label)
    # resize 2 倍
    # start = patchsize // 2 - patchsize // 2 // 2
    # end = patchsize // 2 + patchsize // 2 // 2
    # # resize 4 倍
    start = patchsize // 2 - patchsize // 2 // 2 // 2
    end = patchsize // 2 + patchsize // 2 // 2 // 2
    for u in range(angRes_in):
        for v in range(angRes_in):
            temp = data[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize]
            new_data[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize] = interplot2D(temp[start:end, start:end], factor)
    for u in range(angRes_out):
        for v in range(angRes_out):
            temp = label[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize]
            new_label[u*patchsize:(u+1)*patchsize, v*patchsize:(v+1)*patchsize] = interplot2D(temp[start:end, start:end], factor)

    return new_data, new_label


def interplot2D(original_array, factor):
    # 定义新的x和y坐标轴
    x_old = np.arange(original_array.shape[1])
    y_old = np.arange(original_array.shape[0])
    x_new = np.linspace(0, original_array.shape[1] - 1, original_array.shape[1] * factor)
    y_new = np.linspace(0, original_array.shape[0] - 1, original_array.shape[0] * factor)

    # 创建一个双线性插值函数
    interp_func = interp2d(x_old, y_old, original_array, kind='linear')

    # 使用插值函数对新的坐标轴进行插值
    new_array = interp_func(x_new, y_new)

    # 打印结果
    return new_array


def LFdivide(data, angRes, patch_size, stride):
    uh, vw = data.shape
    h0 = uh // angRes
    w0 = vw // angRes
    bdr = (patch_size - stride) // 2
    h = h0 + 2 * bdr
    w = w0 + 2 * bdr
    if (h - patch_size) % stride:
        numU = (h - patch_size)//stride + 2
    else:
        numU = (h - patch_size)//stride + 1
    if (w - patch_size) % stride:
        numV = (w - patch_size)//stride + 2
    else:
        numV = (w - patch_size)//stride + 1
    hE = stride * (numU-1) + patch_size
    wE = stride * (numV-1) + patch_size

    dataE = torch.zeros(hE*angRes, wE*angRes)
    for u in range(angRes):
        for v in range(angRes):
            Im = data[u*h0:(u+1)*h0, v*w0:(v+1)*w0]
            dataE[u*hE : u*hE+h, v*wE : v*wE+w] = ImageExtend(Im, bdr)
    subLF = torch.zeros(numU, numV, patch_size*angRes, patch_size*angRes)
    for kh in range(numU):
        for kw in range(numV):
            for u in range(angRes):
                for v in range(angRes):
                    uu = u*hE + kh*stride
                    vv = v*wE + kw*stride
                    subLF[kh, kw, u*patch_size:(u+1)*patch_size, v*patch_size:(v+1)*patch_size] = dataE[uu:uu+patch_size, vv:vv+patch_size]
    return subLF


def ImageExtend(Im, bdr):
    h, w = Im.shape
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])
    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[h - bdr: 2 * h + bdr, w - bdr: 2 * w + bdr]

    return Im_out


def LFintegrate(subLF, angRes, pz, stride, h0, w0):
    numU, numV, pH, pW = subLF.shape
    ph, pw = pH //angRes, pW //angRes
    bdr = (pz - stride) //2
    temp = torch.zeros(stride*numU, stride*numV)
    outLF = torch.zeros(angRes, angRes, h0, w0)
    for u in range(angRes):
        for v in range(angRes):
            for ku in range(numU):
                for kv in range(numV):
                    temp[ku*stride:(ku+1)*stride, kv*stride:(kv+1)*stride] = subLF[ku, kv, u*ph+bdr:u*ph+bdr+stride, v*pw+bdr:v*ph+bdr+stride]
            outLF[u, v, :, :] = temp[0:h0, 0:w0]

    return outLF


def cal_psnr(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    return metrics.peak_signal_noise_ratio(img1_np, img2_np)

def cal_ssim(img1, img2):
    img1_np = img1.data.cpu().numpy()
    img2_np = img2.data.cpu().numpy()

    out = metrics.structural_similarity(img1_np, img2_np, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)

    return out

def cal_metrics_RE(img1, img2, angRes_in, angRes_out):
    if len(img1.size())==2:
        [H, W] = img1.size()
        img1 = img1.view(7, H // 7, 7, W // 7).permute(0,2,1,3)
    if len(img2.size())==2:
        [H, W] = img2.size()
        img2 = img2.view(angRes_out, H // angRes_out, angRes_out, W // angRes_out).permute(0,2,1,3)

    [U, V, h, w] = img1.size()
    [U2, V2, h, w] = img2.size()
    step = (U2-U) // (U-1)
    PSNR = np.zeros(shape=(U, V), dtype='float32')
    SSIM = np.zeros(shape=(U, V), dtype='float32')
    bd = 22
    for u in range(U):
        for v in range(V):
            # k = u * U + v
            # if k in indicate:
            #     PSNR[u, v] = 0
            # else:
        # PSNR[u, v] = cal_psnr(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
            # SSIM[u, v] = cal_ssim(img1[u, v, bd:-bd, bd:-bd], img2[u, v, bd:-bd, bd:-bd])
            PSNR[u, v] = cal_psnr(img1[u, v, bd:-bd, bd:-bd], img2[u * (step+1), v * (step+1), bd:-bd, bd:-bd])
            SSIM[u, v] = cal_ssim(img1[u, v, bd:-bd, bd:-bd], img2[u * (step+1), v * (step+1), bd:-bd, bd:-bd])
            # print(u, v, sep=',')
            # print( u * (step+1),v * (step+1), sep=',')
            pass
        pass

    # for u in range(0, angRes_out, (angRes_out - 1) // (angRes_in - 1)):
    #     for v in range(0, angRes_out, (angRes_out - 1) // (angRes_in - 1)):
    #         PSNR[u, v] = 0
    #         SSIM[u, v] = 0

    for u in range(0,angRes_out, (angRes_out - 1)):
        for v in range(0, angRes_out, (angRes_out - 1)):
            PSNR[u, v] = 0
            SSIM[u, v] = 0

    psnr_mean = PSNR.sum() / np.sum(PSNR > 0)
    ssim_mean = SSIM.sum() / np.sum(SSIM > 0)

    return psnr_mean, ssim_mean


if __name__ == '__main__':
    data = np.random.rand(128, 128)
    label = np.random.rand(448, 448)
    d, l = resize_fuc(data, label, 2, 7)
    print(d.shape)
    print(l.shape)

