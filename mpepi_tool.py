import math

import numpy as np

import numpy
from matplotlib import pyplot as plt
from scipy.io import loadmat #用于加载matlab类型数据文件



def fuckDataZL(data,path):
    great = loadmat(path)["data"] #导入path路径下的data文件，必康强制转换类型为mat
    print(np.allclose(data, great))#打印出传入的data数据和路径下的great数据是否在移动精度下一致，如一致则输出ture
    print(f"the x0's max is : {np.max(data)}, min is : {np.min(data)} shape is :{data.shape}")#shape：比如2*3的矩阵，则输出的形状就是（2，3）
    return np.allclose(data, great)

def Propagator_function(N, wavelength, area, z):#用于生成传播函数（传播子）的矩阵表示
    P = np.zeros((N, N),np.complex)
    for i in range(N):
        for j in range(N):
            alpha = wavelength * (i+1-N / 2-1) / area
            beta = wavelength * (j+1-N / 2-1) / area
            if (alpha ** 2 + beta ** 2) <= 1:
                P[i,j] = np.exp(-2 * np.pi * 1j * z * np.sqrt(1 - pow(alpha,2) -pow(beta,2) )/ wavelength)
    return P



def meshgrid(x):#用于生成网格坐标（调用的np.meshgrid）
    y = x
    XX, YY = np.meshgrid(x, y)
    return (XX,YY)


import numpy as np
import math
import copy#拷贝库，能实现浅拷贝和深拷贝：浅拷贝拷贝本身、深拷贝可以拷贝本身和关系 copy.deepcopy()

def fft1(src, dst=None):#一维快速傅里叶变换；能够实现信号的时域和频域的转换，同时能提取信号的频率特征和能量分布
    '''
    src: list is better.One dimension.
    '''
    l = len(src)
    n = int(math.log(l, 2))

    bfsize = np.zeros((l), dtype="complex")

    for i in range(n + 1):
        if i == 0:
            for j in range(l):
                bfsize[j] = src[Dec2Bin_Inverse2Dec(j, n)]
        else:
            tmp = copy.copy(bfsize)
            for j in range(l):
                pos = j % (pow(2, i))
                if pos < pow(2, i - 1):
                    bfsize[j] = tmp[j] + tmp[j + pow(2, i - 1)] * np.exp(complex(0, -2 * np.pi * pos / pow(2, i)))
                    bfsize[j + pow(2, i - 1)] = tmp[j] - tmp[j + pow(2, i - 1)] * np.exp(
                        complex(0, -2 * np.pi * pos / (pow(2, i))))
    return bfsize

def ifft1(src):
    for i in range(len(src)):
        src[i] = complex(src[i].real, -src[i].imag)

    res = fft1(src)

    for i in range(len(res)):
        res[i] = complex(res[i].real, -res[i].imag)

    return res / len(res)

def Dec2Bin_Inverse2Dec(n, m):#进制转换，在FFT中找到位置
    '''
    Especially for fft.To find position.
    '''
    b = bin(n)[2:]
    if len(b) != m:
        b = "0" * (m - len(b)) + b
    b = b[::-1]
    return int(b, 2)



def FT(inData):
    [Nx,Ny] = inData.shape
    h = np.zeros((Nx, Ny),np.complex)

    aaa = list(range(1, Nx+1))
    bbb = list(range(1, Ny+1))
    [m, h1] = meshgrid(aaa)
    [h2, n] = meshgrid(bbb)

    if Ny <= Nx:
        for i in range(Nx - Ny + 1):
            h2[Ny + i - 1,:]=h2[Ny - 1,:]
        h11 = h1[:,0: Ny]
    else:
        h11 = h1


    h = np.exp(1j* np.pi*(h11 + h2))
    FT = np.fft.fft2(h*inData)
    out = h*FT
    return out

def crops(inData,N0,M):#裁剪二维数组为N0大小
    out = np.zeros((N0, N0),inData.dtype)
    for ii in range(N0):
        for jj in range(N0):
            out[ii, jj] = inData[int(ii + np.floor((M - N0) / 2)),int(jj + np.floor((M - N0) / 2))]
    return out

def supportPro0(N):#指定大小的区域
    out = np.zeros((N, N))
    
    out[(N // 2) - 32:(N // 2) + 31, (N // 2) - 32:(N // 2) + 31] = 1#sim
    # out[(N // 2) - 200:(N // 2) + 199, (N // 2) - 200:(N // 2) + 199] = 1  # optics 

    
    return out

'''
def supportProORI():
    N = 1200
    out = np.zeros((N, N),)
    u = int(np.floor(N / 2))
    v = int(np.floor(N / 2))
    out[u-14-1:u + 14, v-14-1:v+14]=1

    return out
    '''

def IFT(inData):
    [Nx,Ny] = inData.shape
    h = np.zeros((Nx, Ny),np.complex)
    aaa = list(range(1, Nx + 1))
    bbb = list(range(1, Ny + 1))
    [m, h1] = meshgrid(aaa)
    [h2, n] = meshgrid(bbb)

    if Ny < Nx:
        for i in range(Nx - Ny + 1):
            h2[Ny + i - 1, :] = h2[Ny - 1, :]
        h11 = h1[:, 0: Ny]
    else:
        h11 = h1

    h = np.exp(-1j * np.pi*(h11 + h2))
    FT2 = np.fft.ifft2(h*inData)
    out = h*FT2
    return out
