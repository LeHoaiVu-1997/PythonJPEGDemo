#!/usr/bin/env python
# coding: utf-8

# Implement image lossy compresssion/decompression method using DCT (JPEG standard).
#
# Evaluate the image quality (using both objective and subjective criteria)
#
# and processing time for different compression ratio.


import numpy as np
import cv2
from numpy.linalg import inv
import time
import os


# In[1]:


def dct_coeff():
    T = np.zeros([8, 8])
    for i in range(8):
        for j in range(8):
            if i == 0:
                T[i, j] = 1 / np.sqrt(8)
            elif i > 0:
                T[i, j] = np.sqrt(2 / 8) * np.cos((2 * j + 1) * i * np.pi / 16)
    return T


# In[2]:


def quantization_level(n, Quality_Factor):
    if n > 100:
        n = 100

    if n < 0:
        n = 1

    Q50 = np.zeros([8, 8])
    Q50 = Quality_Factor

    scalling_factor = 1
    if n >= 50:
        scalling_factor = (100 - n)/50

    if n < 50:
         scalling_factor = 50/n

    Q = np.zeros([8, 8])
    for i in range(8):
        for j in range(8):
            if scalling_factor != 0:
                Q[i, j] = np.round(Q50[i, j] * scalling_factor)
            else:
                Q[i, j] = 1

    Q = Q.astype(dtype= np.uint8)
    return Q


# In[3]:


def dct(M, T, T_prime):
    tmp = np.zeros(M.shape)
    mask = np.zeros([8, 8])
    for i in range(M.shape[0] // 8):
        for j in range(M.shape[1] // 8):
            mask = M[8 * i:8 * i + 8, 8 * j:8 * j + 8]
            tmp[8 * i:8 * i + 8, 8 * j:8 * j + 8] = T @ mask @ T_prime

    return (tmp)


# In[4]:


def quantiz_div(a, b):
    tmp = np.zeros(a.shape)
    for i in range(8):
        for j in range(8):
            tmp[i, j] = np.round(a[i, j] / b[i, j])
    return tmp


# In[5]:


def quantiz(D, Q):
    tmp = np.zeros(D.shape)
    mask = np.zeros([8, 8])
    for i in range(D.shape[0] // 8):
        for j in range(D.shape[1] // 8):
            mask = quantiz_div(D[8 * i:8 * i + 8, 8 * j:8 * j + 8], Q)
            tmp[8 * i:8 * i + 8, 8 * j:8 * j + 8] = mask
    return (tmp)


# In[6]:


def decompress_mul(a, b):
    tmp = np.zeros(a.shape)
    for i in range(8):
        for j in range(8):
            tmp[i, j] = a[i, j] * b[i, j]
    return tmp


# In[7]:


def decompress(C, Q, T, T_prime):
    R = np.zeros(C.shape)
    mask = np.zeros([8, 8])
    for i in range(C.shape[0] // 8):
        for j in range(C.shape[1] // 8):
            mask = decompress_mul(C[8 * i:8 * i + 8, 8 * j:8 * j + 8], Q)
            R[8 * i:8 * i + 8, 8 * j:8 * j + 8] = mask

    N = np.zeros(C.shape)

    for i in range(R.shape[0] // 8):
        for j in range(R.shape[1] // 8):
            mask = T_prime @ R[8 * i:8 * i + 8, 8 * j:8 * j + 8] @ T
            N[8 * i:8 * i + 8, 8 * j:8 * j + 8] = np.round(mask) + 128 * np.ones([8, 8])

    return N


# In[8]:


def Compress_img(file, level, imageType, QLumi, QChroma):
    inputPath = "D:/Python projects/JPEGCompression/" + str(imageType) + " Images"
    I = cv2.imread(os.path.join(inputPath, file))

    H = I.shape[0]
    W = I.shape[1]

    print("Image size: ", I.shape)

    # BGR color space
    #B, G, R = cv2.split(I)
    #B = B - 128 * np.ones([H, W])
    #G = G - 128 * np.ones([H, W])
    #R = R - 128 * np.ones([H, W])


    # LCrCb color space
    lumaI = cv2.cvtColor(I, cv2.COLOR_BGR2YCR_CB)

    Y, Cr, Cb = cv2.split(lumaI)

    # Subsampling Cb, Cr:
    subSamplingRaito_422(Cr, Cr.shape[0], Cr.shape[1])
    subSamplingRaito_422(Cb, Cb.shape[0], Cb.shape[1])

    Y = Y - 128 * np.ones([H, W])
    Cr = Cr - 128 * np.ones([H, W])
    Cb = Cb - 128 * np.ones([H, W])

    T = dct_coeff()
    T_prime = inv(T)
    QL = quantization_level(level, QLumi)
    QCm = quantization_level(level, QChroma)

    D_Y = dct(Y, T, T_prime) #D_R
    D_Cr = dct(Cr, T, T_prime) #D_G
    D_Cb = dct(Cb, T, T_prime) #D_B

    tmp = cv2.merge((D_Y, D_Cr, D_Cb)) #B-G-R

    fileName1 = 'DCT ' + str(file)
    path1 = "D:/Python projects/JPEGCompression/DCT/" + str(imageType) + "/" + str(level)
    cv2.imwrite(os.path.join(path1, fileName1), tmp)

    # Quantiz BGR
    #C_R = quantiz(D_R, Q)
    #C_R[C_R == 0] = 0
    #C_G = quantiz(D_G, Q)
    #C_G[C_G == 0] = 0
    #C_B = quantiz(D_B, Q)
    #C_B[C_B == 0] = 0

    # Quantiz YCrCb
    Q_Y = quantiz(D_Y, QL)
    Q_Cr = quantiz(D_Cr, QCm)
    Q_Cb = quantiz(D_Cb, QCm)

    tmp = cv2.merge((Q_Y, Q_Cr, Q_Cb)) # or C_B - C_G - C_R

    fileName2 = 'After_Quantiz ' + str(file)
    path2 = "D:/Python projects/JPEGCompression/After Quantiz/" + str(imageType) + "/" + str(level)
    cv2.imwrite(os.path.join(path2 , fileName2), tmp)

    #return C_B, C_G, C_R, Q, T, T_prime
    return Q_Y, Q_Cr, Q_Cb, QL, QCm, T, T_prime


# In[9]:


def Decompress_img(file, level, imageType, Q_Y, Q_Cr, Q_Cb, QLumi, QChroma, T, T_prime):
    N_Y = decompress(Q_Y, QLumi, T, T_prime)
    N_Cr = decompress(Q_Cr, QChroma, T, T_prime)
    N_Cb = decompress(Q_Cb, QChroma, T, T_prime)

    N_Y = N_Y.astype(dtype= np.uint8)
    N_Cr = N_Cr.astype(dtype= np.uint8)
    N_Cb = N_Cb.astype(dtype= np.uint8)

    temp = cv2.merge((N_Y, N_Cr, N_Cb))
    N_I = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)
    fileName = 'Decompressed ' + str(file)
    path = "D:/Python projects/JPEGCompression/Decompressed Images/" + str(imageType) + "/" + str(level)
    cv2.imwrite(os.path.join(path, fileName), N_I)


# In[10]:


def Evaluate(file, level, imageType):
    inputPath = "D:/Python projects/JPEGCompression/" + str(imageType) + " Images"
    I = cv2.imread(os.path.join(inputPath, file))

    fileDecompress = 'Decompressed ' + str(file)
    path = "D:/Python projects/JPEGCompression/Decompressed Images/" + str(imageType) + "/" + str(level)
    I1 = cv2.imread(os.path.join(path, fileDecompress))

    err = np.sum((I.astype("float") - I1.astype("float")) ** 2)
    err /= float(I.shape[0] * I1.shape[1])

    return err


# In[12]:


def subSamplingRaito_422(Chroma, H, W):
    result = np.zeros([H, W])
    i = 0
    j = 0
    while((i < H) and (j < W)):
        if ((i + 2 < H) and (j + 4 < W)):
            result[i, j] = result[i, j +1] = np.round((Chroma[i, j] + Chroma[i, j + 1])/2)
            result[i, j + 2] = result[i, j + 3] = np.round((Chroma[i, j + 2] + Chroma[i, j + 3]) / 2)
            result[i + 1, j] = result[i + 1, j + 1] = np.round((Chroma[i + 1, j] + Chroma[i + 1, j + 1]) / 2)
            result[i + 1, j + 2] = result[i + 1, j + 3] = np.round((Chroma[i + 1, j + 2] + Chroma[i + 1, j + 3]) / 2)

        i = i + 2
        j = j + 4

    result = result.astype(dtype=np.uint8)
    return result


# In[13]: Main function:


def CompressNDecompress(file, level, imgType):
    # Define quality factors for luminance and chroma

    QLumi =  np.zeros([8, 8])
    QLumi = np.array([[16, 11, 10, 16, 24, 40, 52, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

    QChroma = np.zeros([8, 8])
    QChroma = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]])

    print("Compressing....")
    start = time.time()
    Q_Y, Q_Cr, Q_Cb, QL, QCm, T, T_prime = Compress_img(file, level, imgType, QLumi, QChroma)
    time_comp = time.time()
    print("Compression Time: ", np.round(time_comp - start, 1), " sec")

    print("Decompressing...")
    Decompress_img(file, level, imgType, Q_Y, Q_Cr, Q_Cb, QL, QCm, T, T_prime)
    time_decomp = time.time()

    print("Decompression Time: ", np.round(time_decomp - time_comp, 1), " sec")

    end = time.time()
    print("Total: ", np.round(end - start, 1), " sec")
    mse = Evaluate(file, level, imgType)
    print("MSE: ", np.round(mse, 4))
    print("---------------------------------------------------------")