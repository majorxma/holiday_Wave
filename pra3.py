import numpy as np
from scipy.fftpack import fft,ifft
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
import random
 
mpl.rcParams['font.sans-serif'] = ['SimHei']   #显示中文
mpl.rcParams['axes.unicode_minus']=False       #显示负号
 
x=np.linspace(0,1,200)      
print(x)

# put the original 200 data to y.
#y=7*np.sin(2*np.pi*200*x) + 5*np.sin(2*np.pi*400*x)+3*np.sin(2*np.pi*600*x) + random.choice(range(0,9)) * 10
y = [490, 477, 467, 458, 450, 442, 433, 426, 419, 413, 411, 428, 445, 441, 434, 436, 446, 442, 427, 414, 402,
         391, 381, 372, 366, 363, 363, 364, 366, 372, 382, 397, 414, 430, 444, 460, 481, 502, 522, 539, 551, 561,
         567, 569, 568, 566, 570, 576, 578, 574, 565, 553, 541, 529, 519, 507, 496, 486, 494, 528, 551, 563, 576,
         596, 612, 624, 631, 636, 639, 640, 640, 638, 635, 633, 630, 625, 620, 615, 609, 603, 597, 590, 584, 578,
         571, 559, 541, 529, 524, 511, 486, 454, 422, 394, 372, 348, 340, 335, 334, 332, 332, 332, 332, 332, 333,
         336, 339, 341, 344, 349, 355, 360, 366, 372, 383, 396, 408, 419, 432, 448, 463, 473, 482, 493, 511, 530,
         551, 568, 580, 595, 597, 597, 595, 593, 598, 606, 619, 632, 642, 653, 659, 658, 653, 645, 640, 641, 643,
         650, 656, 659, 659, 655, 649, 640, 632, 626, 621, 614, 603, 590, 575, 564, 550, 530, 519, 507, 495, 484,
         472, 462, 452, 445, 437, 430, 423, 417, 423, 442, 445, 435, 423, 422, 431, 436, 428, 413, 401, 390, 381,
         373, 367, 363, 364, 365, 367, 371, 378, 396, 411, 428]

plt.figure()
plt.plot(x,y)   
plt.title('原始波形')
 
plt.figure()
plt.plot(x[0:200],y[0:200])   
plt.title('原始部分波形（前200组样本）')
plt.show()
#################################################
#       get the picture of the original         #
#################################################

fft_y=fft(y)                          #快速傅里叶变换
print(len(fft_y))
print(fft_y[0:5])

#################################################
#       show                                    #
#################################################

N=200
x = np.arange(N)           # 频率个数
 
abs_y=np.abs(fft_y)                # 取复数的绝对值，即复数的模(双边频谱)
angle_y=np.angle(fft_y)              #取复数的角度

'''
plt.figure()
plt.plot(x,abs_y)   
plt.title('双边振幅谱（未归一化）')
 
plt.figure()
plt.plot(x,angle_y)   
plt.title('双边相位谱（未归一化）')
plt.show()
'''
normalization_y=abs_y/N            #归一化处理（双边频谱）

for i in range(int(N/2)):
    normalization_y[i] = normalization_y[int(N) - i - 1]
plt.figure()
plt.plot(x,normalization_y,'g')
plt.title('双边频谱(归一化)',fontsize=9,color='green')
plt.show()

half_x = x[range(int(N/2))]                                  #取一半区间
normalization_half_y = normalization_y[range(int(N/2))]      #由于对称性，只取一半区间（单边频谱）
plt.figure()
plt.plot(half_x,normalization_half_y,'b')
plt.title('单边频谱(归一化)',fontsize=9,color='blue')
plt.show()