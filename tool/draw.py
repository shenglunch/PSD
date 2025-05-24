import math
import numpy as np
import matplotlib
import tkinter as tk
import tkinter.filedialog as f
import torch

from PIL import Image, ImageTk

import matplotlib.pyplot as plt

def draw(input, output, h, w):
    print('gt', input['target'][:,:,h,w])
    print('d0', output['depth'][0][:,:,h,w], output['depth'][0][:,:,h,w]-input['target'][:,:,h,w])
    print('d1', output['depth'][1][:,:,h,w], output['depth'][1][:,:,h,w]-input['target'][:,:,h,w])
    print('d2', output['depth'][2][:,:,h,w], output['depth'][2][:,:,h,w]-input['target'][:,:,h,w])
    print('d3', output['depth'][3][:,:,h,w], output['depth'][3][:,:,h,w]-input['target'][:,:,h,w])
    print('d4', output['depth'][4][:,:,h,w], output['depth'][4][:,:,h,w]-input['target'][:,:,h,w])

    sim_0 = output['similarity'][0]
    sim_1 = sim_0 + output['similarity'][1]
    sim_2 = sim_1 + output['similarity'][2]
    sim_3 = sim_2 + output['similarity'][3]
    sim_4 = output['similarity'][4]
    # sim_1 = sim_1 / 2
    # sim_2 = sim_2 / 3
    # sim_3 = sim_3 / 4
    # sim_4 = sim_4 / 5

    print('sim0', sim_0[0, :, h, w])
    print('sim1', sim_1[0, :, h, w])
    print('sim2', sim_2[0, :, h, w])
    print('sim3', sim_3[0, :, h, w])
    print('sim4', sim_4[0, :, h, w])

    lo_1 = output['los'][1]
    lo_2 = output['los'][2]
    lo_3 = output['los'][3]
    lo_4 = output['los'][4]

    gl_1 = output['gls'][1]
    gl_2 = output['gls'][2]
    gl_3 = output['gls'][3]
    gl_4 = output['gls'][4]

    bins_0 = torch.linspace(2, 90, 32).float().cuda()
    bins_1 = bins_0.view(1, -1, 1, 1) + lo_1 + gl_1
    bins_2 = bins_1 + lo_2 + gl_2
    bins_3 = bins_2 + lo_3 + gl_3
    bins_4 = bins_3 + lo_4 + gl_4

    print('b0', bins_0)
    print('b1', bins_1[0, :, h, w])
    print('b2', bins_2[0, :, h, w])
    print('b3', bins_3[0, :, h, w])
    print('b4', bins_4[0, :, h, w])

    plt.figure()
    plt.subplot(211)
    plt.axvline(input['target'][0,0,h,w].cpu(), color='red', ls='-', label='Ground truth', linewidth=1)
    plt.axvline(output['depth'][0][0,0,h,w].cpu(), color='black', ls='-.', label='Initial result', linewidth=0.5)
    plt.axvline(output['depth'][4][0,0,h,w].cpu(), color='blue', ls='--', label='Final result', linewidth=0.5)
    plt.plot(bins_0.cpu(),             sim_0[0, :, h, w].cpu(), color='green', linestyle='-', label='Original distribution', linewidth=1, marker='o')
    # plt.plot(bins_1[0, :, h, w].cpu(), sim_0[0, :, h, w].cpu(), color='blue', linestyle='-', label='sim_1', linewidth=1)
    # plt.plot(bins_2[0, :, h, w].cpu(), sim_0[0, :, h, w].cpu(), color='green', linestyle='-', label='sim_2', linewidth=1)
    # plt.plot(bins_3[0, :, h, w].cpu(), sim_0[0, :, h, w].cpu(), color='black', linestyle='-', label='sim_3', linewidth=1)
    plt.plot(bins_4[0, :, h, w].cpu(), sim_4[0, :, h, w].cpu(), color='orange', linestyle='-', label='Adding offsets', linewidth=1, marker='v')

    plt.xticks(range(0, 91, 10))
    plt.ylabel('Logistic score')
    plt.xlabel('Depth guidance')
    plt.grid()
    plt.legend()
    

    # plt.subplot(212)
    # b = torch.linspace(1, 32, 32)
    # plt.plot(b, bins_0.cpu(), color='black', linestyle='-', label='Original depth guidance', linewidth=1)
    # plt.plot(b, bins_4[0, :, h, w].cpu(), color='red', linestyle='-', label='Adding offsets', linewidth=1)
    # # plt.plot(range(0, len(EPE_2)), EPE_2, color='blue', linestyle='-', label='EPE_2', linewidth=1)
    # # plt.plot(range(0, len(EPE_3)), EPE_3, color='green', linestyle='-', label='EPE_2', linewidth=1)
    # plt.xticks(range(1, 34, 2))
    # plt.xlabel('idx')
    # plt.ylabel('EPE')
    # plt.grid()
    # plt.legend()

    plt.savefig('./curve.png', bbox_inches = 'tight', pad_inches = 0, dpi=192)

# plt.show()

