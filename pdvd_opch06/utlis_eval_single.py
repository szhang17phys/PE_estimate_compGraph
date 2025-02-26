#utlities================================
import os
import re
import sys
import time

import tensorflow as tf

import pickle as pk
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D # draw 3d figure---

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
#from tensorflow.python.framework.graph_util import convert_variables_to_constants

#Suggsted by chatGPT---
from tensorflow.compat.v1.graph_util import convert_variables_to_constants


from pylab import *
from networks import *






#=========================================================
pos_X = []
pos_Y = []
pos_Z = []
value_bias = [] #used to store the biased values---
value_emul = [] #used to store emul (graphModule) values---

XA = []
#Added by Shu, 20231123---
#Here opCh refers to all opch of protodunevd_v4---
opCh = []









#Range of cryostat:=======================================
#X range: [ -375 ,  415 ]
#Y range: [ -427.4 ,  427.4 ]
#Z range: [ -277.75 ,  577.05 ]
#I will set the step length as 10cm for each axis---


verX = list(range(-375, 417, 10))
verY = list(range(-425, 427, 10))
verZ = list(range(-275, 577, 10))

verNum = len(verX) * len(verY) * len(verZ)

print(" verX: ", verX)
print("\n verY: ", verY)
print("\n verZ: ", verZ)
print("Step size along X: ", len(verX))
print("Step size along Y: ", len(verY))
print("Step size along Z: ", len(verZ))
print("\nTotal Num of vertex: ", verNum)
#=========================================================







#dim_pos: vertex dimension (3)============================
def get_data(dim_pos):    
    inputs  = np.zeros(shape=(verNum, dim_pos))

    labels = 0
    for valueX in verX:
        for valueY in verY:
            for valueZ in verZ:
                inputs[labels, 0] = valueX
                inputs[labels, 1] = valueY
                inputs[labels, 2] = valueZ

                labels += 1


    return inputs

#pos = get_data(3)
#print(pos)
#======================================================










#Core function========================================= 
#pos is vertex 3D positions---
#modpath: module path---
#label: opch---      
def eval(pos, modpath, label):

    dim_pdr = 40 #num of opch---
    mtier = 0 #Not importan---

    #Import model!---
    print('Loading protodunehd_v4 40 opch net...')
    model = model_protodunevd_v4(dim_pdr)

    weight = modpath+'best_model.h5'
    if os.path.isfile(weight):
        print('Loading weights...')
        model.load_weights(weight)
    else:
        print('Err: no weight found!')
        return
        
    print('Predicting...')
    tstart = time.time()
    pre = model.predict({'pos_x': pos[:,0], 'pos_y': pos[:,1], 'pos_z': pos[:,2]})
    print('\n')
    print( '\nFinish evaluation in '+str(time.time()-tstart)+'s.')



    #This is in fact the whole space---
    cut_z = (pos[:,2] > -1000) & (pos[:,2] < 1000)
    cut_y = (pos[:,1] > -1000) & (pos[:,1] < 1000)
    coor_x = pos[:,0][cut_z & cut_y]
    emul_x = pre[cut_z & cut_y]
   
    num_x = len(coor_x)
    true_x_s = np.zeros(shape=(num_x, dim_pdr))
    emul_x_s = np.zeros(shape=(num_x, dim_pdr))

    for index, op in zip(range(0, dim_pdr), range(0, dim_pdr)):
        emul_x_s[:,index] = emul_x[:,op]
    


    #OpCh label---
    if label < 10:
        opName = f'opch0{label}'
    else:
        opName = f'opch{label}'


    #Keep data of certain opCh and keep as txt files----------
    for nums in range(0, verNum): 
        pos_X.append(pos[nums, 0])
        pos_Y.append(pos[nums, 1])
        pos_Z.append(pos[nums, 2])
        value_emul.append(emul_x_s[nums, label]*1000000)

    

   #keep x, y, z & values in  txt files---
    with open('./results/{}_xPos.txt'.format(opName), 'w') as filehandle:
        for listitem in pos_X:
            filehandle.write('%f\n' % listitem)

    with open('./results/{}_yPos.txt'.format(opName), 'w') as filehandle:
        for listitem in pos_Y:
            filehandle.write('%f\n' % listitem)

    with open('./results/{}_zPos.txt'.format(opName), 'w') as filehandle:
        for listitem in pos_Z:
            filehandle.write('%f\n' % listitem)


    with open('./results/{}_emulValues.txt'.format(opName), 'w') as filehandle:
        for listitem in value_emul:
            filehandle.write('%d\n' % listitem)

    print("\nsize of value_emul: ", len(value_emul))
    print("Length of pos_X   : ", len(pos_X))
    print("Num of vertex     : ", verNum)
    print("---------------------------------")
#====================================================== 








#Execution CODE=========================================
vertex = get_data(3)
module_path = './output_2048b_10000e0-mod/'

#Just change opch each time------
opch_label = 6

eval(vertex, module_path, opch_label)
#=====================================================
