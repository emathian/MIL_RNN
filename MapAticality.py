import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import Image, display
import seaborn as sns
import PIL
import PIL.Image
import numpy as np
import os
import io
import umap
import matplotlib
import shutil
import argparse
parser = argparse.ArgumentParser(description='Map MIL Scores')
parser.add_argument('--prob_file', type=str, default='', help='path to probability.csv')
parser.add_argument('--path_tne_tiles', type=str, default='/data/gcs/lungNENomics/work/MathianE/Tiles_512_512_1802', help='path to tne tiles main folder')
parser.add_argument('--path_NL_tiles', type=str, default='/data/gcs/lungNENomics/work/MathianE/Tiles_512_512_NormalLungHENorm', help='path to normal lung tiles main folder')
parser.add_argument('--path_full_LNEN_slides', type=str, default='/data/gcs/lungNENomics/work/MathianE/FullSlidesToJpegHENormHighQuality/', help='path to full LNEN WSI main folder')
parser.add_argument('--path_full_NL_slides', type=str, default='/data/gcs/lungNENomics/work/MathianE/FullSlidesToJpegHENormHighQualityNormalLung/', help='path to full NL WSI main folder')
parser.add_argument('--outputdir', type=str, default='', help='path to output directory')

args = parser.parse_args()
pred_file = args.prob_file   #'EfficientNetb2_NormalTumor_TrainingInf/probability.csv'
df_pred_test = pd.read_csv(pred_file,  index_col='Sample')
print(df_pred_test.head(), '\n\n\n' , df_pred_test.columns, df_pred_test.index)
outputdir = args.outputdir
full_LNEN_WSI = args.path_full_LNEN_slides
full_NL_WSI = args.path_full_NL_slides
try:
    os.mkdir(outputdir)
except:
    print('ResMap already created ')
sample = []
x = []
y = []
for i in range(df_pred_test.shape[0]):
    filen = df_pred_test.iloc[i,0]
    sample_c = df_pred_test.index[i]
    x_c = int(filen.split('/')[-1].split('_')[1])
    try:
        y_c = int(filen.split('/')[-1].split('_')[-1].split('.')[0])
    except:
        if filen.split('/')[-1].split('_')[-1].split('.')[0].find('flip') != -1:
            y_c = int(filen.split('/')[-1].split('_')[-1].split('.')[0].split('flip')[0])
        else:
            y_c = int(filen.split('/')[-1].split('_')[-1].split('.')[0].split('rotated')[0])
    sample.append(sample_c)
    x.append(x_c)
    y.append(y_c)
df_pred_test['sample'] = sample
df_pred_test['x'] = x
df_pred_test['y'] = y
print('head  ', df_pred_test.head(), '\n\n')
sample_maxX_maxY = {}
path_main_TNE = args.path_tne_tiles
path_main_NL = args.path_NL_tiles
for sample in set(df_pred_test['sample']):
    if sample.find('TNE')!= -1:
        try:
            os.mkdir(os.path.join(outputdir, sample))
        except:
            print('sample_folder created')
        path_main = path_main_TNE
    elif sample.find('NL')!= -1:
        try:
            os.mkdir(os.path.join(outputdir, sample))
        except:
            print('sample_folder created')
        path_main = path_main_NL
    sample_folder = os.path.join(path_main, sample)
    xmax = 0
    ymax = 0
    for folder in os.listdir(sample_folder):
        tiles_p = os.path.join(path_main, sample, folder)
        for tiles_l in os.listdir(tiles_p):
            xmax_c = int(tiles_l.split('_')[1])
            ymax_c  = int(tiles_l.split('_')[2].split('.')[0])
            if xmax < xmax_c:
                xmax = xmax_c
            else:
                xmax = xmax
            if ymax < ymax_c:
                ymax = ymax_c
            else:
                ymax = ymax

    sample_maxX_maxY[sample] = [xmax, ymax]
for k in sample_maxX_maxY.keys():
    if k in list(df_pred_test['sample']):
        w =  tuple(sample_maxX_maxY[k])[0] + 924
        h = tuple(sample_maxX_maxY[k])[1] + 924
        seq = 924
        W = len(list(range(1, w, seq)))
        H = len(list(range(1, h, seq)))
        mat_prob_atypical =   np.zeros((W*10, H*10)) -1
        mat_prob_norm_atypical =   np.zeros((W*10, H*10)) -1
        df_test_pred_s = df_pred_test[df_pred_test['sample'] == k]
        min_p = df_test_pred_s['probability'].min()
        max_p =  df_test_pred_s['probability'].max()
        df_test_pred_s['prob_norm'] =  df_test_pred_s['probability'] - min_p / (max_p - min_p)
        #print(df_test_pred_s.head())
        for i in range(df_test_pred_s.shape[0]):
            x_ = df_test_pred_s.iloc[i,:]['x']
            y_ = df_test_pred_s.iloc[i,:]['y']
            mat_prob_atypical[x_ // 924 * 10 :x_ // 924 *10 + 10 ,  y_ // 924 * 10 :y_ // 924 * 10 + 10 ]= df_test_pred_s.iloc[i,2]
            mat_prob_norm_atypical[x_ // 924 * 10 :x_ // 924 *10 + 10 ,  y_ // 924 * 10 :y_ // 924 * 10 + 10 ]= df_test_pred_s.iloc[i,6]


        try:
            # Full WSI
            if k.find('TNE') != -1:
                get_full_img = full_LNEN_WSI + k + '.jpg'
                print('get_full_img  ', get_full_img)
            elif k.find('NL') != -1:
                get_full_img = full_NL_WSI + k + '.jpg'
                print('get_full_img  ', get_full_img)
            im = cv2.imread(get_full_img)
            fig=plt.figure(1,figsize=(15,15))
            plt.imshow(im.astype('uint8'))
            types =  list(df_pred_test[df_pred_test['sample'] == k].iloc[:,1])[0]
            if types == 1:
                typesN = 'Normal'
            elif types == 0:
                typesN = 'Tumoral'
            else:
                typesN = 'Normal'
            plt.title('WSI_{}_{}'.format(k,typesN))
            fig.savefig(os.path.join(outputdir, k,'WSI_{}_{}.png'.format(k, typesN)), dpi=fig.dpi)
            plt.close()
        except:
            print('WSI not available')



        try:
            # ATypical
            color_map = plt.cm.get_cmap('coolwarm')
            fig=plt.figure(1,figsize=(15,15))
            plt.matshow(mat_prob_atypical,  cmap=color_map,
                        interpolation='none',  fignum=1)
            mtitle = 'Normal tiles scores sample {} '.format(k)
            plt.title(mtitle)
            plt.colorbar()
            fig.savefig(os.path.join(outputdir, k,'Normality_tiles_map_{}.png'.format(k)), dpi=fig.dpi)
            plt.colorbar()
            plt.close()
        except:
            print('Fail atypucal map')

        try:
            # ATypical NORM
            color_map = plt.cm.get_cmap('coolwarm')
            fig=plt.figure(1,figsize=(15,15))
            plt.matshow(mat_prob_norm_atypical,  cmap=color_map,
                        interpolation='none',  fignum=1)
            mtitle = 'Normal tiles scores sample {} '.format(k)
            plt.title(mtitle)
            plt.colorbar()
            fig.savefig(os.path.join(outputdir, k,'Normality_tiles_map_norm_{}.png'.format(k)), dpi=fig.dpi)
            plt.colorbar()
            plt.close()
        except:
            print('Atypical map norm fail')











