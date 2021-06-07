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

pred_file = 'ResMILBs64_2/probability.csv'

df_pred_test = pd.read_csv(pred_file,  index_col='Sample')
print(df_pred_test.head(), '\n\n\n' , df_pred_test.columns, df_pred_test.index)
try:
    os.mkdir('ResMap64BS_training2')
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
print('heqad  ', df_pred_test.head(), '\n\n')
sample_maxX_maxY = {}
path_main_TNE = '/data/gcs/lungNENomics/work/MathianE/Tiles_512_512_1802'
for sample in set(df_pred_test['sample']):
    if sample.find('TNE')!= -1:
        try:
            os.mkdir(os.path.join('ResMap64BS_training2', sample))
        except:
            print('sample_folder created')
        sample_folder = os.path.join(path_main_TNE, sample)
        xmax = 0
        ymax = 0
        for folder in os.listdir(sample_folder):
            tiles_p = os.path.join(path_main_TNE, sample, folder)
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
        mat_prob_atypical =   np.zeros((W*10, H*10))
        mat_prob_norm_atypical =   np.zeros((W*10, H*10))
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
            get_full_img = '/data/gcs/lungNENomics/work/MathianE/FullSlidesToJpegHENormHighQuality/' + k + '.jpg'
            print('get_full_img  ', get_full_img)
            im = cv2.imread(get_full_img)
            fig=plt.figure(1,figsize=(15,15))
            plt.imshow(im.astype('uint8'))
            types =  list(df_pred_test[df_pred_test['sample'] == k].iloc[:,1])[0]
            if types == 1:
                typesN = 'Atypical'
            elif types == 0:
                typesN = 'Typical'
            else:
                typesN = 'Normal'
            plt.title('WSI_{}_{}'.format(k,typesN))
            fig.savefig(os.path.join('ResMap64BS_training2', k,'WSI_{}_{}.png'.format(k, typesN)), dpi=fig.dpi)
            plt.close()
        except:
            print('WSI not available')



        try:
            # ATypical
            color_map = plt.cm.get_cmap('coolwarm')
            fig=plt.figure(1,figsize=(15,15))
            plt.matshow(mat_prob_atypical,  cmap=color_map,
                        interpolation='none', vmin=0, vmax=1,  fignum=1)
            mtitle = 'Atypical tiles scores sample {} '.format(k)
            plt.title(mtitle)
            plt.colorbar()
            fig.savefig(os.path.join('ResMap64BS_training2', k,'Atypical_tiles_map_{}.png'.format(k)), dpi=fig.dpi)
            plt.colorbar()
            plt.close()
        except:
            print('Fail atypucal map')

        try:
            # ATypical NORM
            color_map = plt.cm.get_cmap('coolwarm')
            fig=plt.figure(1,figsize=(15,15))
            plt.matshow(mat_prob_norm_atypical,  cmap=color_map,
                        interpolation='none', vmin=0, vmax=1,  fignum=1)
            mtitle = 'Atypical tiles scores sample {} '.format(k)
            plt.title(mtitle)
            plt.colorbar()
            fig.savefig(os.path.join('ResMap64BS_training2', k,'Atypical_tiles_map_norm_{}.png'.format(k)), dpi=fig.dpi)
            plt.colorbar()
            plt.close()
        except:
            print('Atypical map norm fail')














#         df_test_pred_s = df_pred_test[df_pred_test['sample'] == k]
#         df_test_pred_s = df_test_pred_s.sort_values(by= 2)
#         df_test_pred_s.iloc[:10,0]
#         try:
#             os.mkdir(os.path.join('res'+'_'+split, k, 'best_scores_normal'))
#         except:
#             print('folder created')
#         try:
#             os.mkdir(os.path.join('res'+'_'+split, k, 'best_scores_atypical'))
#         except:
#             print('folder created')
#         try:
#             os.mkdir(os.path.join('res'+'_'+split, k, 'best_scores_typical'))
#         except:
#             print('folder created')
#         for ele in df_test_pred_s.iloc[:10,0]:
#             if ele.find('home')!= -1:
#                 pname = ele[2:-1]
#                 print(pname)
#                 tname = ele.split('/')[-1][:-1]
#                 print('tname ', tname)
#                 shutil.copy(pname, os.path.join('res'+'_'+split, k, 'best_scores_normal',tname ))
#             else:
#                 if ele.find('Normal')!= -1:
#                     b  = 38
#                 else :
#                      b= 37
#                 pname = '/data/gcs/lungNENomics/work/MathianE'+ ele[b:-1]
#                 print(pname)
#                 tname = '/data/gcs/lungNENomics/work/MathianE'+ ele[b:-1].split('/')[-1][:-1]
#                 print(os.path.join('res'+'_'+split, k, 'best_scores_normal',tname ), '\n')
#                 shutil.copy(pname, os.path.join('res'+'_'+split, k, 'best_scores_normal',tname ))


#         df_test_pred_s = df_pred_test[df_pred_test['sample'] == k]
#         df_test_pred_s = df_test_pred_s.sort_values(by= 3)
#         df_test_pred_s.iloc[:10,0]

#         for ele in df_test_pred_s.iloc[:10,0]:
#             print('ele ', ele)
#             if ele.find('home')!= -1:
#                 pname = ele[2:-1]
#                 print('pname ', pname)
#                 tname =  ele.split('/')[-1][:-1]
#                 print('tname ', tname)
#                 shutil.copy(pname, os.path.join('res'+'_'+split, k, 'best_scores_typical',tname ))
#             else:
#                 if ele.find('Normal')!= -1:
#                     b  = 38
#                 else :
#                      b= 37
#                 pname = '/data/gcs/lungNENomics/work/MathianE'+ ele[b:-1]
#                 print('\n sample ',k,pname)
#                 tname = '/data/gcs/lungNENomics/work/MathianE'+ ele[b:-1].split('/')[-1][:-1]
#                 print(os.path.join('res'+'_'+split, k, 'best_scores_typical',tname ), '\n')
#                 shutil.copy(pname, os.path.join('res'+'_'+split, k, 'best_scores_typical',tname ))
#         df_test_pred_s = df_pred_test[df_pred_test['sample'] == k]
#         df_test_pred_s = df_test_pred_s.sort_values(by= 4)
#         df_test_pred_s.iloc[:10,0]
#         for ele in df_test_pred_s.iloc[:10,0]:
#             print('ATYPICAL TILES ')
#             if ele.find('home')!= -1:
#                 pname = ele[2:-1]
#                 tname =  ele.split('/')[-1][:-1]
#                 shutil.copy(pname, os.path.join('res'+'_'+split, k, 'best_scores_atypical',tname ))
#             else:
#                 if ele.find('Normal')!= -1:
#                     b  = 38
#                 else:
#                     b= 37
#                 pname = '/data/gcs/lungNENomics/work/MathianE'+ ele[b:-1]
#                 tname = '/data/gcs/lungNENomics/work/MathianE'+ ele[b:-1].split('/')[-1][:-1]
#                 shutil.copy(pname, os.path.join('res'+'_'+split, k, 'best_scores_atypical',tname ))
