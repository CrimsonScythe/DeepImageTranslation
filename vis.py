import matplotlib.pyplot as plt
import pickle
import torch
import pandas as pd
import tensorflow as tf
import numpy

torch.set_printoptions(edgeitems=10)
numpy.set_printoptions(edgeitems=10)

def vis(image, title):
    plt.imshow(image[0].detach().cpu().permute(1,2,0)[:,:,0],cmap='gray')
    plt.title(title)
    plt.show()

'''
Visualize images
'''
lst=['0','1000','2000','3000', '4000', '5000', '6000', '7000', '8000', '9000', '10000','11000', '12000', '13000', '14000', '15000', '16000', '17000', '18000', '19000', 'final']
for st in lst:
    with open(f'C:/Users/hasee/Documents/bachelor/logbook pictures/SIFA FINAL2/super/pred_mask_fake_b{st}.pickle', 'rb') as f:
        ''' images '''
        # vis(image, st)
        ''' masks '''
        image=pickle.load(f)
        image=image.detach().cpu()
        print(image.shape)
        soft=torch.nn.functional.softmax(image,dim=1)
        args=torch.argmax(soft.detach().cpu(), dim=1)
        args=torch.squeeze(args)
        ne=args.clone()
        for i in range(65):
            for j in range(65):
                ne[i][j]=image[0].permute(1,2,0)[i,j,args[i][j]]
                # ne[i][j]=soft[0].permute(1,2,0)[i,j,args[i][j]]
        plt.imshow(ne, cmap='gray')
        plt.title(st)
        plt.show()
        # plt.savefig(f'pred{st}.png')
        # print(st)
    
'''
Visualize Loss curves 
'''
# df=pd.read_csv('C:/Users/hasee/Documents/bachelor/logbook pictures/withskip/dtu/data2.csv')
# print(df.columns)
# for col in df.columns:
#     if str(col)=='Unnamed: 0' or str(col)=='epoch' or str(col)=='time' or str(col)=='g_b loss' or str(col)=='g_a loss' or str(col)=='d_a loss' or str(col)=='d_b loss':
#         continue
#     plt.plot(df['epoch'], df[col], label=str(col))  

# plt.ylim([0,5])
# plt.legend()
# plt.show()
