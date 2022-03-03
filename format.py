import numpy as np
import torch
import pickle
import scipy.io

n = 5 # Number of speakers
data = pickle.load(open('spmel/train.pkl',"rb"))
data_out = [['p225','p226','p227','p228','p400']] # Speakers
data_out.append([])
data_list = []
speakers_list = ['p225','p226','p227','p228','p400']
speakers_files = ['p225_003.npy','p226_005.npy','p227_008.npy','p228_019.npy','p400_006.npy']
for i in range(n):
  data_list.append(np.load('spmel/' + speakers_list[i] + '/' + speakers_files[i]))
  data_out[1].append(data[i][1])
data_out.append([])
for i in range(n):
  data_out[2].append(data_list[i])
data_out = np.array(data_out)
data_out = data_out.T
print(np.shape(data_out))
scipy.io.savemat('metadata_fromwavs_mat.mat',mdict={'metadata_fromwavs':data_out})
with open('metadata_fromwavs.pkl', 'wb') as handle: ## File change
    pickle.dump(data_out, handle) 