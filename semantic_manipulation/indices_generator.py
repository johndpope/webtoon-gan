import numpy as np
import glob 
import pandas as pd 

total_file = glob.glob('../data/sketch256/raw_images/*/*/*.jpg')
total_file = sorted([i.split('/')[-1] for i in total_file])
left_right = pd.read_csv('../semantic_manipulation/left_right.csv', index_col='file')

total_file = pd.DataFrame({'file': total_file, '#' : np.arange(len(total_file))}).set_index('file')
left_right = left_right.join(total_file).dropna()
print(left_right['left_right'].value_counts())

np.save('lr_neg_indices.npy', left_right[left_right['left_right']==-1]['#'].values)
np.save('lr_pos_indices.npy', left_right[left_right['left_right']==1]['#'].values)

print(np.load('lr_neg_indices.npy'))