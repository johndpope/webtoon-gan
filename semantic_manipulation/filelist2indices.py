import numpy as np
import pandas as pd
import glob

np.random.seed(0)


left_select = np.load('left_file_index_npy')
right_select = np.load('right_file_index_npy')
# front = np.load('./right_training_front.npy')

idx = np.concatenate([
    left_select, 
    right_select, 
    # front
    ], axis=0)
lr = [1 for i in range(len(left_select))] + [-1 for i in range(len(right_select))] # + np.random.choice([1, -1], size=len(front)).tolist()

total_file = glob.glob('../data/sketch256/raw_images/*/*/*.jpg')
total_file = sorted([i.split('/')[-1] for i in total_file])

left_right = pd.DataFrame({
    'file':idx,
    'left_right':lr
}).set_index('file')



total_file = pd.DataFrame({'file': total_file, '#' : np.arange(len(total_file))}).set_index('file')
left_right = left_right.join(total_file).dropna()

print(left_right['left_right'].value_counts())

np.save('lr_neg_indices.npy', left_right[left_right['left_right']==-1]['#'].values)
np.save('lr_pos_indices.npy', left_right[left_right['left_right']==1]['#'].values)

print('Done')