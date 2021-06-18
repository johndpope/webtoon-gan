import numpy as np
import glob 

left_select = np.load('lr_neg_indices.npy')
right_select = np.load('lr_pos_indices.npy')

total_file = glob.glob('../data/sketch256/raw_images/*/*/*.jpg')
total_file = np.array(sorted([i.split('/')[-1] for i in total_file]))

np.save('lr_neg_filelist.npy', total_file[left_select])
np.save('lr_pos_filelist.npy',total_file[right_select])