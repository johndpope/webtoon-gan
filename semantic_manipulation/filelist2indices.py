import numpy as np
import pandas as pd
import glob
import argparse

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=str)  
    

    args = parser.parse_args()
    
    part = args.part
    neg_file_list = np.load(f'{part}_neg_filelist.npy')
    pos_file_list = np.load(f'{part}_pos_filelist.npy')
    

    idx = np.concatenate([
        neg_file_list, 
        pos_file_list, 
        # front
        ], axis=0)

    target = [1 for i in range(len(neg_file_list))] + [-1 for i in range(len(pos_file_list))] 

    total_file = glob.glob('../data/sketch256/raw_images/*/*/*.jpg')
    total_file = sorted([i.split('/')[-1] for i in total_file])
    print(len(total_file))
    

    target_df = pd.DataFrame({
        'file' : idx,
        'target' : target
    }).set_index('file')

    print(target_df.head(10))

    total_file = pd.DataFrame({'file': total_file, '#' : np.arange(len(total_file))}).set_index('file')
    target_df = target_df.join(total_file).dropna()

    print(target_df.shape)
    print(target_df['target'].value_counts())

    np.save(f'{part}_neg_indices.npy', target_df[target_df['target']==1]['#'].values)
    np.save(f'{part}_pos_indices.npy', target_df[target_df['target']==-1]['#'].values)

    print('Done')