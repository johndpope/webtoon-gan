import shutil
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--distribution", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--folder", type=str, default="./output")
parser.add_argument("--select", type=str, default="0,1")

args = parser.parse_args()


## FOLDERS SHOULD EXISTS BEFOREHAND !!

if __name__ == '__main__':
    ## for distribution
    ## all images in a single folder (output) --> 10 batch folders (batch0, batch1, ..., batch9)
    if args.distribution:
        source = args.folder
        files = os.listdir(source)

        batches = []
        INTERVAL = 30000
        for i in range(9):
            batches.append(files[INTERVAL*i:INTERVAL*(i+1)])
        batches.append(files[INTERVAL*(i+1):])

        for j, batch in enumerate(batches):
            dest = f'./sketch/batch{j}/'
            for f in batch:
                f = os.path.join(source, f)
                shutil.move(f, dest)
    # selected batches to train/test/val set
    else:
        selected = [int(s.strip()) for s in args.select.split(",")]

        all_files = []
        for i in selected:
            source = f'./sketch/batch{i}/'
            files = os.listdir(source)
            for j in range(len(files)):
                all_files.append(os.path.join(source, files[j]))

        k = args.select.replace(',', '')
        folder = f'./sketch-batch{k}'

        for f in all_files[:500]:
            shutil.copy(f, os.path.join(folder, 'test', 'images'))
        for f in all_files[500:1000]:
            shutil.copy(f, os.path.join(folder, 'val', 'images'))
        for f in all_files[1000:]:
            shutil.copy(f, os.path.join(folder, 'train', 'images'))
