import numpy as np
from skimage import data, color 
from skimage.transform import resize
from skimage.io import imsave
import glob
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--early_stop", default=False, type=lambda x: (str(x).lower() == 'true'))
parser.add_argument("--replace_from", type=str, default="batch0")
parser.add_argument("--replace_to", type=str, default="rescale")
parser.add_argument("--scale", type=int, default=256)

args = parser.parse_args()

# ./sketch/batch0/1.jpg --> ./sketch/rescale/1.jpg

SCALE = args.scale
REPLACE_FROM = args.replace_from
REPLACE_TO = args.replace_to
FOLDER = './sketch/' + REPLACE_FROM + '/*.*'

if __name__ == '__main__':
    img_lst = glob.glob(FOLDER)
    for idx, name in enumerate(img_lst):
        if args.early_stop:
            if idx > 0 and idx % 1500 == 0:
                exit()
        img = Image.open(name)
        img = img.resize((SCALE, SCALE), Image.ANTIALIAS)
        img.save(name.replace(REPLACE_FROM, REPLACE_TO))
        if idx%5000==0:
            print(idx)
