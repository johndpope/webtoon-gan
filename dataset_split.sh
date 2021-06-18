ls -1 | head -280000 | xargs -i mv "{}" ../data/sketch256_full/raw_images/train/images/
ls -1 | head -10000 | xargs -i mv "{}" ../data/sketch256_full/raw_images/test/images/
ls -1 | head -10000 | xargs -i mv "{}" ../data/sketch256_full/raw_images/val/images/