"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import datetime
import random, os
import argparse
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torchvision import transforms, utils
from training.model import Generator, Encoder
from training.dataset import MultiResolutionDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm
from sklearn import svm, preprocessing
import glob
import pandas as pd

import cv2


seed = 0
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


class Sampler_with_index(data.Sampler):
    def __init__(self, indexes):
        self.indexes = indexes

    def __iter__(self):
        return (i for i in self.indexes)

    def __len__(self):
        return len(self.indexes)


class Model(nn.Module):
    def __init__(self, device="cuda"):
        super(Model, self).__init__()

        self.g_ema = Generator(
            train_args.size,
            train_args.mapping_layer_num,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            lr_mul=train_args.lr_mul,
            channel_multiplier=train_args.channel_multiplier,
            normalize_mode=train_args.normalize_mode,
            small_generator=train_args.small_generator,
        )
        self.e_ema = Encoder(
            train_args.size,
            train_args.latent_channel_size,
            train_args.latent_spatial_size,
            channel_multiplier=train_args.channel_multiplier,
        )

    def forward(self, images, mode, direction_vector=None, mask=None):

        if mode == "project_latent":
            fake_stylecode = self.e_ema(images)
            return fake_stylecode
        elif mode == "transform_to_other_part":
            get_cuda_device = images.get_device()

            fake_stylecode = self.e_ema(images)

            direction_vector = (
                direction_vector.to(get_cuda_device)
                .unsqueeze(0)
                .repeat(images.shape[0], 1, 1, 1)
            )

            transformed_images = []

            for distance in range(-20, 21, 5):
                modified_stylecode = fake_stylecode + distance * direction_vector
                transformed_image, _ = self.g_ema(
                    modified_stylecode, input_is_stylecode=True
                )
                transformed_images.append(transformed_image)

            return torch.cat(transformed_images)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


if __name__ == "__main__":
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--save_dir", type=str, default="expr/semantic_manipulation")
    parser.add_argument("--LMDB", type=str, default="data/sketch256/LMDB")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--part", type=str, default="lr")  
    parser.add_argument("--svm_train_iter", type=int, default=1000)
    parser.add_argument("--augmentation", type=str, default='default') # flip
    args = parser.parse_args()

    args.train_lmdb = f"{args.LMDB}_train"
    args.val_lmdb = f"{args.LMDB}_val"
    args.test_lmdb = f"{args.LMDB}_test"

    model_name = args.ckpt.split("/")[-1].replace(".pt", "")
    part = args.part
    
    os.makedirs(os.path.join(args.save_dir, part), exist_ok=True)
    args.inverted_npy = os.path.join(args.save_dir, f"{model_name}_inverted.npy")
    args.boundary = os.path.join(args.save_dir, args.part, f"{model_name}_{part}_boundary_{args.augmentation}_{args.svm_train_iter}.npy")
    print(args)

    ckpt = torch.load(args.ckpt)

    train_args = ckpt["train_args"]  # load arguments!

    model = Model()
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model = model.to(device)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    # Dataset Load
    train_dataset = MultiResolutionDataset(args.train_lmdb, transform, train_args.size)
    val_dataset = MultiResolutionDataset(args.val_lmdb, transform, train_args.size)
    test_dataset = MultiResolutionDataset(args.test_lmdb, transform, train_args.size)
    dataset = data.ConcatDataset([train_dataset, test_dataset, val_dataset])
    
    

    with torch.no_grad():
        # if inverted latent does not exist, save inverted latent as npy format
        if os.path.isfile(args.inverted_npy):
            print("inverted_npy exists!")
            latent_codes = np.load(args.inverted_npy)
        else:
            loader = data.DataLoader(
                dataset,
                batch_size=args.batch,
                sampler=data_sampler(dataset, shuffle=False),
                num_workers=args.num_workers,
                pin_memory=True,
            )
            latent_codes_enc = []
            for images in tqdm(loader):
                images = images.to(device)
                project_latent = model(images, "project_latent")
                latent_codes_enc.append(project_latent.cpu().numpy())
            latent_codes = np.concatenate(latent_codes_enc, axis=0)
            np.save(f"{args.inverted_npy}", latent_codes)

        if args.augmentation == 'flip':

            if os.path.isfile(args.inverted_npy.replace('inverted','flip_inverted')):
                print("flipped_inverted_npy exists!")
                latent_codes_flip = np.load(args.inverted_npy.replace('inverted','flip_inverted'))
            else:
                loader = data.DataLoader(
                    dataset,
                    batch_size=args.batch,
                    sampler=data_sampler(dataset, shuffle=False),
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                latent_codes_enc_flip = []
                for images in tqdm(loader):
                    reverse_index = np.arange(255, -1, -1)
                    images = images[:,:,reverse_index,:].to(device)
                    project_latent = model(images, "project_latent")
                    latent_codes_enc_flip.append(project_latent.cpu().numpy())
                latent_codes_flip = np.concatenate(latent_codes_enc_flip, axis=0)
                np.save(f"{args.inverted_npy.replace('inverted','flip_inverted')}", latent_codes_flip)
                
            latent_codes_aug = latent_codes_flip.reshape(len(latent_codes_flip), -1)

        if args.augmentation == 'aug':
            if os.path.isfile(args.inverted_npy.replace('inverted','aug_inverted')):
                print("aug_inverted_npy exists!")
                latent_codes_aug = np.load(args.inverted_npy.replace('inverted','aug_inverted'))
            else:
                loader = data.DataLoader(
                    dataset,
                    batch_size=args.batch,
                    sampler=data_sampler(dataset, shuffle=False),
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                transform = A.Compose([
                    A.RandomSizedCrop(width=256, height=256, min_max_height=[200, 256], w2h_ratio=1.0),
                    # ToTensorV2()
                ])

                latent_codes_enc_aug = []
                for images in tqdm(loader):
                    for idx in range(images.shape[0]):
                        images[idx] = torch.from_numpy(transform(image=images[idx].numpy().transpose(1, 2, 0))['image'].transpose(2, 0, 1))

                    images = images.to(device)
                    project_latent = model(images, "project_latent")
                    latent_codes_enc_aug.append(project_latent.cpu().numpy())
                latent_codes_aug = np.concatenate(latent_codes_enc_aug, axis=0)
                
                np.save(f"{args.inverted_npy.replace('inverted','aug_inverted')}", latent_codes_aug)
            
            if latent_codes_aug.shape[-1] == 3:
                latent_codes_aug == torch.from_numpy(latent_codes_aug.numpy().transpose(2, 0, 1))

            latent_codes_aug = latent_codes_aug.reshape(len(latent_codes_aug), -1)

                

        latent_code_shape = latent_codes[0].shape
        print(f"latent_code_shape {latent_code_shape}")

        # flatten latent: Nx8x8x64 -> N x 8*8*64
        latent_codes = latent_codes.reshape(len(latent_codes), -1)

        # get boundary
        

        if os.path.isfile(args.boundary):
            print(f"{part} boundary exists!")
            boundary_infos = torch.load(args.boundary)
            boundary = torch.Tensor(boundary_infos["boundary"])
            part1_indexes = boundary_infos["part1_indexes"]
            part2_indexes = boundary_infos["part2_indexes"]

        else:
            part1_indexes = np.load(
                f"semantic_manipulation/{part}_pos_indices.npy"
            ).astype(int)
            part2_indexes = np.load(
                f"semantic_manipulation/{part}_neg_indices.npy"
            ).astype(int)
                        
            # get boundary using two parts.
            testset_ratio = 0.1
            np.random.shuffle(part1_indexes)
            np.random.shuffle(part2_indexes)

            positive_len = len(part1_indexes)
            negative_len = len(part2_indexes)



            positive_train = latent_codes[
                part1_indexes[int(positive_len * testset_ratio) :]
            ]

            positive_val = latent_codes[
                part1_indexes[: int(positive_len * testset_ratio)]
            ]

            negative_train = latent_codes[
                part2_indexes[int(negative_len * testset_ratio) :]
            ]
            negative_val = latent_codes[
                part2_indexes[: int(negative_len * testset_ratio)]
            ]

            if args.augmentation != 'default':
                positive_train = np.concatenate([
                    positive_train, 
                    latent_codes_aug[part2_indexes if part=='lr' and args.augmentation=='flip' else part1_indexes]
                ], axis=0)
                
                negative_train = np.concatenate([
                    negative_train, 
                    latent_codes_aug[part1_indexes if part=='lr' and args.augmentation=='flip' else part2_indexes]
                ], axis=0)



            # Training set.
            train_data = np.concatenate([positive_train, negative_train], axis=0)
            
            scaler = preprocessing.MinMaxScaler()
            train_data = scaler.fit_transform(train_data)
            
            train_label = np.concatenate(
                [
                    np.ones(len(positive_train), dtype=np.int),
                    np.zeros(len(negative_train), dtype=np.int),
                ],
                axis=0,
            )



            # Validation set.
            val_data = np.concatenate([positive_val, negative_val], axis=0)

            val_data = scaler.transform(val_data)
            val_label = np.concatenate(
                [
                    np.ones(len(positive_val), dtype=np.int),
                    np.zeros(len(negative_val), dtype=np.int),
                ],
                axis=0,
            )

            print(
                f"positive_train: {len(positive_train)}, \
                negative_train:{len(negative_train)}, \
                positive_val:{len(positive_val)}, \
                negative_val:{len(negative_val)}"
            )

            print(f"Training boundary. {datetime.datetime.now()}")
            
            clf = svm.SVC(kernel="linear", max_iter=args.svm_train_iter, C=0.0001, decision_function_shape='ovo')
            classifier = clf.fit(train_data, train_label)
            print(f"Finish training. {datetime.datetime.now()}")

            print(f"validate boundary.")
            val_prediction = classifier.predict(val_data)
            
            correct_num = np.sum(val_label == val_prediction)

            print(
                f"Accuracy for validation set: "
                f"{correct_num} / {len(val_data)} = "
                f"{correct_num / (len(val_data)):.6f}"
            )

            print("classifier.coef_.shape", classifier.coef_.shape)
            boundary = classifier.coef_.reshape(1, -1).astype(np.float32)
            boundary = scaler.inverse_transform(boundary)
            boundary = boundary / np.linalg.norm(boundary)
            # print(train_data.shape)
            # print(boundary.shape)
            
            boundary = boundary.reshape(latent_code_shape)
            
            print("boundary.shape", boundary.shape)
            
            boundary = torch.from_numpy(boundary).float()

            torch.save(
                {
                    "boundary": boundary,
                    "part1_indexes": part1_indexes,
                    "part2_indexes": part2_indexes,
                },
                args.boundary,
            )

        if True:
            part1_loader = data.DataLoader(
                dataset,
                batch_size=args.batch,
                sampler=Sampler_with_index(part1_indexes[:500]),
            )
            part1_loader = sample_data(part1_loader)
            part1_img = next(part1_loader).to(device)
            part2_loader = data.DataLoader(
                dataset,
                batch_size=args.batch,
                sampler=Sampler_with_index(part2_indexes[:500]),
            )

            part2_loader = sample_data(part2_loader)
            # next(part2_loader)
            part2_img = next(part2_loader).to(device)

            # heavy makup
            # lw, lh = latent_code_shape
            # ref_p_x, ref_p_y, width, height = 0, 0, lw, lh

            # mask = (ref_p_x, ref_p_y, width, height)
            mask = torch.zeros(part2_img.shape[0], 64, 8, 8)
            # mask[:, :, ref_p_y : ref_p_y + height, ref_p_x : ref_p_x + width] = 1
            mask[:, :, : ] = 1

            part2_to_part1 = model(
                part2_img, "transform_to_other_part", boundary, mask=mask
            )

            utils.save_image(
                torch.cat([part2_img, part2_to_part1]),
                f"{args.save_dir}/{part}/{model_name}_others_to_{part}_{args.augmentation}_{args.svm_train_iter}_1.png",
                nrow=args.batch,
                normalize=True,
                range=(-1, 1),
            )

            part1_to_part2 = model(
                part1_img, "transform_to_other_part", boundary, mask=mask
            )

            utils.save_image(
                torch.cat([part1_img, part1_to_part2]),
                f"{args.save_dir}/{part}/{model_name}_others_to_{part}_{args.augmentation}_{args.svm_train_iter}_2.png",
                nrow=args.batch,
                normalize=True,
                range=(-1, 1),
            )

