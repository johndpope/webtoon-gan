"""
StyleMapGAN
Copyright (c) 2021-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import flask
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import base64
import os
import secrets
import argparse
import json
from PIL import Image
import re
from typing import List, Optional

import click
import dnnlib

######
import torch
from torch import nn
from training.model import Generator, Encoder
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import io


import legacy
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(
    __name__,
    template_folder="demo/templates",
    static_url_path="/demo/static",
    static_folder="demo/static",
)

app.config["MAX_CONTENT_LENGTH"] = 10000000  # allow 10 MB post

# for 1 gpu only.
class Model(nn.Module):
    def __init__(self, train_args):
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
        self.device = device

    def forward(self, original_image, references, masks, shift_values):

        combined = torch.cat([original_image, references], dim=0)

        ws = self.e_ema(combined)
        original_stylemap, reference_stylemaps = torch.split(
            ws, [1, len(ws) - 1], dim=0
        )

        mixed = self.g_ema(
            [original_stylemap, reference_stylemaps],
            input_is_stylecode=True,
            mix_space="demo",
            mask=[masks, shift_values, args['interpolation_step']],
        )[0]

        return mixed

@app.route("/")
def index():
    image_paths = []
    return render_template(
        "index.html",
        canvas_size=canvas_size,
        base_path=base_path,
        image_paths=list(os.walk(base_path)),
    )

@app.route("/single")
def single():
    image_paths = []
    return render_template(
        "single.html",
        canvas_size=canvas_size,
        base_path=base_path,
        image_paths=list(os.walk(base_path)),
    )


@app.route("/single2")
def single2():
    image_paths = []
    return render_template(
        "single2.html",
        canvas_size=canvas_size,
        base_path=base_path,
        image_paths=list(os.walk(base_path)),
    )

# "#010FFF" -> (1, 15, 255)
def hex2val(hex):
    if len(hex) != 7:
        raise Exception("invalid hex")
    val = int(hex[1:], 16)
    return np.array([val >> 16, (val >> 8) & 255, val & 255])


@torch.no_grad()
def my_morphed_images(original, references, masks, shift_values, interpolation=8, save_dir=None):
    original_path = original.split('?')[0] if 'demo' in original else base_path + original
    original_image = Image.open(original_path)
    reference_images = []

    for ref in references:
        ref_path = ref.split('?')[0] if 'demo' in ref else base_path + ref
        reference_image =  TF.to_tensor(Image.open(ref_path).resize((canvas_size, canvas_size)))
        if reference_image.ndim == 2 :
            reference_image = reference_image.unsqueeze(0)
        if reference_image.shape[0] == 1 :
            reference_image = reference_image.repeat(3, 1, 1)

        reference_images.append(reference_image)
        

    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(
        original_image, size=(canvas_size, canvas_size)
    )
    original_image = (original_image - 0.5) * 2

    reference_images = torch.stack(reference_images)
    reference_images = F.interpolate(
        reference_images, size=(canvas_size, canvas_size)
    )
    reference_images = (reference_images - 0.5) * 2

    if original_image.shape[1] == 1:  # for grey-scale image
        original_image = original_image.repeat(1, 3, 1, 1)
   

    masks = masks[: len(references)]
    masks = torch.from_numpy(np.stack(masks))

    original_image, reference_images, masks = (
        original_image.to(device),
        reference_images.to(device),
        masks.to(device),
    )

    mixed = model(original_image, reference_images, masks, shift_values).cpu()
    mixed = np.asarray(
        np.clip(mixed * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8
    ).transpose(
        (0, 2, 3, 1)
    )  # 0~255

    return mixed

def generate_image(    
    G,
    seed,
    truncation_psi: float,
    noise_strength,
    outdir: str,
    add_vector = None
    ):

    os.makedirs(outdir, exist_ok=True)
    
    label = torch.zeros([1, G.c_dim], device=device)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    z += noise_strength[1] * torch.from_numpy(np.random.RandomState(noise_strength[0]).randn(1, G.z_dim)).to(device)
    if add_vector is not None:
        z += torch.from_numpy(add_vector).to(device)
    
    img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    path = f'{outdir}/{seed:04d}-{noise_strength[0]:04d}-{noise_strength[1]:.3f}-{np.random.rand(1)}.png'
    
    Image.fromarray(img, 'RGB').save(path)
    return path
    
def generate_images(
    G,
    N,
    seed,
    truncation_psi,
    noise_strength,
    outdir,
    add_vector=None
    ):
     
    print(outdir)
    os.makedirs(outdir, exist_ok=True)
    

    label = torch.zeros([N, G.c_dim], device=device)
    if len(seed) == 1:
        z = torch.from_numpy(np.random.RandomState(seed[0]).randn(1, G.z_dim)).repeat(N, 1).to(device)
    else:
        z =  torch.from_numpy(np.vstack([np.random.RandomState(random_seed).randn(1, G.z_dim) for random_seed in seed])).to(device)
    
    random_vectors = np.vstack([np.random.RandomState(random_seed).randn(1, G.z_dim) for random_seed in noise_strength[0]])
    
    z +=  torch.from_numpy(random_vectors * noise_strength[1].reshape(-1, 1)).to(device)
    if add_vector is not None:
        z +=  torch.from_numpy(add_vector).to(device)
    img = G(z, label, truncation_psi=truncation_psi, noise_mode='const')
    
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    paths = []
    for idx in range(N):
        print(idx)
        seed_name = seed[idx] if len(seed) != 1 else seed[0]
        path = f'{outdir}/{seed_name:04d}-{noise_strength[0][idx]:04d}-{noise_strength[1][idx]:.3f}-{np.random.rand(1)}.png'
        Image.fromarray(img[idx], 'RGB').save(path)
        paths.append(path)
    
    return paths
    


@app.route("/post", methods=["POST"])
def post():
    global rand
    if request.method == "POST":
        user_id = request.json["id"]
        
        save_dir = f"demo/static/generated/{user_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        if request.json['type'] == 'random_generate':
            N = 12
            print(request.json['seed_key'])
            seed_list_tmp = seed_list[request.json['seed_key']]
            rand = np.random.choice(seed_list_tmp, N)
            paths = generate_images(G, N, rand, args['truncation_psi'], (np.zeros(N).astype(int), np.zeros(N)), args['outdir'])
            return flask.jsonify(result=paths)

        elif request.json['type'] == 'random_generate_noise':
            rand = request.json['random_seed']
            N = 12
            rand_sizes =  np.linspace(0, 0.80, num=N, endpoint=False)
            rand_noise_seed = np.random.randint(0, 2**31-1, N)
            paths = generate_images(G, N, [rand], args['truncation_psi'], (rand_noise_seed, rand_sizes), args['outdir'])
            return flask.jsonify(result=paths)

        elif request.json['type'] == 'direct_manipulation_example':
            rand, rand_noise, rand_size, rand_hash = request.json['random_seed'].split('-')

            N = 10
            manipulation_size = [
                [-10, 0, 0, 0, 0],
                [10, 0, 0, 0, 0],
                [0, -10, 0, 0, 0],
                [0, 10, 0, 0, 0],
                [0, 0, -10, 0, 0],
                [0, 0, 10, 0, 0],
                [0, 0, 0, -10, 0],
                [0, 0, 0, 10, 0],
                [0, 0, 0, 0, -10],
                [0, 0, 0, 0, 10]
            ]
                                
            dmv_total = np.matmul(np.array(manipulation_size), sefa_vector[:5].numpy())
            print(dmv_total.shape)
            
            path = generate_images(G, N, [int(rand)], args['truncation_psi'], (np.full((N), rand_noise, dtype=int), np.full((N), rand_size, dtype=float)), args['outdir'], dmv_total)
            print(path)
            return flask.jsonify(result=path)

        elif request.json['type'] == 'direct_manipulation':
            
            rand, rand_noise, rand_size, rand_hash = request.json['random_seed'].split('-')
            manipulation_size = request.json['manipulation']


            dmv_total = np.sum(sefa_vector[:len(manipulation_size),:].numpy()*np.array(manipulation_size).reshape(-1, 1) , axis=0)
            
            path = generate_image(G, int(rand), args['truncation_psi'], (int(rand_noise), float(rand_size)), args['outdir'], dmv_total)
            return flask.jsonify(result=path)

        
        
        elif request.json['type'] == 'generate':
            
            original = request.json["original"]
            references = request.json["references"]
            
            colors = [hex2val(hex) for hex in request.json["colors"]]
            data_reference_bin = []
            shift_values = request.json["shift_original"]
            
            masks = []
            for i, d_ref in enumerate(request.json["data_reference"]):
                data_reference_bin.append(base64.b64decode(d_ref))

                with open(f"{save_dir}/classmap_reference_{i}.png", "wb") as f:
                    f.write(bytearray(data_reference_bin[i]))

            for i in range(len(colors)):
                class_map = Image.open(io.BytesIO(data_reference_bin[i]))
                class_map = np.array(class_map)[:, :, :3]
                mask = np.array(
                    (np.isclose(class_map, colors[i], atol=2.0)).all(axis=2), dtype=np.uint8
                )  # * 255
                mask = np.asarray(mask, dtype=np.float32).reshape(
                    (1, mask.shape[0], mask.shape[1])
                )
                masks.append(mask)

            generated_images = my_morphed_images(
                original,
                references,
                masks,
                shift_values,
                interpolation=args['interpolation_step'],
                save_dir=save_dir,
            )
            paths = []

            for i in range(args['interpolation_step']):
                path = f"{save_dir}/{i}.png"
                Image.fromarray(generated_images[i]).save(path)
                paths.append(path + "?{}".format(secrets.token_urlsafe(16)))
            
            return flask.jsonify(result=paths)

        elif request.json['type'] == 'upload':
            cropped_image = request.json["image"].replace('data:image/jpeg;base64,','')
            
            directory = request.json["directory"]
            file_name = request.json["file_name"]
            if directory == 'etc' : directory = ""
            im =  np.array(Image.open(io.BytesIO(base64.b64decode(cropped_image))))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # perform brightness correction in tiles
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            im = clahe.apply(im)
            R = 8
            im_predict = cv2.resize(im, (im.shape[1] // R * R, im.shape[0] // R * R), interpolation=cv2.INTER_AREA)
            im_predict = np.reshape(im_predict, (1, im_predict.shape[0], im_predict.shape[1], 1))
            # im_predict = ((im_predict/255)*220)/255
            im_predict = im_predict.astype(np.float32) * 0.003383

            with graph.as_default():
                result = sketch_model.predict(im_predict, batch_size=1)[0]

            im_res = (result - np.mean(result) + 1.) * 255
            im_res = cv2.resize(im_res, (im.shape[1], im.shape[0]))

            cv2.imwrite(os.path.join("demo/static/components/img/sketch", directory, file_name), im_res)

            
            

            return  flask.jsonify(result=['end'])
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    with open("config.json", 'r') as config:
        args = json.load(config)

    device = "cuda"
    base_path = f"demo/static/components/img/{args['dataset']}/"
    
    canvas_size = 256
    
    with dnnlib.util.open_url(args['stylegan2_ckpt']) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
    # StyleMapGAN
    ckpt = torch.load(args['stylemapgan_ckpt'])
    model = Model(ckpt["train_args"]).to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()
    
    print('Import Sketch Model')
    sketch_model = None
    rand = 0
    # graph = tf.get_default_graph()
    # sketch_model = load_model('./sketch_model.h5')
    # sketch_model._make_predict_function()
    print('Success.')
    
    direct_manipulation_vectors = [] 
    for dmv_path in args['direct_manipulation_vector']:
        print(dmv_path)
        direct_manipulation_vectors.append(torch.load(dmv_path)['boundary'].cpu())
        
    with open(args["seed_info"], 'r') as seed_info:
        seed_list = json.load(seed_info)

    direct_manipulation_vectors  = np.vstack(direct_manipulation_vectors)
    
    sefa_vector = torch.load(args['sefa_vector'])['eigvec']
    

    app.run(host="127.0.0.1", port=7000,
    debug=True,
    #  debug=False, 
    threaded=False
     
     )
