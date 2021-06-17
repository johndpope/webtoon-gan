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
from PIL import Image

######
import torch
from torch import nn
from training.model import Generator, Encoder
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import io

app = Flask(
    __name__,
    template_folder="demo/templates",
    static_url_path="/demo/static",
    static_folder="demo/static",
)

app.config["MAX_CONTENT_LENGTH"] = 10000000  # allow 10 MB post

# for 1 gpu only.
class Model(nn.Module):
    def __init__(self):
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
            mask=[masks, shift_values, args.interpolation_step],
        )[0]

        return mixed

    def forward_direction(self, image, distance, direction_vector=None):
        get_cuda_device = image.get_device()
        fake_stylecode = self.e_ema(image)
        direction_vector = direction_vector.unsqueeze(0).to(get_cuda_device)
        modified_stylecode = fake_stylecode + distance * direction_vector
        transformed_image, _ = self.g_ema(
            modified_stylecode, 
            input_is_stylecode=True
        )
    
        return transformed_image
        



@app.route("/")
def index():
    image_paths = []
    return render_template(
        "index.html",
        canvas_size=train_args.size,
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
def my_morphed_images(
    original, references, masks, shift_values, interpolation=8, save_dir=None
):
    
    original_path = original.split('?')[0] if 'demo' in original else base_path + original
    original_image = Image.open(original_path)
    reference_images = []

    for ref in references:
        ref_path = ref.split('?')[0] if 'demo' in ref else base_path + ref
        reference_image =  TF.to_tensor(Image.open(ref_path).resize((train_args.size, train_args.size)))
        if reference_image.ndim == 2 :
            reference_image = reference_image.unsqueeze(0)
        if reference_image.shape[0] == 1 :
            reference_image = reference_image.repeat(3, 1, 1)

        reference_images.append(reference_image)
        

    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(
        original_image, size=(train_args.size, train_args.size)
    )
    original_image = (original_image - 0.5) * 2

    reference_images = torch.stack(reference_images)
    reference_images = F.interpolate(
        reference_images, size=(train_args.size, train_args.size)
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

@torch.no_grad()
def direction_change(original, distance, direction, save_dir=None):
    original_path = original.split('?')[0] if 'demo' in original else base_path + original
    original_image = Image.open(original_path)
    original_image = TF.to_tensor(original_image).unsqueeze(0)
    original_image = F.interpolate(
        original_image, size=(train_args.size, train_args.size)
    )
    original_image = (original_image - 0.5) * 2

    if original_image.shape[1] == 1:  # for grey-scale image
        original_image = original_image.repeat(1, 3, 1, 1)
        
 

    original_image, direction = original_image.to(device), direction.to(device)

    mixed = model.forward_direction(original_image, distance, direction).cpu()
    mixed = np.asarray(
        np.clip(mixed * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8
    ).transpose(
        (0, 2, 3, 1)
    )  # 0~255

    return mixed

@app.route("/post", methods=["POST"])
def post():
    if request.method == "POST":
        user_id = request.json["id"]
        save_dir = f"demo/static/generated/{user_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if request.json['type'] == 'original':
            original = request.json["original"]
            distance = request.json["distance"]
           
            generated_images = direction_change(
                original,
                distance,
                direction,
                save_dir=save_dir,
            )
            paths = []

            
            path = f"{save_dir}/origin_{distance}.png"
            Image.fromarray(generated_images[0]).save(path)
            path += "?{}".format(secrets.token_urlsafe(16))
            
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
                interpolation=args.interpolation_step,
                save_dir=save_dir,
            )
            paths = []

            for i in range(args.interpolation_step):
                path = f"{save_dir}/{i}.png"
                Image.fromarray(generated_images[i]).save(path)
                paths.append(path + "?{}".format(secrets.token_urlsafe(16)))
            
            return flask.jsonify(result=paths)

        elif request.json['type'] == 'upload':
            cropped_image = request.json["image"].replace('data:image/jpeg;base64,','')
            
            directory = request.json["directory"]
            file_name = request.json["file_name"]
            if directory == 'etc' : directory = ""
            print(file_name)
            with open(os.path.join("demo/static/components/img/sketch", directory, file_name), "wb") as f:
                f.write(bytearray(base64.b64decode(cropped_image)))
            return  flask.jsonify(result=['end'])
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="celeba_hq",
        choices=["celeba_hq", "afhq", "lsun/church_outdoor", "lsun/car", "sketch"],
    )
    parser.add_argument("--interpolation_step", type=int, default=16)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--boundary", type=str, required=True)
    parser.add_argument(
        "--MAX_CONTENT_LENGTH", type=int, default=10000000
    )  # allow maximum 10 MB POST
    args = parser.parse_args()

    device = "cuda"
    base_path = f"demo/static/components/img/{args.dataset}/"
    ckpt = torch.load(args.ckpt)

    train_args = ckpt["train_args"]
    print("train_args: ", train_args)

    print('Import Model...')
    model = Model().to(device)
    model.g_ema.load_state_dict(ckpt["g_ema"])
    model.e_ema.load_state_dict(ckpt["e_ema"])
    model.eval()
    print('Success.')
    
    print('Import Boundary...')
    boundary_infos = torch.load(args.boundary)
    direction = -boundary_infos["boundary"]
    print('Success.')

    app.debug = True
    app.run(host="127.0.0.1", port=7000)
