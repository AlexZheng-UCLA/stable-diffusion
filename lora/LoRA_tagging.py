#!/usr/bin/env python
# coding: utf-8
# ******************************************************************************************************
## Main Settings

# directory 
dir_name = "redscarf-512-100-DA"
data_name = "redscarfsnow"

# dataset
instance_token = "" 
class_token = ""  
add_token_to_caption = False
resolution = 512
flip_aug = True 
data_anotation = "waifu"  # @param ["none", "waifu", "blip", "combined"]
caption_extension = ".txt"  # @param ["none", ".txt", ".caption", "combined"]


# ******************************************************************************************
# ## Other Settings

root_dir = "/home/ubuntu/alex/lora"
v2 = False 
v_parameterization = False

# waifu
undesired_tags = "solo,snow,snowing,scarf,red scarf,cape,red cape"
general_threshold = 0.3 #@param {type:"slider", min:0, max:1, step:0.05}
character_threshold = 0.5 #@param {type:"slider", min:0, max:1, step:0.05}

batch_size = 2 #@param {type:'number'}
max_data_loader_n_workers = 2 #@param {type:'number'}
model = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2" #@param ["SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
#@markdown Use the `recursive` option to process subfolders as well, useful for multi-concept training.
recursive = False #@param {type:"boolean"} 
#@markdown Debug while tagging, it will print your image file with general tags and character tags.
verbose_logging = False #@param {type:"boolean"}

# **************************************************************************************************

import os
import shutil
import random
import toml
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import torch
import subprocess
from subprocess import getoutput
from accelerate.utils import write_basic_config

import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("cuDNN version:", torch.backends.cudnn.version())
print("cuDNN enabled:", torch.backends.cudnn.enabled)

# **************************************************************************************************
repo_dir = os.path.join(root_dir, "kohya-trainer")
training_dir = os.path.join(root_dir, dir_name)
dataset_dir = os.path.join(root_dir, "dataset", data_name)
train_data_dir = os.path.join(training_dir, "train_data")
reg_data_dir = os.path.join(training_dir, "reg_data")
config_dir = os.path.join(training_dir, "config")
accelerate_config = os.path.join(repo_dir, "accelerate_config/config.yaml")
tools_dir = os.path.join(repo_dir, "tools")
finetune_dir = os.path.join(repo_dir, "finetune")
output_dir = os.path.join(training_dir, "output")
sample_dir = os.path.join(output_dir, "sample")
inference_dir = os.path.join(output_dir, "inference")
logging_dir = os.path.join(training_dir, "log")

shutil.copytree(dataset_dir, train_data_dir, dirs_exist_ok=True)


for dir in [
    training_dir, 
    config_dir,
    output_dir, 
    sample_dir,
    inference_dir
    ]:
    os.makedirs(dir, exist_ok=True)

if not os.path.exists(accelerate_config):
    write_basic_config(save_location=accelerate_config)


test = os.listdir(train_data_dir)
# @markdown This section will delete unnecessary files and unsupported media such as `.mp4`, `.webm`, and `.gif`.

supported_types = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".caption",
    ".combined",
    ".npz",
    ".txt",
    ".json",
]

for item in test:
    file_ext = os.path.splitext(item)[1]
    if file_ext not in supported_types:
        print(f"Deleting file {item} from {train_data_dir}")
        os.remove(os.path.join(train_data_dir, item))

# @markdown ### <br> Convert Transparent Images
# @markdown This code will convert your transparent dataset with alpha channel (RGBA) to RGB and give it a white background.

convert = True  # @param {type:"boolean"}
random_color = False  # @param {type:"boolean"}

batch_size = 32

images = [
    image
    for image in os.listdir(train_data_dir)
    if image.endswith(".png") or image.endswith(".webp")
]
background_colors = [
    (255, 255, 255),
    (0, 0, 0),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def process_image(image_name):
    img = Image.open(f"{train_data_dir}/{image_name}")

    if img.mode in ("RGBA", "LA"):
        if random_color:
            background_color = random.choice(background_colors)
        else:
            background_color = (255, 255, 255)
        bg = Image.new("RGB", img.size, background_color)
        bg.paste(img, mask=img.split()[-1])

        if image_name.endswith(".webp"):
            bg = bg.convert("RGB")
            bg.save(f'{train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
            os.remove(f"{train_data_dir}/{image_name}")
            print(
                f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
            )
        else:
            bg.save(f"{train_data_dir}/{image_name}", "PNG")
            print(f" Converted image: {image_name}")
    else:
        if image_name.endswith(".webp"):
            img.save(f'{train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
            os.remove(f"{train_data_dir}/{image_name}")
            print(
                f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
            )
        else:
            img.save(f"{train_data_dir}/{image_name}", "PNG")


num_batches = len(images) // batch_size + 1

if convert:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in tqdm(range(num_batches)):
            start = i * batch_size
            end = start + batch_size
            batch = images[start:end]
            executor.map(process_image, batch)

    print("All images have been converted")


# ## Data Annotation
# You can choose to train a model using captions. We're using [BLIP](https://huggingface.co/spaces/Salesforce/BLIP) for image captioning and [Waifu Diffusion 1.4 Tagger](https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags) for image tagging similar to Danbooru.
# - Use BLIP Captioning for: `General Images`
# - Use Waifu Diffusion 1.4 Tagger V2 for: `Anime and Manga-style Images`

if data_anotation == "blip" or data_anotation == "combined":

    os.chdir(finetune_dir)

    batch_size = 2 #@param {type:'number'}
    max_data_loader_n_workers = 2 #@param {type:'number'}
    beam_search = True #@param {type:'boolean'}
    min_length = 5 #@param {type:"slider", min:0, max:100, step:5.0}
    max_length = 75 #@param {type:"slider", min:0, max:100, step:5.0}

    command = f'''python make_captions.py "{train_data_dir}" --batch_size {batch_size} {"--beam_search" if beam_search else ""} --min_length {min_length} --max_length {max_length} --caption_extension .caption --max_data_loader_n_workers {max_data_loader_n_workers}'''

    subprocess.run(command, shell=True, check=True)

# 4.2.2. Waifu Diffusion 1.4 Tagger V2
import os

if data_anotation == "waifu" or data_anotation == "combined":
    os.chdir(finetune_dir)

    batch_size = 2 #@param {type:'number'}
    max_data_loader_n_workers = 2 #@param {type:'number'}
    model = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2" #@param ["SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
    #@markdown Use the `recursive` option to process subfolders as well, useful for multi-concept training.
    recursive = False #@param {type:"boolean"} 
    #@markdown Debug while tagging, it will print your image file with general tags and character tags.
    verbose_logging = False #@param {type:"boolean"}

    config = {
        "_train_data_dir": train_data_dir,
        "batch_size": batch_size,
        "repo_id": model,
        "recursive": recursive,
        "remove_underscore": True,
        "general_threshold": general_threshold,
        "character_threshold": character_threshold,
        "caption_extension": ".txt",
        "max_data_loader_n_workers": max_data_loader_n_workers,
        "debug": verbose_logging,
        "undesired_tags": undesired_tags
    }

    args = ""
    for k, v in config.items():
        if k.startswith("_"):
            args += f'"{v}" '
        elif isinstance(v, str):
            args += f'--{k}="{v}" '
        elif isinstance(v, bool) and v:
            args += f"--{k} "
        elif isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        elif isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    final_args = f"python tag_images_by_wd14_tagger.py {args}"

    os.chdir(finetune_dir)
    subprocess.run(final_args, shell=True, check=True)


# ### Combine BLIP and Waifu

# os.chdir(train_data_dir)

def read_file_content(file_path):
    with open(file_path, "r") as file:
        content = file.read()
    return content

def remove_redundant_words(content1, content2):
    return content1.rstrip('\n') + ', ' + content2

def write_file_content(file_path, content):
    with open(file_path, "w") as file:
        file.write(content)

def combine():
    directory = train_data_dir
    extension1 = ".caption"
    extension2 = ".txt"
    output_extension = ".combined"

    for file in os.listdir(directory):
        if file.endswith(extension1):
            filename = os.path.splitext(file)[0]
            file1 = os.path.join(directory, filename + extension1)
            file2 = os.path.join(directory, filename + extension2)
            output_file = os.path.join(directory, filename + output_extension)

            if os.path.exists(file2):
                content1 = read_file_content(file1)
                content2 = read_file_content(file2)

                combined_content = remove_redundant_words(content1, content2)

                write_file_content(output_file, combined_content)

combine()