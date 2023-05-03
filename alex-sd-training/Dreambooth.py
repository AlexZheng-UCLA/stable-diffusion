#!/usr/bin/env python
# coding: utf-8
import os
import sys
import shutil
import random
from typing import Any
import toml
import concurrent.futures
from tqdm import tqdm
from PIL import Image
import torch
import subprocess
from subprocess import getoutput
from accelerate.utils import write_basic_config

# define class Dreambooth
class Dreambooth():
    def __init__(
        self, 

        # project           
        dir_name,
        data_name,

        # path
        sd_path,
        resume_path="",
        v2=False,

        # dataset
        instance_token=None,
        class_token=None,

        # training 
        train_repeats = 10,
        reg_repeats = 1,
        max_train_steps = 600,
        save_n_epoch_ratio = 1,
        optimizer_type = "DAdaptation",  # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
        learning_rate = 1,
        prior_loss_weight = 1.0,

        # sampling
        prompts = None,
        sample_every_n_steps = 100,
        images_per_prompt = 1,
    ):

        self.dir_name = dir_name
        self.data_name = data_name
        self.sd_path = sd_path
        self.resume_path = resume_path
        self.v2 = v2
        self.instance_token = instance_token
        self.class_token = class_token
        self.train_repeats = train_repeats
        self.reg_repeats = reg_repeats
        self.max_train_steps = max_train_steps
        self.save_n_epoch_ratio = save_n_epoch_ratio
        self.prompts = prompts
        self.sample_every_n_steps = sample_every_n_steps
        self.images_per_prompt = images_per_prompt
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.prior_loss_weight = prior_loss_weight

        self.project_name = dir_name
        self.vae_path = "/root/autodl-tmp/webui_models/VAE/vae-ft-mse-840000-ema-pruned.safetensors"
        self.blip_path = "/root/autodl-tmp/webui_models/BLIP/model_large_caption.pth"


        self.add_token_to_caption = True
        self.resolution = 512
        self.flip_aug = True
        self.clip_skip = 1 


        self.train_batch_size = 1
        self.lr_scheduler = "polynomial"  #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]

        self.root_dir = "/root/alex-sd-training"
        self.output_dir = "/root/autodl-tmp/training-outputs"
        self.save_model_dir = "/root/autodl-fs/webui_models/Stable-diffusion"
        self.v_parameterization = False

        self.repo_dir = os.path.join(self.root_dir, "kohya-trainer")
        self.training_dir = os.path.join(self.output_dir, dir_name)
        self.dataset_dir = os.path.join(self.root_dir, "dataset", data_name)
        self.train_data_dir = os.path.join(self.training_dir, "train_data")
        self.reg_data_dir = os.path.join(self.training_dir, "reg_data")
        self.config_dir = os.path.join(self.training_dir, "config")
        self.accelerate_config = os.path.join(self.repo_dir, "accelerate_config/config.yaml")
        self.tools_dir = os.path.join(self.repo_dir, "tools")
        self.finetune_dir = os.path.join(self.repo_dir, "finetune")
        self.sample_dir = os.path.join(self.training_dir, "sample")
        self.inference_dir = os.path.join(self.training_dir, "inference")
        self.logging_dir = os.path.join(self.training_dir, "log")

        for dir in [
            self.training_dir,
            self.train_data_dir,
            self.reg_data_dir,
            self.config_dir,
            self.output_dir, 
            self.sample_dir,
            self.inference_dir
            ]:
            os.makedirs(dir, exist_ok=True)

        shutil.copytree(self.dataset_dir, self.train_data_dir, dirs_exist_ok=True)
        if not os.path.exists(self.accelerate_config):
            write_basic_config(save_location=self.accelerate_config)


        test = os.listdir(self.train_data_dir)
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
                print(f"Deleting file {item} from {self.train_data_dir}")
                os.remove(os.path.join(self.train_data_dir, item))

        
    def prepare(self,
                
        data_anotation = "combined",  # @param ["none", "waifu", "blip", "combined"]
        # dataset
        caption_dropout_rate = 0,  # @param {type:"slider", min:0, max:1, step:0.05}
        caption_dropout_every_n_epochs = 0,  

        # waifu
        undesired_tags = "",
        general_threshold = 0.3, #@param {type:"slider", min:0, max:1, step:0.05}
        character_threshold = 0.5, #@param {type:"slider", min:0, max:1, step:0.05}

    ):
        self.data_anotation = data_anotation
        self.caption_extension = ".txt"
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.keep_tokens = 0
        self.undesired_tags = undesired_tags
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold


        convert = True  # @param {type:"boolean"}
        random_color = True  # @param {type:"boolean"}
        batch_size = 32
        images = [
            image
            for image in os.listdir(self.train_data_dir)
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
            img = Image.open(f"{self.train_data_dir}/{image_name}")

            if img.mode in ("RGBA", "LA"):
                if random_color:
                    background_color = random.choice(background_colors)
                else:
                    background_color = (255, 255, 255)
                bg = Image.new("RGB", img.size, background_color)
                bg.paste(img, mask=img.split()[-1])

                if image_name.endswith(".webp"):
                    bg = bg.convert("RGB")
                    bg.save(f'{self.train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
                    os.remove(f"{self.train_data_dir}/{image_name}")
                    print(
                        f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
                    )
                else:
                    bg.save(f"{self.train_data_dir}/{image_name}", "PNG")
                    print(f" Converted image: {image_name}")
            else:
                if image_name.endswith(".webp"):
                    img.save(f'{self.train_data_dir}/{image_name.replace(".webp", ".jpg")}', "JPEG")
                    os.remove(f"{self.train_data_dir}/{image_name}")
                    print(
                        f" Converted image: {image_name} to {image_name.replace('.webp', '.jpg')}"
                    )
                else:
                    img.save(f"{self.train_data_dir}/{image_name}", "PNG")


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
        os.chdir(self.repo_dir)
        if data_anotation == "blip" or data_anotation == "combined":

            batch_size = 2 #@param {type:'number'}
            max_data_loader_n_workers = 2 #@param {type:'number'}
            beam_search = True #@param {type:'boolean'}
            min_length = 5 #@param {type:"slider", min:0, max:100, step:5.0}
            max_length = 75 #@param {type:"slider", min:0, max:100, step:5.0}

            command = f'''python make_captions.py "{self.train_data_dir}" --caption_weights {self.blip_path} --batch_size {batch_size} {"--beam_search" if beam_search else ""} --min_length {min_length} --max_length {max_length} --caption_extension .caption --max_data_loader_n_workers {max_data_loader_n_workers}'''
            subprocess.run(command, shell=True, check=True)

        # 4.2.2. Waifu Diffusion 1.4 Tagger V2

        if data_anotation == "waifu" or data_anotation == "combined":

            batch_size = 2 #@param {type:'number'}
            max_data_loader_n_workers = 2 #@param {type:'number'}
            model = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2" #@param ["SmilingWolf/wd-v1-4-convnextv2-tagger-v2", "SmilingWolf/wd-v1-4-swinv2-tagger-v2", "SmilingWolf/wd-v1-4-convnext-tagger-v2", "SmilingWolf/wd-v1-4-vit-tagger-v2"]
            #@markdown Use the `recursive` option to process subfolders as well, useful for multi-concept training.
            recursive = False #@param {type:"boolean"} 
            #@markdown Debug while tagging, it will print your image file with general tags and character tags.
            verbose_logging = False #@param {type:"boolean"}
            #@markdown Separate `undesired_tags` with comma `(,)` if you want to remove multiple tags, e.g. `1girl,solo,smile`.

            config = {
                "_train_data_dir": self.train_data_dir,
                "batch_size": batch_size,
                "repo_id": model,
                "recursive": recursive,
                "remove_underscore": True,
                "general_threshold": self.general_threshold,
                "character_threshold": character_threshold,
                "caption_extension": self.caption_extension,
                "max_data_loader_n_workers": max_data_loader_n_workers,
                "debug": verbose_logging,
                "undesired_tags": self.undesired_tags
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
            subprocess.run(final_args, shell=True, check=True)


        # ### Combine BLIP and Waifu

        if data_anotation == "combined":
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
                directory = self.train_data_dir
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


    def train(self,
    ):
        
        lr_scheduler_num_cycles = 0  # @param {'type':'number'}
        lr_scheduler_power = 1 
        lr_warmup_steps = 0 
        noise_offset = 0.0  # @param {type:"number"}

        # sample 
        enable_sample_prompt = True
        scale = 7  # @param {type: "slider", min: 1, max: 40}
        sampler = "ddim"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
        steps = 28  # @param {type: "slider", min: 1, max: 100}
        # precision = "fp16"  # @param ["fp16", "bf16"] {allow-input: false}
        width = 512  # @param {type: "integer"}
        height = 512  # @param {type: "integer"}
        pre = "masterpiece, best quality" 
        negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"  

        # 
        mixed_precision = "fp16"  # @param ["no","fp16","bf16"]
        save_precision = "fp16"  # @param ["float", "fp16", "bf16"] 
        save_model_as = "safetensors"  # @param ["ckpt", "safetensors", "diffusers", "diffusers_safetensors"] {allow-input: false}
        max_token_length = 225  # @param {type:"number"}
        gradient_checkpointing = False  # @param {type:"boolean"}
        gradient_accumulation_steps = 1  # @param {type:"number"}
        seed = -1  # @param {type:"number"}

        if self.add_token_to_caption and self.keep_tokens < 2:
            self.keep_tokens = 1

        def read_file(filename):
            with open(filename, "r") as f:
                contents = f.read()
            return contents

        def write_file(filename, contents):
            with open(filename, "w") as f:
                f.write(contents)

        def add_tag(filename, tag):
            contents = read_file(filename)
            # move the "cat" or "dog" to the beginning of the contents
            if "cat" in contents:
                contents = contents.replace("cat, ", "")
                contents = contents.replace(", cat", "")
                contents = "cat, " + contents
            if "dog" in contents:
                contents = contents.replace("dog, ", "")
                contents = contents.replace(", dog", "")
                contents = "dog, " + contents

            # add the tag
            tag = ", ".join(tag.split())
            tag = tag.replace("_", " ")
            if tag in contents:
                return
            contents = tag + ", " + contents
            write_file(filename, contents)

        def delete_tag(filename, tag):
            contents = read_file(filename)
            tag = ", ".join(tag.split())
            tag = tag.replace("_", " ")
            if tag not in contents:
                return
            contents = "".join([s.strip(", ") for s in contents.split(tag)])
            write_file(filename, contents)

        if self.caption_extension != "none":

            tag = f"{self.instance_token}"
            for filename in os.listdir(self.train_data_dir):
                if filename.endswith(self.caption_extension):
                    file_path = os.path.join(self.train_data_dir, filename)

                    if self.add_token_to_caption:
                        add_tag(file_path, tag)
                    else:
                        delete_tag(file_path, tag)

        config = {
            "general": {
                "enable_bucket": True,
                "caption_extension": self.caption_extension,
                "shuffle_caption": True,
                "keep_tokens": self.keep_tokens,
                "bucket_reso_steps": 64,
                "bucket_no_upscale": False,
            },
            "datasets": [
                {
                    "resolution": self.resolution,
                    "min_bucket_reso": 320 if self.resolution > 640 else 256,
                    "max_bucket_reso": 1280 if self.resolution > 640 else 1024,
                    "caption_dropout_rate": self.caption_dropout_rate if self.caption_extension == ".caption" else 0,
                    "caption_tag_dropout_rate": self.caption_dropout_rate if self.caption_extension == ".txt" else 0,
                    "caption_tag_dropout_rate": self.caption_dropout_rate if self.caption_extension == ".combined" else 0,
                    "caption_dropout_every_n_epochs": self.caption_dropout_every_n_epochs,
                    "flip_aug": self.flip_aug,
                    "color_aug": False,
                    "face_crop_aug_range": None,
                    "subsets": [
                        {
                            "image_dir": self.train_data_dir,
                            "class_tokens": self.instance_token,
                            "num_repeats": self.train_repeats,
                        },
                        {
                            "is_reg": True,
                            "image_dir": self.reg_data_dir,
                            "class_tokens": self.class_token,
                            "num_repeats": self.reg_repeats,
                        },
                    ],
                }
            ],
        }

        config_str = toml.dumps(config)

        dataset_config = os.path.join(self.config_dir, "dataset_config.toml")

        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None

        config_str = toml.dumps(config)

        with open(dataset_config, "w") as f:
            f.write(config_str)

        print(config_str)

        # 5.3. Optimizer Config

        optimizer_args = ""  # @param {'type':'string'}
        stop_train_text_encoder = -1 #@param {'type':'number'}


        # @title ## 5.4. Training Config
        noise_offset = 0.0  # @param {type:"number"}
        save_state = False  # @param {type:"boolean"}

        os.chdir(self.repo_dir)

        config = {
            "model_arguments": {
                "v2": self.v2,
                "v_parameterization": self.v_parameterization if self.v2 and self.v_parameterization else False,
                "pretrained_model_name_or_path": self.sd_path,
                "vae": self.vae_path,
            },
            "optimizer_arguments": {
                "optimizer_type": self.optimizer_type,
                "learning_rate": self.learning_rate,
                "max_grad_norm": 1.0,
                "stop_train_text_encoder": stop_train_text_encoder if stop_train_text_encoder > 0 else None,
                "optimizer_args": eval(optimizer_args) if optimizer_args else None,
                "lr_scheduler": self.lr_scheduler,
                "lr_warmup_steps": lr_warmup_steps,
                "lr_scheduler_num_cycles": lr_scheduler_num_cycles if self.lr_scheduler == "cosine_with_restarts" else None,
                "lr_scheduler_power": lr_scheduler_power if self.lr_scheduler == "polynomial" else None,
            },
            "dataset_arguments": {
                "cache_latents": True,
                "debug_dataset": False,
            },
            "training_arguments": {
                "output_dir": self.save_model_dir,
                "output_name": self.project_name,
                "save_precision": save_precision,
                "save_every_n_epochs": None,
                "save_n_epoch_ratio": self.save_n_epoch_ratio,
                "save_last_n_epochs": None,
                "save_state": save_state,
                "save_last_n_epochs_state": None,
                "resume": self.resume_path if self.resume_path else None,
                "train_batch_size": self.train_batch_size,
                "max_token_length": 225,
                "mem_eff_attn": False,
                "xformers": False,
                "max_train_steps": self.max_train_steps,
                "max_data_loader_n_workers": 8,
                "persistent_data_loader_workers": True,
                "seed": seed if seed > 0 else None,
                "gradient_checkpointing": gradient_checkpointing,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "mixed_precision": mixed_precision,
                "clip_skip": self.clip_skip if not self.v2 else None,
                "logging_dir": self.logging_dir,
                "log_prefix": self.project_name,
                "noise_offset": noise_offset if noise_offset > 0 else None,
            },
            "sample_prompt_arguments": {
                "sample_dir": self.sample_dir,
                "sample_every_n_steps": self.sample_every_n_steps if enable_sample_prompt else 999999,
                "sample_every_n_epochs": None,
                "sample_sampler": sampler,
            },
            "dreambooth_arguments": {
                "prior_loss_weight": self.prior_loss_weight,
            },
            "saving_arguments": {"save_model_as": save_model_as},
        }

        config_path = os.path.join(self.config_dir, "config_file.toml")
        prompt_path = os.path.join(self.config_dir, "sample_prompt.txt")

        for key in config:
            if isinstance(config[key], dict):
                for sub_key in config[key]:
                    if config[key][sub_key] == "":
                        config[key][sub_key] = None
            elif config[key] == "":
                config[key] = None

        config_str = toml.dumps(config)

        def write_file(filename, contents):
            with open(filename, "w") as f:
                f.write(contents)

        write_file(config_path, config_str)

        final_prompts = []
        for prompt in self.prompts:
            final_prompts.append(
            f"{self.instance_token}, {pre}, {prompt} --n {negative} --w {width} --h {height} --l {scale} --s {steps}"
            if self.add_token_to_caption
            else f"{pre}, {prompt} --n {negative} --w {width} --h {height} --l {scale} --s {steps}"
            )
        with open(prompt_path, 'w') as file:
            # Write each string to the file on a new line
            for string in final_prompts:
                for i in range(self.images_per_prompt):
                    file.write(string + '\n')
            
        print(config_str)

        sample_prompt = os.path.join(self.config_dir, "sample_prompt.txt")
        config_file = os.path.join(self.config_dir, "config_file.toml")
        dataset_config = os.path.join(self.config_dir, "dataset_config.toml")
        
        os.chdir(self.repo_dir)
        command = f'''accelerate launch --config_file={self.accelerate_config} --num_cpu_threads_per_process=1 train_db.py --sample_prompts={sample_prompt} --dataset_config={dataset_config} --config_file={config_file}'''

        subprocess.run(command, shell=True, check=True)


