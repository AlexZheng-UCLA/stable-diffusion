#!/usr/bin/env python
# coding: utf-8

# # Inference
# ### Main Settings

dir_name = "chenweiting_512_66_DA_combined"
network_weight = "/home/ubuntu/alex/lora/chenweiting_640_66_DA_combined/output/chenweiting.safetensors" # leave empty to use all the loras in the dir 
inference_ckpt = "/home/ubuntu/stable-diffusion-webui/models/Stable-diffusion/chilloutmix_NiPrunedFp32Fix.safetensors"
vae = "/home/ubuntu/stable-diffusion-webui/models/VAE/vae-ft-mse-840000-ema-pruned.ckpt" 
network_mul = 1  # @param {type:"slider", min:-1, max:2, step:0.05}

instance_prompt = ""  # @param {type: "string"}
pre = "masterpiece, best quality" 
negative = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"  # @param {type: "string"}

scale = 7  # @param {type: "slider", min: 1, max: 40}
sampler = "k_dpm_2"  # @param ["ddim", "pndm", "lms", "euler", "euler_a", "heun", "dpm_2", "dpm_2_a", "dpmsolver","dpmsolver++", "dpmsingle", "k_lms", "k_euler", "k_euler_a", "k_dpm_2", "k_dpm_2_a"]
steps = 28  # @param {type: "slider", min: 1, max: 100}
precision = "fp16"  # @param ["fp16", "bf16"] {allow-input: false}
width = 512  # @param {type: "integer"}
height = 512  # @param {type: "integer"}
images_per_prompt = 1  # @param {type: "integer"}
batch_size = 1  # @param {type: "integer"}
clip_skip = 1  # @param {type: "slider", min: 1, max: 40}
seed = -1  # @param {type: "integer"}

prompts = [
    "1 man in white, upper body",
    "1 man in white, full body",
    "futuristic cyberpunk portrait of a person wearing neon-lit clothing and accessories, surrounded by a cityscape with flying cars and holographic billboards", 
    "a vintage sepia-toned photograph of a person dressed in Victorian-era clothing, complete with ornate patterns and intricate lace details, against a backdrop of a 19th-century parlor",
    "an impressionist painting of a person in a sunlit garden surrounded by vividly colored flowers, with soft brush strokes and a dreamy, light-filled atmosphere",
    "a person in the style of a classic 1950s American comic book, with bold colors, dynamic poses, and exaggerated facial expressions, against the backdrop of a bustling city street",
    "a black-and-white film noir inspired portrait of a person dressed as a detective or femme fatale, with high contrast lighting, dramatic shadows, and a mysterious atmosphere",
    "a portrait of a person in the style of a Renaissance painting, with rich colors, detailed textures, and a background featuring classical architecture or lush landscapes",
    "a person in the style of a Japanese ukiyo-e woodblock print, with vibrant colors, bold outlines, and traditional patterns, set against a backdrop of a mountainous landscape or bustling marketplace",
    "a street art-inspired portrait of a person with graffiti-like elements, bold colors, and urban textures, set against a brick wall or other urban background",
    "a person's portrait in the style of a 1960s psychedelic poster, with bright colors, swirling patterns, and a groovy, surreal atmosphere",
    "an image of a person in the style of a minimalist line art drawing, using only simple shapes and lines to convey the subject's features and form, set against a solid colored background"
    ]

#*******************************************************************
# ### Other settings 

root_dir = "/home/ubuntu/alex/lora"
v2 = False  # @param {type:"boolean"}
v_parameterization = False  # @param {type:"boolean"}

network_module = "networks.lora"
network_args = ""

# *******************************************************************
final_prompts = []
for prompt in prompts:
    final_prompts.append(
    f"{instance_prompt}, {pre}, {prompt} --n {negative}"
    if instance_prompt
    else f"{pre}, {prompt} --n {negative}"
    )

import os
import glob
import subprocess

repo_dir = os.path.join(root_dir, "kohya-trainer")
output_dir = os.path.join(root_dir, dir_name, "output")
inference_dir = os.path.join(output_dir, "inference")

prompt_path = os.path.join(root_dir, "inference_prompts.txt")
with open(prompt_path, 'w') as file:
    # Write each string to the file on a new line
    for string in final_prompts:
        file.write(string + '\n')

if network_weight:
    lora_files = [network_weight]

else:
    os.chdir(output_dir)
    lora_files = [os.path.abspath(file) for file in glob.glob('*.safetensors')]

os.chdir(repo_dir)

for network_weight in lora_files:
    print(network_weight)

    config = {
        "v2": v2,
        "v_parameterization": v_parameterization,
        "network_module": network_module,
        "network_weight": network_weight,
        "network_mul": float(network_mul),
        "network_args": eval(network_args) if network_args else None,
        "ckpt": inference_ckpt,
        "outdir": inference_dir,
        "xformers": False,
        "vae": vae if vae else None,
        "fp16": True,
        "W": width,
        "H": height,
        "seed": seed if seed > 0 else None,
        "scale": scale,
        "sampler": sampler,
        "steps": steps,
        "max_embeddings_multiples": 3,
        "batch_size": batch_size,
        "images_per_prompt": images_per_prompt,
        "clip_skip": clip_skip if not v2 else None,
        # "prompt":final_prompts,
        "from_file": prompt_path
    }

    args = ""
    for k, v in config.items():
        if isinstance(v, str):
            args += f'--{k}="{v}" '
        if isinstance(v, list):
            print(f"{k}: {v[0]}")
            args += f"--{k}={v} "
        if isinstance(v, bool) and v:
            args += f"--{k} "
        if isinstance(v, float) and not isinstance(v, bool):
            args += f"--{k}={v} "
        if isinstance(v, int) and not isinstance(v, bool):
            args += f"--{k}={v} "

    print(type(prompts))
    final_args = f"python gen_img_diffusers.py {args}"
    subprocess.call(final_args, shell=True)



