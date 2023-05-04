from Lora import Lora
configs = [
    {
        "dir_name": "chenweiting-512-10-Ada-hassan",
        "data_name": "",
        "resolution": 512,
        "sd_path": "/root/autodl-tmp/webui_models/Stable-diffusion/hassanblend14.safetensors",
        "instance_token": "chenweiting man", 
        "class_token": "man",
        "train_repeats": 10,
        "reg_repeats": 1,
        "num_epochs": 1,
        "network_dim": 128,
        "network_alpha": 64,
        "optimizer_type": "AdaFactor", # @param ["AdamW", "AdamW8bit", "Lion", "DAdaptation", "AdaFactor", "SGDNesterov", "SGDNesterov8bit"]
        "unet_lr": 1e-5,
        "text_encoder_lr": 0.5e-5,
        "prior_loss_weight": 1,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
        ],
        "images_per_prompt": 1,
        "save_n_epochs_type_value": 0.5,

    },
] 

for config in configs:
    model = Lora(**config)

    # model.prepare(
    # data_anotation = "blip",  # @param ["none", "waifu", "blip", "combined"]
    # undesired_tags = "",
    # general_threshold = 0.5,  # for waifu tags threshold
    # )

    model.train()