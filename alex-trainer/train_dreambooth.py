from Dreambooth import Dreambooth
configs = [
    {
        "dir_name": "chenweiting-512-200-8bit-hassan",
        "data_name": "chenweiting/chenweiting-512",
        "resolution": 512,
        "sd_path": "/root/autodl-tmp/webui_models/Stable-diffusion/hassanblend14.safetensors",
        "instance_token": "chenweiting man", 
        "class_token": "man",
        "train_repeats": 10,
        "reg_repeats": 1,
        "max_train_steps" : 200,
        "optimizer_type": "AdamW8bit", # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
        "learning_rate" : 1e-6,
        "prior_loss_weight": 1.0,
        "save_n_epoch_ratio" : 0.5,
        "sample_every_n_steps" : 100,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
        ],
        "images_per_prompt": 1,
    }
] 



for config in configs:
    model = Dreambooth(**config)

    # model.prepare(
    # data_anotation = "blip",  # @param ["none", "waifu", "blip", "combined"]
    # undesired_tags = "",
    # general_threshold = 0.5,  # for waifu tags threshold
    # )

    model.train()