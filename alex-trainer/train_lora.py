from Lora import Lora
configs = [

    ## maximal input
    {
        "dir_name": "chenweiting-512-10-Ada-hassan",
        "data_name": "chenweiting/chenweiting-512",
        "resolution": 512,
        "v2" : False,
        "sd_path": "/root/autodl-tmp/webui_models/Stable-diffusion/hassanblend14.safetensors",
        "instance_token": "chenweiting man", 
        "class_token": "man",
        "train_repeats": 10,
        "reg_repeats": 1,
        "num_epochs": 1,
        "network_dim": 128,
        "network_alpha": 64,
        "train_batch_size": 1,
        "optimizer_type": "AdaFactor", # @param ["AdamW", "AdamW8bit", "Lion", "DAdaptation", "AdaFactor", "SGDNesterov", "SGDNesterov8bit"]
        "unet_lr": 1e-5,
        "text_encoder_lr": 0.5e-5,
        "lr_scheduler" : "polynomial",  #@param ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "adafactor"]
        "prior_loss_weight": 1,
        "prompts": [
            "1 chenweiting man in white shirt",
            "1 chenweiting man in black jacket",
        ],
        "images_per_prompt": 1,
        "save_n_epochs_ratio" : 0.5,

    },

    # ## minimal input
    # {
    #     "dir_name": "chenweiting-512-10-Ada-hassan",
    #     "data_name": "chenweiting/chenweiting-512",
    #     "instance_token": "chenweiting man", 
    #     "class_token": "man",
    #     "train_repeats": 10,
    #     "reg_repeats": 1,
    #     "prompts": [
    #         "1 chenweiting man in white shirt",
    #         "1 chenweiting man in black jacket",
    #     ],

    # },
] 

for config in configs:
    model = Lora(**config)

    model.prepare(data_anotation = "blip")  # @param ["none", "waifu", "blip", "combined"]
    
    model.train()