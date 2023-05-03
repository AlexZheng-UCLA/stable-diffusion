from Lora import Lora

dir_name = "chenweiting-512-10-DA-hassan"
data_name = "chenweiting/chenweiting-512"
sd_path = "/root/autodl-tmp/webui_models/Stable-diffusion/hassanblend14.safetensors"

instance_token="chenweiting man" 
class_token="man"

train_repeats = 10
reg_repeats = 1
network_dim = 128
network_alpha = 64
optimizer_type = "DAdaptation" # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
unet_lr = 1
text_encoder_lr = 0.5
prior_loss_weight = 1.0

prompts = prompts = [
    "1boy, white shirt",
    "1boy, black jacket",
]
images_per_prompt = 1


model = Lora(
    dir_name=dir_name,
    data_name=data_name,
    sd_path=sd_path,
    instance_token=instance_token,
    class_token=class_token,
    train_repeats=train_repeats,
    reg_repeats=reg_repeats,
    network_dim=network_dim,
    network_alpha=network_alpha,
    optimizer_type=optimizer_type,
    unet_lr=unet_lr,
    text_encoder_lr=text_encoder_lr,
    prior_loss_weight=prior_loss_weight,
    prompts=prompts,
    images_per_prompt=images_per_prompt,
)

model.prepare(
    data_anotation = "combined",  # @param ["none", "waifu", "blip", "combined"]
    undesired_tags = "",
    general_threshold = 0.5, 
)

model.train()