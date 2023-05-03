from Dreambooth import Dreambooth

dir_name = "chenweiting-512-500-DA-hassan"
data_name = "chenweiting/chenweiting-512"
sd_path = "/root/autodl-tmp/webui_models/Stable-diffusion/hassanblend14.safetensors"

instance_token="chenweiting man" 
class_token="man"

train_repeats = 10
reg_repeats = 1
max_train_steps = 600
save_n_epoch_ratio = 1
optimizer_type = "DAdaptation" # @param ["AdamW", "AdamW8bit", "Lion", "SGDNesterov", "SGDNesterov8bit", "DAdaptation", "AdaFactor"]
learning_rate = 1
prior_loss_weight = 1.0

prompts = prompts = [
    "1boy, white shirt",
    "1boy, black jacket",
]
images_per_prompt = 1

model = Dreambooth(
    dir_name=dir_name,
    data_name=data_name,
    sd_path=sd_path,
    instance_token=instance_token,
    class_token=class_token,
    train_repeats=train_repeats,
    reg_repeats=reg_repeats,
    max_train_steps=max_train_steps,
    save_n_epoch_ratio=save_n_epoch_ratio,
    optimizer_type=optimizer_type,
    learning_rate=learning_rate,
    prior_loss_weight=prior_loss_weight,
    prompts=prompts,
    images_per_prompt=images_per_prompt,
)

model.prepare(
    data_anotation = "none",  # @param ["none", "waifu", "blip", "combined"]
    undesired_tags = "",
    general_threshold = 0.5, 
)

model.train()