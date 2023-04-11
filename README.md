# Stable Diffusion 



## Dreambooth 

<b>object_dreambooth_server</b>: no package install

<b>object_dreambooth_colab</b>: with package install, with blip and manual captioning

<b>blip</b>: blip and manual captioning

<b>train_dreambooth</b>: core change written in top of file

<b>environment.yml</b>: you can create conda environment for dreambooth training by

  1. change the first line of environment.yml to your name (xxx -> your name)

  2. conda env create -f environment.yml

     

## Lora

LoRA_training: train the LoRA using Dreambooth, with all the settings in the beginning

LoRA_inference: testing the LoRA

LoRA_all: contains training + inference
