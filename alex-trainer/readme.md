1. 如果是新机器，将 autodl-fs/webui_models 文件夹 复制到 autodl-tmp 下, 
    因为程序从autodl-tmp/webui_models读取模型, 复制过程约15分钟
2. 如果仍要读取autodl-fs下的sd模型，修改 train_lora/train_dreambooth 的 {sd_path}
    （基本不需要）如果有需要修改blip模型和vae模型的位置，修改 Dreambooth/Lora 中的{self.vae_path/self.blip_path}
3. 单次训练文件夹在autodl-tmp/training-outputs 下的{dir_name}, 训练出的模型在 autodl-fs/webui_models/ 下的lora或者dreambooth 文件夹,可以直接被automatic1111读取  
4. 修改configs 中的模型参数，可以设置多个dict来实现多次训练
5. 如果需要打标签并检查，先将model.run 注释，运行model.prepare， 支持blip 和waifu tagging
6. 如果已经打好标签，可以在{dataset}或者{dir_name} 中， 注释掉model.prepare(), 运行model.run， 不注释标签会覆盖

export http_proxy=http://10.0.0.7:12798 && export https_proxy=http://10.0.0.7:12798
unset http_proxy && unset https_proxy