# Universal Pre-trained Energy Consumption Transformer (UPECT)
#### UPECT is a large-scale open-source pre-trained model for predicting trip energy consumption. Boosted by 40 million learnable parameters and 300,412 real-world trips, UPECT effectively learns prior knowledge and transferable representations about energy consumption.
#### To maximize the utility of UPECT for the research community, this study provides open-access resources including the pre-trained model (UPECT-40M), all codes, and licensed fine-tuning data. Researchers can freely access open-source repository. This study encourages other researchers to use various EV databases to study UPECT and explore its potential in other domains.

# How to use?

#### 1. You can download our pre-training model UPECT-40M on [OneDrive](https://1drv.ms/u/c/284956e407934917/Ed6g9DN4KRFJh5Zbyo50MowByxbMMutr_ExWMJwA2qzWEA?e=IP2TJq)
#### 2. Then loading a pre-trained model with the following code
```ckpt = torch.load('11-pretrained_incremental_learning/model/UPECT_40M.pt')
% 提取 DataParallel 包装的模型中的实际模型参数
state_dict = ckpt["model_state_dict"]
new_state_dict = {}
for key, value in state_dict.items():
    if key.startswith('module.'):
        new_key = key[7:]  
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value
UPECT.load_state_dict(new_state_dict)
```
#### 3. Four downstream tasks we used are saved as .pkl in the data folder, change the path and start trying!


##### The code for pre-training the model is in file 7. We tried two structures decoder-only and encoder-decoder structure, and found that decoder-only works better
##### If you are also using data from the Chinese GB32960 technical specification, you can find out how to extract the trips in files 1-5. If your data is collected via OBD, perhaps you can find inspiration for extracting trips from file 8-11
##### The universal framework is implemented in files 14-17.
