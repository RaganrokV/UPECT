# Universal Pre-trained Energy Consumption Transformer (UPECT)
#### UPECT is a large-scale open-source pre-trained model for predicting trip energy consumption. 

#### To maximize the utility of UPECT for the research community, we provide open-access resources including the pre-trained model (UPECT-40M), all codes, and licensed fine-tuning data. We encourage other researchers to use various EV databases to study UPECT and explore its potential in other domains.

# Framework

![image](https://github.com/RaganrokV/UPECT/assets/73992419/e5724b74-4a8b-4d1f-aacd-700dce2bb595)

# How to start?

#### 1. You can download our pre-training model UPECT-40M on [OneDrive](https://1drv.ms/u/c/284956e407934917/Ed6g9DN4KRFJh5Zbyo50MowByxbMMutr_ExWMJwA2qzWEA?e=IP2TJq)
#### 2. Then, loading a pre-trained model with the following code
```ckpt = torch.load('11-pretrained_incremental_learning/model/UPECT_40M.pt')
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
#### 3. Four downstream tasks we used are saved as .pkl in the data folder.
#### 4. Change the path and start trying!

# How to reproduce？

#### you can follow thses hyperparameters to reproduce the results in our study

|   **    ChargeCar   **  | Training size | Batch size |  Lr  | Epochs |
|:-----------------------:|:-------------:|:----------:|:----:|:------:|
|  Selective fine-tuning  |      70%      |     16     | 1e-4 |    5   |
|                         |      10%      |            | 1e-4 |    5   |
|     Full fine-tuning    |      70%      |            | 1e-5 |   25   |
|                         |               |            |      |        |
|      **    BHD   **     | Training size | Batch size |  Lr  | Epochs |
|  Selective fine-tuning  |      70%      |     64     | 1e-4 |    2   |
|                         |      10%      |            | 1e-4 |    5   |
|     Full fine-tuning    |      70%      |            | 1e-5 |   50   |
|                         |               |            |      |        |
| **    SpritMonitor   ** | Training size | Batch size |  Lr  | Epochs |
|  Selective fine-tuning  |      70%      |     128    | 1e-5 |    4   |
|                         |      10%      |            | 1e-5 |   20   |
|     Full fine-tuning    |      70%      |            | 1e-6 |   10   |
|                         |               |            |      |        |
|      **    VED   **     | Training size | Batch size |  Lr  | Epochs |
|  Selective fine-tuning  |      70%      |     64     | 1e-5 |    4   |
|                         |      10%      |            | 1e-5 |    4   |
|     Full fine-tuning    |      70%      |            | 1e-4 |   50   |


# UPDATE
We uploaded some new baselines and partially fine-tuned the model according to our scenario

# Note: 
#### We anticipate releasing Version 2 by 2026. V2 will be trained on 600,000 trips and feature a flexible architecture, enabling it to extend beyond energy consumption prediction to multi-task scenarios such as trip generation, charging demand prediction, and aging assessment. Once fully tested and debugged, we will upload it to a new repository and update a redirect link here. If you have any ideas or suggestions regarding in-vehicle general artificial intelligence, we welcome your communication and collaboration.

# Data disclosure

##### We publicize trip data for four datasets in the data folder, which are in our pre-processed format. Unfortunately, we are not authorized to disclose the more than 300,000 trip data used for pre-training

# Tips
##### The code for pre-training the model is in file 7. We tried two structures: decoder-only and encoder-decoder structure, and found that decoder-only works better
##### If you are also using data from the Chinese GB32960 technical specification, you can find out how to extract the trips in files 1-5. If your data is collected via OBD, perhaps you can find inspiration for extracting trips from file 8-11
##### The universal framework/transfer learning is implemented in files 14-17.
