# Universal Pre-trained Energy Consumption Transformer (UPECT)
### UPECT is a large-scale open-source pre-trained model for predicting trip energy consumption. Boosted by 40 million learnable parameters and 300,412 real-world trips, UPECT effectively learns prior knowledge and transferable representations about energy consumption.

# How to use?
### The code for pre-training the model is in file 7. We tried two structures decoder-only and encoder-decoder structure, and found that decoder-only works better
### If you are also using data from the Chinese GB32960 technical specification, you can find out how to extract the trips in files 1-5. If your data is collected via OBD, perhaps you can find inspiration for extracting trips from file 8-11
