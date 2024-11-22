from transformers import AutoConfig

# If you have a local path, make sure the path is correct and that there are no extra spaces in the path string
BERT_CHI_EXT_dir = '/data/0WYJ/newdata_wyj/CMLTES_codes/multi_class/ roformer_v2_chinese_char_base'

# Loading configurations from pre-trained model paths
config = AutoConfig.from_pretrained(BERT_CHI_EXT_dir)

# Print the number of hidden layers of the model
print("The model has", config.num_hidden_layers, "hidden layers.")
