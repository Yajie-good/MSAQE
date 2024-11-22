import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class BertCNNClassifier(nn.Module):
    def __init__(self, num_labels, mlp_size, bert_output_dim=768, conv_out_channels=256, kernel_sizes=[2, 3]):
        super(BertCNNClassifier, self).__init__()
        BERT_CHI_EXT_dir = '/data/0WYJ/newdata_wyj/CMLTES_codes/multi_class/roformer_v2_chinese_char_base'
        self.bert = AutoModel.from_pretrained(BERT_CHI_EXT_dir)

        # Set up the convolutional layer, here different sized convolutional kernels are used
        self.conv1 = nn.Conv1d(in_channels=bert_output_dim, out_channels=conv_out_channels, kernel_size=kernel_sizes[0], padding=1)
        self.conv2 = nn.Conv1d(in_channels=bert_output_dim, out_channels=conv_out_channels, kernel_size=kernel_sizes[1], padding=1)
        
        # Set weight parameters for global features
        self.num_bert_layers = 13  # Include the initial embedding layer + transformer layer, the total number of layers in the BERT model is 13
        self.layer_weights = nn.Parameter(torch.ones(self.num_bert_layers) / self.num_bert_layers)
       
        # Set the fully connected layer, where in_features needs to be set according to the number of output features of the convolutional layer
        # Fully connected layer before adding local feature splicing, use local feature dimensions as in_features and bert_output_dim as out_features
        self.local_fc = nn.Linear(in_features=2 * 2 * conv_out_channels, out_features=bert_output_dim)


        # Set up the MLP layer to further process the fused features to get the final output logits
        self.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=mlp_size),  # Resize to middle layer features
            nn.ReLU(),  # activation function
            nn.Dropout(0.1),  # Dropout layer to prevent overfitting
            nn.Linear(in_features=mlp_size, out_features=num_labels)  # Output layer, converted to the size of the number of labels
        )



    def forward(self, input_ids, attention_mask=None):
        # BERT global feature extraction using multi-layer outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = bert_outputs.hidden_states  # Get the output of each layer
        # print("Hidden states shape:", [x.shape for x in hidden_states])  # Print the shape of each layer 13 different tensors, each of which is [4, 512, 768]

        # Compute weighted summation of global features
        weighted_hidden_states = torch.stack(hidden_states, dim=0) * self.layer_weights.view(-1, 1, 1, 1) # Reshape the weight tensor into the shape [13, 1, 1, 1] so that it can be multiplied by each element of the hidden state.
        weighted_sum = torch.sum(weighted_hidden_states, dim=0)  # Sum by layer weights 
        # print("Weighted sum shape:", weighted_sum.shape)  # Print the shape of the summed weights weighted_sum has the shape [4, 512, 768], which means that the weighted features were summed for each layer.
        global_feature = weighted_sum[:, 0, :] # Take the [CLS] tagged output as the global representation of the sentence
        # print("Global feature shape:", global_feature.shape)  # Print the shape of the global feature global_feature changes shape to [4, 768], indicating that there are 4 samples of global features, each of which is a 768-dimensional vector.
        
    
        # Convert to a shape suitable for convolution operations (batch_size, num_channels, sequence_length)
        conv_input = weighted_sum.permute(0, 2, 1) ## Adjusts the dimensionality of the BERT output to match the input requirements of the convolutional layer Variable in terms of weighted_sum, which was 0, 1, 2 becomes 0, 2, 1
        # print("Conv input shape:", conv_input.shape)  # Print the shape of the convolutional input [4, 768, 512]

        # Local Feature Extraction by Convolutional Layer CNN Local Feature Extraction
        local_feature1 = F.relu(self.conv1(conv_input))
        local_feature2 = F.relu(self.conv2(conv_input))
        # print("Local feature 1 shape:", local_feature1.shape)  # Print the shape of local feature 1 [4, 256, 513]
        # print("Local feature 2 shape:", local_feature2.shape)  # Print the shape of the local feature 2 [4, 256, 512]
        # Apply Maximum Pooling and Average Pooling
        local_feature1_max = F.max_pool1d(local_feature1, kernel_size=local_feature1.size(2)).squeeze(2)
        local_feature1_avg = F.avg_pool1d(local_feature1, kernel_size=local_feature1.size(2)).squeeze(2)
        local_feature2_max = F.max_pool1d(local_feature2, kernel_size=local_feature2.size(2)).squeeze(2)
        local_feature2_avg = F.avg_pool1d(local_feature2, kernel_size=local_feature2.size(2)).squeeze(2)        
        # print("Local feature 1 max shape:", local_feature1_max.shape)  # Print the shape of the maximum value of local feature 1 [4, 256]
        # print("Local feature 1 avg shape:", local_feature1_avg.shape)  # Print the shape of the mean value of local feature 1 [4, 256]
        # print("Local feature 2 max shape:", local_feature2_max.shape)  # Print the shape of the maximum value of local feature 2 [4, 256]
        # print("Local feature 2 avg shape:", local_feature2_avg.shape)  # Print the shape of the average of local feature 2 [4, 256]

        # Splice of local features
        local_features = torch.cat((local_feature1_max, local_feature1_avg, local_feature2_max, local_feature2_avg), 1)
        # print("Local features shape:", local_features.shape)  # Print the shape of the stitched local features [4,1024]


        # Process local features over fully connected layers
        local_features = self.local_fc(local_features)
        # print("Processed local features shape:", local_features.shape)  # Print the shape of the processed local features [4, 768]

        
        # print("Global feature shape:", global_feature.shape)  # Print the shape of the global feature ([4, 768])
        # print("Local features shape:", local_features.shape)  # Print the shape of a local feature ([4, 768])


        # Concat global and local features
        combined_feature = torch.cat((global_feature, local_features), 1)
        # print("Combined feature shape:", combined_feature.shape)  # Print the shape of the assembled features after stitching [4, 1536]

        # classifier
        logits = self.classifier(combined_feature)
        # print("Logits shape:", logits.shape)  # Print the shape of logits

        return logits
