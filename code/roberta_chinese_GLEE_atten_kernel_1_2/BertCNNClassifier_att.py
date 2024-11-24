import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import math


class MultiHeadedAttention(nn.Module):
    def __init__(self, h=8, n_hidden=768, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert n_hidden % h == 0
        # We assume d_v always equals d_k
        self.d_k = n_hidden // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # On the cpu.
        # scores = scores.cuda()  # Prevent equipment inconsistencies
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        residual, nbatches = query, query.size(0)

        # 1) Do all the linear projections in batch from n_hidden => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x + residual)  # Here the two dimensions are required to be the same


class LayerNorm(nn.Module):
    
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class BertCNNClassifier_att(nn.Module):
    def __init__(self, num_labels, mlp_size, bert_output_dim=768, conv_out_channels=256, kernel_sizes=[1, 2], d_model=768, d_k=96, d_v=96, n_heads=8):
        super(BertCNNClassifier_att, self).__init__()
        BERT_CHI_EXT_dir = '/data/0WYJ/newdata_wyj/CMLTES_codes/multi_class/ROBERTA_RRE_LARGE'
        self.bert = AutoModel.from_pretrained(BERT_CHI_EXT_dir)

        self.n_heads = n_heads
        self.conv1 = nn.Conv1d(in_channels=bert_output_dim, out_channels=conv_out_channels, kernel_size=kernel_sizes[0], padding=1)
        self.conv2 = nn.Conv1d(in_channels=bert_output_dim, out_channels=conv_out_channels, kernel_size=kernel_sizes[1], padding=1)
        
        self.num_bert_layers = 13  # Include the initial embedding layer + transformer layer, the total number of layers in the BERT model is 13
        self.layer_weights = nn.Parameter(torch.ones(self.num_bert_layers) / self.num_bert_layers)

        self.attention_global = MultiHeadedAttention(n_heads, d_model)
        self.attention_local = MultiHeadedAttention(n_heads, d_model)
        
        self.local_fc = nn.Linear(in_features=2 * 2 * conv_out_channels, out_features=bert_output_dim)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1536, out_features=mlp_size),
            LayerNorm(mlp_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features=mlp_size, out_features=num_labels)
        )

    def forward(self, input_ids, attention_mask=None):
        '''
        input_ids: [4, 512]
        attention_mask: [4, 512]
        '''
        # BERT global feature extraction using multi-layer outputs
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = bert_outputs.hidden_states  # 
        weighted_hidden_states = torch.stack(hidden_states, dim=0) * self.layer_weights.view(-1, 1, 1, 1) # [13, 4, 512, 768]
        weighted_sum = torch.sum(weighted_hidden_states, dim=0)  # [4, 512, 768]
        global_feature = weighted_sum[:, 0, :] # Take the [CLS]-tagged output as the global representation of the sentence [4, 768]
        
        conv_input = weighted_sum.permute(0, 2, 1) ## Adjusts the dimensionality of the BERT output to match the input requirements of the convolutional layer Variable in terms of weighted_sum, originally 0, 1, 2 becomes 0, 2, 1 [4, 768, 512]

        local_feature1 = F.relu(self.conv1(conv_input))  # [4, 256, 513]
        local_feature2 = F.relu(self.conv2(conv_input))  # [4, 256, 512]

        local_feature1_max = F.max_pool1d(local_feature1, kernel_size=local_feature1.size(2)).squeeze(2) # [4, 256]
        local_feature1_avg = F.avg_pool1d(local_feature1, kernel_size=local_feature1.size(2)).squeeze(2) # [4, 256]
        local_feature2_max = F.max_pool1d(local_feature2, kernel_size=local_feature2.size(2)).squeeze(2) # [4, 256]
        local_feature2_avg = F.avg_pool1d(local_feature2, kernel_size=local_feature2.size(2)).squeeze(2) # [4, 256]

        local_features = torch.cat((local_feature1_max, local_feature1_avg, local_feature2_max, local_feature2_avg), 1)  # [4,1024]

        local_features = self.local_fc(local_features)  # [4, 768]

        attn_output_global = self.attention_global(global_feature, local_features, local_features) # torch.Size([4, 4, 768])
        attn_output_local = self.attention_local(local_features, global_feature, global_feature) # torch.Size([4, 5, 768])
        attn_output_global = torch.mean(attn_output_global, dim=1, keepdim=False)  # torch.Size([4, 1, 768])
        attn_output_local = torch.mean(attn_output_local, dim=1, keepdim=False)  # torch.Size([4, 1, 768])
        
        combined_attn_output = torch.cat((attn_output_global.squeeze(1), attn_output_local.squeeze(1)), 1) # torch.Size([4, 1536])
        
        logits = self.classifier(combined_attn_output)  # torch.Size([4, 8])
        # print(logits.shape)
        # exit()
        return logits
