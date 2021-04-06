import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention

import numpy as np

class LSTMAttention(nn.Module):
    def __init__(self, config, feature_type='acoustic'):
        super().__init__()
        assert feature_type in ['acoustic','semantic']
        self.feature_type = feature_type
        if self.feature_type == 'acoustic':
            self.acoustic_lstm = nn.LSTM(
                config['acoustic']['embedding_dim'],
                config['acoustic']['hidden_dim'], 1, bidirectional=True,
                batch_first=True, dropout=0.5
            )

            self.classifier = nn.Linear(
                2*config['semantic']['hidden_dim'], 
                config['classifier']['class_num']
            )
            
            self.attention = nn.Parameter(
                torch.randn(2*config['acoustic']['hidden_dim'])
            )

        else:
            if config['semantic']['embedding_path'] is not None:
                semantic_embed = np.load(config['semantic']['embedding_path'])
                semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
                self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
            else:
                self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])

            self.semantic_lstm = nn.LSTM(
                config['semantic']['embedding_dim'],
                config['semantic']['hidden_dim'], 1, bidirectional=True,
                batch_first=True, dropout=0.5
            )

            self.classifier = nn.Linear(
                2*config['semantic']['hidden_dim'], 
                config['classifier']['class_num']
            )

            self.attention = nn.Parameter(
                torch.randn(2*config['semantic']['hidden_dim'])
            )

        self.loss_name = config['loss']['name']

        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        if self.feature_type == 'acoustic':
            # use the rnn for embedding
            acoustic_pack = nn.utils.rnn.pack_padded_sequence(
                acoustic_input, acoustic_length.cpu(), batch_first=True, enforce_sorted=False
            )
            acoustic_embed, _ = self.acoustic_lstm(acoustic_pack)
            acoustic_embed, _ = nn.utils.rnn.pad_packed_sequence(acoustic_embed, batch_first=True) # [B,A,D]

            # mask some of the attention score # [B,A,1]
            attention_mask = torch.arange(
                acoustic_input.size(1))[None,:].repeat(acoustic_input.size(0),1
            ).to(acoustic_input.device)
            attention_mask = (attention_mask<acoustic_length[:,None].repeat(1,acoustic_input.size(1))).float()[:,:,None] # [B,A,1]
        else:
            semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
            semantic_pack = nn.utils.rnn.pack_padded_sequence(
                semantic_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
            )
            semantic_embed, _ = self.semantic_lstm(semantic_pack)
            semantic_embed, _ = nn.utils.rnn.pad_packed_sequence(semantic_embed, batch_first=True) # [B,A,D]

            # mask some of the attention score # [B,A,1]
            attention_mask = torch.arange(
                semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
            ).to(semantic_input.device)
            attention_mask = (attention_mask<semantic_length[:,None].repeat(1,semantic_input.size(1))).float()[:,:,None] # [B,A,1]

        if self.loss_name == 'BCE':
            if self.feature_type == 'acoustic':
                # Then we need to use attention to find the result
                attention_score = torch.matmul(acoustic_embed, self.attention[None,:,None].repeat(acoustic_input.size(0),1,1)) #[B,A,1]
            else:
                attention_score = torch.matmul(semantic_embed, self.attention[None,:,None].repeat(semantic_input.size(0),1,1)) #[B,A,1]

            attention_score = attention_score / np.sqrt(self.attention.size(0))

            attention_score = attention_score*attention_mask - 1e6*(1-attention_mask)
            attention_score = F.softmax(attention_score, dim=1)

            if self.feature_type == 'acoustic':
                acoustic_embed = torch.matmul(attention_score.permute(0,2,1),acoustic_embed).squeeze(1) # [B,D]
                logits = self.classifier(acoustic_embed)
            else:
                semantic_embed = torch.matmul(attention_score.permute(0,2,1),semantic_embed).squeeze(1) # [B,D]
                logits = self.classifier(semantic_embed)
        
        elif self.loss_name == 'CTC':
            if self.feature_type == 'acoustic':
                logits = self.classifier(acoustic_embed) # [B,T,Dim]
            else:
                logits = self.classifier(semantic_embed) # [B,T,Dim]
            
            logits = F.log_softmax(logits)
            logits = logits * attention_mask

        else:
            raise ValueError('Loss type not supported!')

        return logits


class NeoMHA2(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Here we define the two different encoder - acoustic encoder
        self.acoustic_lstm = nn.LSTM(
            config['acoustic']['embedding_dim'],
            config['acoustic']['hidden_dim'], 1, bidirectional=True,
            batch_first=True, dropout=0.5
        )
        # Here we define the two different encoder - semantic encoder
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])

        self.semantic_lstm = nn.LSTM(
            config['semantic']['embedding_dim'],
            config['semantic']['hidden_dim'], 1, bidirectional=True,
            batch_first=True, dropout=0.5
        )

        self.classifier = nn.Linear(
            2*config['semantic']['hidden_dim']+2*2*config['semantic']['hidden_dim'], 
            config['classifier']['class_num']
        )

        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        acoustic_pack = nn.utils.rnn.pack_padded_sequence(
            acoustic_input, acoustic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        acoustic_embed, acoustic_hidden = self.acoustic_lstm(acoustic_pack)
        acoustic_embed, _ = nn.utils.rnn.pad_packed_sequence(acoustic_embed, batch_first=True) # [B,A,D]
        acoustic_hidden_0 = acoustic_hidden[0].view([-1,1,acoustic_embed.size(2)])[:,-1,:] # # [B,Dim*2]

        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_pack = nn.utils.rnn.pack_padded_sequence(
            semantic_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        semantic_embed, semantic_hidden = self.semantic_lstm(semantic_pack)
        semantic_embed, _ = nn.utils.rnn.pad_packed_sequence(semantic_embed, batch_first=True) # [B,A,D]
        # semantic_hidden_0 = semantic_hidden[0].view([-1,1,semantic_embed.size(2)])[:,-1,:]

        a1 = torch.matmul(acoustic_hidden_0[:,None,:],semantic_embed.permute(0,2,1))
        a1_mask = torch.arange(semantic_embed.size(1))[None,:].repeat(semantic_embed.size(0),1).to(semantic_embed.device)
        a1_mask = (a1_mask < semantic_length[:,None].repeat(1,semantic_embed.size(1))).float().unsqueeze(1)
        a1_mask = (1.0 - a1_mask) * -10000.0

        a1 = a1 + a1_mask 
        a1 = nn.Softmax(dim=-1)(a1)
        semantic_hidden_0 = torch.matmul(a1,semantic_embed).squeeze(1)

        fuse_hidden_0 = torch.cat([acoustic_hidden_0, semantic_hidden_0],dim=-1)

        # Here we apply the second hop attention
        a2 = torch.matmul(semantic_hidden_0[:,None,:],acoustic_embed.permute(0,2,1))
        a2_mask = torch.arange(acoustic_embed.size(1))[None,:].repeat(acoustic_embed.size(0),1).to(acoustic_embed.device)
        a2_mask = (a2_mask < acoustic_length[:,None].repeat(1,acoustic_embed.size(1))).float().unsqueeze(1)
        a2_mask = (1.0 - a2_mask) * -10000.0

        a2 = a2 + a2_mask
        a2 = nn.Softmax(dim=-1)(a2)
        acoustic_hidden_1 = torch.matmul(a2,acoustic_embed).squeeze(1)
        
        fuse_hidden_1 = torch.cat([fuse_hidden_0, acoustic_hidden_1],dim=-1)

        logits = self.classifier(fuse_hidden_1)

        return logits


class NeoExcite(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])
        
        self.semantic_linear = nn.Linear(config['semantic']['embedding_dim'], config['semantic']['hidden_dim'])
        # This the embedding for the audio features
        # self.acoustic_cnn1 = nn.Conv1d(34,64,5,1)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'],64,5,1)
        self.acoustic_cnn2 = nn.Conv1d(64,128,2,1)
        self.acoustic_cnn3 = nn.Conv1d(128,config['acoustic']['hidden_dim'],2,1)
        self.acoustic_mean1 = nn.AvgPool2d((1,5),(1,1))
        self.acoustic_mean2 = nn.AvgPool2d((1,2),(1,1))
        self.acoustic_mean3 = nn.AvgPool2d((1,2),(1,1))

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']+config['acoustic']['hidden_dim'],
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        # Add the cross-modal excitement layer
        self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size']+1, config['acoustic']['hidden_dim'])
        self.semantic_excit = nn.Linear(config['acoustic']['hidden_dim'], config['semantic']['hidden_dim'])
        
        self.classifier = nn.Linear(2*config['fusion']['hidden_dim'],config['classifier']['class_num'])
        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0,2,1))
        acoustic_align = self.acoustic_mean1(align_input[:,None,:,:])
        
        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_align = self.acoustic_mean2(acoustic_align)
        
        acoustic_embed = self.acoustic_cnn3(acoustic_embed) # [B,C,A]
        acoustic_align = self.acoustic_mean3(acoustic_align) # [B,1,T,A]

        ## for the align result, we need to normalize it into summation equals 1
        acoustic_align = acoustic_align - (acoustic_align == 0).float() * 1e6
        acoustic_align = F.softmax(acoustic_align, dim=3)
        ## based on that align results we put the feature into the alignment results
        acoustic_embed = torch.matmul(
            torch.squeeze(acoustic_align,1),acoustic_embed.permute(0,2,1)) # [B,T,C]
        # then we use the cross modal excitement information
        acoustic_excit = F.sigmoid(self.acoustic_excit(semantic_input))
        semantic_excit = F.sigmoid(self.semantic_excit(acoustic_embed))
        acoustic_embed = acoustic_embed * acoustic_excit
        semantic_embed = semantic_embed * semantic_excit
        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
        ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,semantic_input.size(1))).float()
        fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
        fuse_embed = torch.max(fuse_embed, dim=1)[0]

        logits = self.classifier(fuse_embed)
        return logits


class NeoMeanMaxExcite(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])
        
        self.semantic_linear = nn.Linear(config['semantic']['embedding_dim'], config['semantic']['hidden_dim'])
        # This the embedding for the audio features
        # self.acoustic_cnn1 = nn.Conv1d(34,64,5,1)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'],64,5,1)
        self.acoustic_cnn2 = nn.Conv1d(64,128,2,1)
        self.acoustic_cnn3 = nn.Conv1d(128,int(config['acoustic']['hidden_dim']/2),2,1)
        self.acoustic_mean1 = nn.AvgPool2d((1,5),(1,1))
        self.acoustic_mean2 = nn.AvgPool2d((1,2),(1,1))
        self.acoustic_mean3 = nn.AvgPool2d((1,2),(1,1))

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']+config['acoustic']['hidden_dim'],
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        # Add the cross-modal excitement layer
        if config['acoustic']['excite']:
            self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size']+1, config['acoustic']['hidden_dim'])
        else:
            self.acoustic_excit = None
        if config['semantic']['excite']:
            self.semantic_excit = nn.Linear(config['acoustic']['hidden_dim'], config['semantic']['hidden_dim'])
        else:
            self.semantic_excit = None
        
        self.loss_name = config['loss']['name']
        self.classifier = nn.Linear(
            2*config['fusion']['hidden_dim'], 
            config['classifier']['class_num']+1*int(self.loss_name=='CTC')
        )
        
        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0,2,1))
        acoustic_align = self.acoustic_mean1(align_input[:,None,:,:])
        
        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_align = self.acoustic_mean2(acoustic_align)
        
        acoustic_embed = self.acoustic_cnn3(acoustic_embed) # [B,C,A]
        acoustic_align = self.acoustic_mean3(acoustic_align) # [B,1,T,A]

        acoustic_embed = acoustic_embed.permute(0,2,1)[:,None,:,:].repeat(1,semantic_embed.size(1),1,1) # [B,T,A,C]
        acoustic_align = (acoustic_align.squeeze(1)[:,:,:,None] > 0).float() # [B,T,A,1]
        # Think about the new way of calculate the mean
        acoustic_mean = torch.sum(acoustic_embed*acoustic_align, dim=2) / (torch.sum(acoustic_align, dim=2) + 1e-6) # [B,T,C]
        # Think about the new way of calculate the max
        acoustic_max, _ = torch.max(acoustic_embed*acoustic_align-1e6*(1.0-acoustic_align), dim=2)
        # concat it with both embed
        acoustic_embed = torch.cat([acoustic_mean,acoustic_max], dim=-1)

        # then we use the cross modal excitement information
        if self.semantic_excit is not None:
            semantic_excit = F.sigmoid(self.semantic_excit(acoustic_embed))
            semantic_embed = semantic_embed * semantic_excit
        if self.acoustic_excit is not None:
            acoustic_excit = F.sigmoid(self.acoustic_excit(semantic_input))
            acoustic_embed = acoustic_embed * acoustic_excit
        
        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
        ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,semantic_input.size(1))).float()

        if self.loss_name == 'BCE':
            fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
            fuse_embed = torch.max(fuse_embed, dim=1)[0]
            logits = self.classifier(fuse_embed)
        
        elif self.loss_name == 'CTC':
            logits = self.classifier(fuse_embed) # [B,T,Dim]
            logits = F.log_softmax(logits)
            logits = logits * fuse_mask[:,:,None]

        else:
            raise ValueError('Loss type not supported!')

        return logits


class NeoDiDi(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])

        self.semantic_lstm = nn.LSTM(
            config['semantic']['embedding_dim'],
            config['semantic']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        self.acoustic_lstm = nn.LSTM(
            config['acoustic']['embedding_dim'],
            config['acoustic']['hidden_dim'], 1, bidirectional=True,
            batch_first=True, dropout=0.5
        )

        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']*2+config['acoustic']['hidden_dim']*2,
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        self.align_attention = MultiheadAttention(
            2*config['semantic']['hidden_dim'], 
            config['fusion']['num_heads'], dropout=0.5
        )

        self.classifier = nn.Linear(
            2*config['fusion']['hidden_dim'], 
            config['classifier']['class_num']
        )
        

        self.loss_name = config['loss']['name']

    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,
    ):
    # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_pack = nn.utils.rnn.pack_padded_sequence(
            semantic_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        semantic_embed, _ = self.semantic_lstm(semantic_pack)
        semantic_embed, _ = nn.utils.rnn.pad_packed_sequence(semantic_embed, batch_first=True)
        # first perform the encode for the first-step acoustic partterns
        acoustic_pack = nn.utils.rnn.pack_padded_sequence(
            acoustic_input, acoustic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        acoustic_embed, _ = self.acoustic_lstm(acoustic_pack)
        acoustic_embed, _ = nn.utils.rnn.pad_packed_sequence(acoustic_embed, batch_first=True)
        
        # print(semantic_embed.shape)
        # print(acoustic_embed.shape)
        
        acoustic_embed, _ = self.align_attention(
            semantic_embed.permute(1,0,2), acoustic_embed.permute(1,0,2), acoustic_embed.permute(1,0,2),
        )
        acoustic_embed = acoustic_embed.permute(1,0,2)

        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
        ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,semantic_input.size(1))).float()

        if self.loss_name == 'BCE':
            fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
            fuse_embed = torch.max(fuse_embed, dim=1)[0]
            logits = self.classifier(fuse_embed)
        
        elif self.loss_name == 'CTC':
            logits = self.classifier(fuse_embed) # [B,T,Dim]
            logits = F.log_softmax(logits)
            logits = logits * fuse_mask[:,:,None]

        else:
            raise ValueError('Loss type not supported!')

        return logits


class NeoMeanMaxExcite_v2(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])
        
        self.semantic_linear = nn.Linear(config['semantic']['embedding_dim'], config['semantic']['hidden_dim'])
        # This the embedding for the audio features
        # self.acoustic_cnn1 = nn.Conv1d(34,64,5,1)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'],64,5,1)
        self.acoustic_cnn2 = nn.Conv1d(64,128,2,1)
        self.acoustic_cnn3 = nn.Conv1d(128,int(config['acoustic']['hidden_dim']/2),2,1)
        self.acoustic_mean1 = nn.AvgPool2d((1,5),(1,1))
        self.acoustic_mean2 = nn.AvgPool2d((1,2),(1,1))
        self.acoustic_mean3 = nn.AvgPool2d((1,2),(1,1))

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']+config['acoustic']['hidden_dim'],
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        # Add the cross-modal excitement layer
        if config['acoustic']['excite']:
            self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size']+1, config['acoustic']['hidden_dim'])
        else:
            self.acoustic_excit = None
        if config['semantic']['excite']:
            self.semantic_excit = nn.Linear(config['acoustic']['hidden_dim'], config['semantic']['hidden_dim'])
        else:
            self.semantic_excit = None
        
        self.loss_name = config['loss']['name']
        self.classifier = nn.Linear(
            2*config['fusion']['hidden_dim'], 
            config['classifier']['class_num']+1*int(self.loss_name=='CTC')
        )
        
        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0,2,1))
        acoustic_align = self.acoustic_mean1(align_input[:,None,:,:])
        
        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_align = self.acoustic_mean2(acoustic_align)
        
        acoustic_embed = self.acoustic_cnn3(acoustic_embed) # [B,C,A]
        acoustic_align = self.acoustic_mean3(acoustic_align) # [B,1,T,A]

        acoustic_embed = acoustic_embed.permute(0,2,1)[:,None,:,:].repeat(1,semantic_embed.size(1),1,1) # [B,T,A,C]
        acoustic_align = (acoustic_align.squeeze(1)[:,:,:,None] > 0).float() # [B,T,A,1]
        # Think about the new way of calculate the mean
        acoustic_mean = torch.sum(acoustic_embed*acoustic_align, dim=2) / (torch.sum(acoustic_align, dim=2) + 1e-6) # [B,T,C]
        # Think about the new way of calculate the max
        acoustic_max, _ = torch.max(acoustic_embed*acoustic_align-1e6*(1.0-acoustic_align), dim=2)
        # concat it with both embed
        acoustic_embed = torch.cat([acoustic_mean,acoustic_max], dim=-1)

        # then we use the cross modal excitement information
        if self.semantic_excit is not None:
            semantic_excit = F.sigmoid(self.semantic_excit(acoustic_embed))
            semantic_embed = semantic_embed * semantic_excit + semantic_embed # These two lines are different, we add the residual connection
        if self.acoustic_excit is not None:
            acoustic_excit = F.sigmoid(self.acoustic_excit(semantic_input))
            acoustic_embed = acoustic_embed * acoustic_excit + acoustic_embed # These two lines are different, we add the residual connection

        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
        ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,semantic_input.size(1))).float()

        if self.loss_name == 'BCE':
            fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
            fuse_embed = torch.max(fuse_embed, dim=1)[0]
            logits = self.classifier(fuse_embed)
        
        elif self.loss_name == 'CTC':
            logits = self.classifier(fuse_embed) # [B,T,Dim]
            logits = F.log_softmax(logits)
            logits = logits * fuse_mask[:,:,None]

        else:
            raise ValueError('Loss type not supported!')

        return logits


class NeoMeanMaxExcite_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])
        
        self.semantic_linear = nn.Linear(config['semantic']['embedding_dim'], config['semantic']['hidden_dim'])
        # This the embedding for the audio features
        # self.acoustic_cnn1 = nn.Conv1d(34,64,5,1)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'],64,5,1)
        self.acoustic_cnn2 = nn.Conv1d(64,128,2,1)
        self.acoustic_cnn3 = nn.Conv1d(128,int(config['acoustic']['hidden_dim']/2),2,1)
        self.acoustic_mean1 = nn.AvgPool2d((1,5),(1,1))
        self.acoustic_mean2 = nn.AvgPool2d((1,2),(1,1))
        self.acoustic_mean3 = nn.AvgPool2d((1,2),(1,1))

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']+config['acoustic']['hidden_dim'],
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        # Add the cross-modal excitement layer
        if config['acoustic']['excite']:
            self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size']+1, config['acoustic']['hidden_dim'])
        else:
            self.acoustic_excit = None
        if config['semantic']['excite']:
            self.semantic_excit = nn.Linear(config['acoustic']['hidden_dim'], config['semantic']['hidden_dim'])
        else:
            self.semantic_excit = None
        
        self.loss_name = config['loss']['name']
        self.classifier = nn.Linear(
            2*config['fusion']['hidden_dim'], 
            config['classifier']['class_num']+1*int(self.loss_name=='CTC')
        )
        
        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0,2,1))
        acoustic_align = self.acoustic_mean1(align_input[:,None,:,:])
        
        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_align = self.acoustic_mean2(acoustic_align)
        
        acoustic_embed = self.acoustic_cnn3(acoustic_embed) # [B,C,A]
        acoustic_align = self.acoustic_mean3(acoustic_align) # [B,1,T,A]

        acoustic_embed = acoustic_embed.permute(0,2,1)[:,None,:,:].repeat(1,semantic_embed.size(1),1,1) # [B,T,A,C]
        acoustic_align = (acoustic_align.squeeze(1)[:,:,:,None] > 0).float() # [B,T,A,1]
        # Think about the new way of calculate the mean
        acoustic_mean = torch.sum(acoustic_embed*acoustic_align, dim=2) / (torch.sum(acoustic_align, dim=2) + 1e-6) # [B,T,C]
        # Think about the new way of calculate the max
        acoustic_max, _ = torch.max(acoustic_embed*acoustic_align-1e6*(1.0-acoustic_align), dim=2)
        # concat it with both embed
        acoustic_embed = torch.cat([acoustic_mean,acoustic_max], dim=-1)

        # then we use the cross modal excitement information
        if self.semantic_excit is not None:
            semantic_excit = self.semantic_excit(acoustic_embed)
            semantic_embed = semantic_embed + semantic_excit
        if self.acoustic_excit is not None:
            acoustic_excit = self.acoustic_excit(semantic_input)
            acoustic_embed = acoustic_embed + acoustic_excit

        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
        ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,semantic_input.size(1))).float()

        if self.loss_name == 'BCE':
            fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
            fuse_embed = torch.max(fuse_embed, dim=1)[0]
            logits = self.classifier(fuse_embed)
        
        elif self.loss_name == 'CTC':
            logits = self.classifier(fuse_embed) # [B,T,Dim]
            logits = F.log_softmax(logits)
            logits = logits * fuse_mask[:,:,None]

        else:
            raise ValueError('Loss type not supported!')

        return logits


class NeoMeanMaxExcite_v4(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])
        
        self.semantic_linear = nn.Linear(config['semantic']['embedding_dim'], config['semantic']['hidden_dim'])
        # This the embedding for the audio features
        # self.acoustic_cnn1 = nn.Conv1d(34,64,5,1)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'],64,5,1)
        self.acoustic_cnn2 = nn.Conv1d(64,128,2,1)
        self.acoustic_cnn3 = nn.Conv1d(128,int(config['acoustic']['hidden_dim']/2),2,1)
        self.acoustic_mean1 = nn.AvgPool2d((1,5),(1,1))
        self.acoustic_mean2 = nn.AvgPool2d((1,2),(1,1))
        self.acoustic_mean3 = nn.AvgPool2d((1,2),(1,1))

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']+config['acoustic']['hidden_dim'],
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        # Add the cross-modal excitement layer
        if config['acoustic']['excite']:
            self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size']+1, config['acoustic']['hidden_dim'])
        else:
            self.acoustic_excit = None
        if config['semantic']['excite']:
            self.semantic_excit = nn.Linear(config['acoustic']['hidden_dim'], config['semantic']['hidden_dim'])
        else:
            self.semantic_excit = None
        
        self.loss_name = config['loss']['name']
        self.classifier = nn.Linear(
            2*config['fusion']['hidden_dim'], 
            config['classifier']['class_num']+1*int(self.loss_name=='CTC')
        )
        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0,2,1))
        acoustic_align = self.acoustic_mean1(align_input[:,None,:,:])
        
        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_align = self.acoustic_mean2(acoustic_align)
        
        acoustic_embed = self.acoustic_cnn3(acoustic_embed) # [B,C,A]
        acoustic_align = self.acoustic_mean3(acoustic_align) # [B,1,T,A]

        acoustic_embed = acoustic_embed.permute(0,2,1)[:,None,:,:].repeat(1,semantic_embed.size(1),1,1) # [B,T,A,C]
        acoustic_align = (acoustic_align.squeeze(1)[:,:,:,None] > 0).float() # [B,T,A,1]
        # Think about the new way of calculate the mean
        acoustic_mean = torch.sum(acoustic_embed*acoustic_align, dim=2) / (torch.sum(acoustic_align, dim=2) + 1e-6) # [B,T,C]
        # Think about the new way of calculate the max
        acoustic_max, _ = torch.max(acoustic_embed*acoustic_align-1e6*(1.0-acoustic_align), dim=2)
        # concat it with both embed
        acoustic_embed = torch.cat([acoustic_mean,acoustic_max], dim=-1)

        # then we use the cross modal excitement information
        if self.semantic_excit is not None:
            semantic_excit = self.semantic_excit(acoustic_embed)
            semantic_embed = semantic_embed + semantic_excit
        if self.acoustic_excit is not None:
            acoustic_excit = F.sigmoid(self.acoustic_excit(semantic_input))
            acoustic_embed = acoustic_embed * acoustic_excit

        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
        ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,semantic_input.size(1))).float()

        if self.loss_name == 'BCE':
            fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
            fuse_embed = torch.max(fuse_embed, dim=1)[0]
            logits = self.classifier(fuse_embed)
        
        elif self.loss_name == 'CTC':
            logits = self.classifier(fuse_embed) # [B,T,Dim]
            logits = F.log_softmax(logits)
            logits = logits * fuse_mask[:,:,None]

        else:
            raise ValueError('Loss type not supported!')

        return logits

# This is what we copy from the Multimodal Transformer
# To implement that idea, probably we need to pad the sequence into the same length?
class CTCModule(nn.Module):
    def __init__(self, in_dim, out_seq_len):
        '''
        This module is performing alignment from A (e.g., audio) to B (e.g., text).
        :param in_dim: Dimension for input modality A
        :param out_seq_len: Sequence length for output modality B
        '''
        super(CTCModule, self).__init__()
        # Use LSTM for predicting the position from A to B
        self.pred_output_position_inclu_blank = nn.LSTM(in_dim, out_seq_len+1, num_layers=2, batch_first=True) # 1 denoting blank
        
        self.out_seq_len = out_seq_len
        
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        '''
        :input x: Input with shape [batch_size x in_seq_len x in_dim]
        '''
        # NOTE that the index 0 refers to blank. 
        pred_output_position_inclu_blank, _ = self.pred_output_position_inclu_blank(x)

        prob_pred_output_position_inclu_blank = self.softmax(pred_output_position_inclu_blank) # batch_size x in_seq_len x out_seq_len+1
        prob_pred_output_position = prob_pred_output_position_inclu_blank[:, :, 1:] # batch_size x in_seq_len x out_seq_len
        prob_pred_output_position = prob_pred_output_position.transpose(1,2) # batch_size x out_seq_len x in_seq_len
        pseudo_aligned_out = torch.bmm(prob_pred_output_position, x) # batch_size x out_seq_len x in_dim
        
        # pseudo_aligned_out is regarded as the aligned A (w.r.t B)
        return pseudo_aligned_out, (pred_output_position_inclu_blank)



class NeoMeanMaxExciteCTC(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])
        
        self.semantic_linear = nn.Linear(config['semantic']['embedding_dim'], config['semantic']['hidden_dim'])
        # This the embedding for the audio features
        # self.acoustic_cnn1 = nn.Conv1d(34,64,5,1)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'],64,5,1)
        self.acoustic_cnn2 = nn.Conv1d(64,128,2,1)
        self.acoustic_cnn3 = nn.Conv1d(128,config['acoustic']['hidden_dim'],2,1)
        self.acoustic_mean1 = nn.AvgPool2d((1,5),(1,1))
        self.acoustic_mean2 = nn.AvgPool2d((1,2),(1,1))
        self.acoustic_mean3 = nn.AvgPool2d((1,2),(1,1))

        self.align_ctc = CTCModule(config['acoustic']['hidden_dim'],config['semantic']['max_length'])
        self.logsoftmax = nn.LogSoftmax(dim=2)

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']+config['acoustic']['hidden_dim'],
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        # Add the cross-modal excitement layer
        if config['acoustic']['excite']:
            self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size']+1, config['acoustic']['hidden_dim'])
        else:
            self.acoustic_excit = None
        if config['semantic']['excite']:
            self.semantic_excit = nn.Linear(config['acoustic']['hidden_dim'], config['semantic']['hidden_dim'])
        else:
            self.semantic_excit = None
        
        self.loss_name = config['loss']['name']
        self.classifier = nn.Linear(
            2*config['fusion']['hidden_dim'], 
            config['classifier']['class_num']
        )
        
        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0,2,1))
        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_embed = self.acoustic_cnn3(acoustic_embed) # [B,C,A]

        acoustic_embed, align_logits = self.align_ctc(acoustic_embed.permute(0,2,1))
        align_logits = self.logsoftmax(align_logits)

        # then we use the cross modal excitement information
        if self.semantic_excit is not None:
            semantic_excit = self.semantic_excit(acoustic_embed)
            semantic_embed = semantic_embed + semantic_excit
        if self.acoustic_excit is not None:
            acoustic_excit = self.acoustic_excit(semantic_input)
            acoustic_embed = acoustic_embed + acoustic_excit

        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            fuse_embed.size(1))[None,:].repeat(fuse_embed.size(0),1
        ).to(fuse_embed.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,fuse_embed.size(1))).float()

        fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
        fuse_embed = torch.max(fuse_embed, dim=1)[0]
        logits = self.classifier(fuse_embed)
        
        return logits, align_logits



class NeoMeanMaxExciteVisual_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['semantic']['embedding_path'] is not None:
            semantic_embed = np.load(config['semantic']['embedding_path'])
            semantic_embed = np.concatenate([np.zeros([1,semantic_embed.shape[1]]),semantic_embed],axis=0)
            self.semantic_embed = nn.Embedding.from_pretrained(torch.FloatTensor(semantic_embed), freeze=False)
        else:
            self.semantic_embed = nn.Embedding(config['semantic']['embedding_size']+1, config['semantic']['embedding_dim'])
        
        self.semantic_linear = nn.Linear(config['semantic']['embedding_dim'], config['semantic']['hidden_dim'])
        # This the embedding for the audio features
        # self.acoustic_cnn1 = nn.Conv1d(34,64,5,1)
        self.acoustic_cnn1 = nn.Conv1d(config['acoustic']['embedding_dim'],64,5,1)
        self.acoustic_cnn2 = nn.Conv1d(64,128,2,1)
        self.acoustic_cnn3 = nn.Conv1d(128,int(config['acoustic']['hidden_dim']/2),2,1)
        self.acoustic_mean1 = nn.AvgPool2d((1,5),(1,1))
        self.acoustic_mean2 = nn.AvgPool2d((1,2),(1,1))
        self.acoustic_mean3 = nn.AvgPool2d((1,2),(1,1))

        # This the embedding for the semantic features
        self.fuse_lstm = nn.LSTM(
            config['semantic']['hidden_dim']+config['acoustic']['hidden_dim'],
            config['fusion']['hidden_dim'], 1, bidirectional=True, 
            batch_first=True, dropout=0.5
        )

        # Add the cross-modal excitement layer
        if config['acoustic']['excite']:
            self.acoustic_excit = nn.Embedding(config['semantic']['embedding_size']+1, config['acoustic']['hidden_dim'])
        else:
            self.acoustic_excit = None
        if config['semantic']['excite']:
            self.semantic_excit = nn.Linear(config['acoustic']['hidden_dim'], config['semantic']['hidden_dim'])
        else:
            self.semantic_excit = None
        
        self.loss_name = config['loss']['name']
        self.classifier = nn.Linear(
            2*config['fusion']['hidden_dim'], 
            config['classifier']['class_num']+1*int(self.loss_name=='CTC')
        )
        
        
    def forward(
        self, 
        acoustic_input, 
        acoustic_length, 
        semantic_input, 
        semantic_length, 
        align_input,):
        # first perform the encode for the first-step semantic partterns
        semantic_embed = self.semantic_embed(semantic_input) # [B,T,C]
        semantic_embed = self.semantic_linear(semantic_embed)
        # first perform the encode for the first-step acoustic partterns
        acoustic_embed = self.acoustic_cnn1(acoustic_input.permute(0,2,1))
        acoustic_align = self.acoustic_mean1(align_input[:,None,:,:])
        
        acoustic_embed = self.acoustic_cnn2(acoustic_embed)
        acoustic_align = self.acoustic_mean2(acoustic_align)
        
        acoustic_embed = self.acoustic_cnn3(acoustic_embed) # [B,C,A]
        acoustic_align = self.acoustic_mean3(acoustic_align) # [B,1,T,A]

        acoustic_embed = acoustic_embed.permute(0,2,1)[:,None,:,:].repeat(1,semantic_embed.size(1),1,1) # [B,T,A,C]
        acoustic_align = (acoustic_align.squeeze(1)[:,:,:,None] > 0).float() # [B,T,A,1]
        # Think about the new way of calculate the mean
        acoustic_mean = torch.sum(acoustic_embed*acoustic_align, dim=2) / (torch.sum(acoustic_align, dim=2) + 1e-6) # [B,T,C]
        # Think about the new way of calculate the max
        acoustic_max, _ = torch.max(acoustic_embed*acoustic_align-1e6*(1.0-acoustic_align), dim=2)
        # concat it with both embed
        acoustic_embed = torch.cat([acoustic_mean,acoustic_max], dim=-1)

        # then we use the cross modal excitement information
        if self.semantic_excit is not None:
            semantic_excit = self.semantic_excit(acoustic_embed)
            semantic_embed = semantic_embed + semantic_excit
        if self.acoustic_excit is not None:
            acoustic_excit = self.acoustic_excit(semantic_input)
            acoustic_embed = acoustic_embed + acoustic_excit

        fuse_embed = torch.cat([semantic_embed,acoustic_embed],dim=2)
        # Then we use the fuse lstm to encode the multimodal information
        fuse_pack = nn.utils.rnn.pack_padded_sequence(
            fuse_embed, semantic_length.cpu(), batch_first=True, enforce_sorted=False
        )
        fuse_embed, _ = self.fuse_lstm(fuse_pack)
        fuse_embed, _ = nn.utils.rnn.pad_packed_sequence(
            fuse_embed, batch_first=True
        )
        # Here we get the final results, we use the max pooling to generate the results
        fuse_mask = torch.arange(
            semantic_input.size(1))[None,:].repeat(semantic_input.size(0),1
        ).to(semantic_input.device)
        fuse_mask = (fuse_mask < semantic_length[:,None].repeat(1,semantic_input.size(1))).float()

        if self.loss_name == 'BCE':
            fuse_embed = fuse_embed - (1 - fuse_mask[:,:,None]) * 1e6
            fuse_embed = torch.max(fuse_embed, dim=1)[0]
            logits = self.classifier(fuse_embed)
        
        elif self.loss_name == 'CTC':
            logits = self.classifier(fuse_embed) # [B,T,Dim]
            logits = F.log_softmax(logits)
            logits = logits * fuse_mask[:,:,None]

        else:
            raise ValueError('Loss type not supported!')

        return logits, fuse_embed