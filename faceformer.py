import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import math
import time
from wav2vec import Wav2Vec2Model
import types

# Temporal Bias, inspired by ALiBi: https://github.com/ofirpress/attention_with_linear_biases
def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask

# Alignment Bias
def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

# Periodic Positional Encoding
class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=25, max_seq_len=600):
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1)
        self.register_buffer('pe', pe)
        self.float_func = nn.quantized.FloatFunctional()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x):
        if not self.pe.is_quantized:
            self.pe = self.quant(self.pe)
        temp = self.pe[:, :x.size(1), :]
        x = self.float_func.add(x, temp)
        return self.dropout(x)

def quantize_decoder_forward(
        self,
        tgt,
        memory,
        tgt_mask = None,
        memory_mask = None,
        tgt_key_padding_mask = None,
        memory_key_padding_mask = None,
        tgt_is_causal = False,
        memory_is_causal = False):
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing tgt_mask. Default: ``False``.
            memory_is_causal: If specified, applies a causal mask as tgt mask.
                Mutually exclusive with providing memory_mask. Default: ``False``.
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
# nn.quantized.FloatFunctional().add(x, self_attention)
# nn.quantized.QFunctional().add(x, self_attention)
        x = tgt
        if self.norm_first:
            x = self.quant_func.add(x, self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.quant_func.add(x, self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.quant_func.add(x, self._ff_block(self.norm3(x)))
        else:
            x = self.norm1(x + self.dequant(self._sa_block(self.quant_sa_block_x(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)))
            x = self.norm2(x + self.dequant(self._mha_block(self.quant_mha_block_x(x), self.quant_mha_block_memory(memory), memory_mask, memory_key_padding_mask, memory_is_causal)))
            x = self.norm3(x + self._ff_block(x))

        return x
class Faceformer(nn.Module):
    def __init__(self, args):
        super(Faceformer, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        self.dataset = args.dataset
        self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # wav2vec 2.0 weights initialization
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = nn.Linear(768, args.feature_dim)
        # motion encoder
        self.vertice_map = nn.Linear(args.vertice_dim, args.feature_dim)
        # periodic positional encoding 
        self.PPE = PeriodicPositionalEncoding(args.feature_dim, period = args.period)
        # temporal bias

        self.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 600, period=args.period)
        decoder_layer = nn.TransformerDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)        
        #TODO quantizise the decoder, super improtant!
        decoder_layer.forward = types.MethodType(quantize_decoder_forward, decoder_layer)
        setattr(decoder_layer, 'quant_func', nn.quantized.FloatFunctional())
        setattr(decoder_layer, 'quant_sa_block_x', torch.ao.quantization.QuantStub())
        setattr(decoder_layer, 'quant_mha_block_x', torch.ao.quantization.QuantStub())
        setattr(decoder_layer, 'quant_mha_block_memory', torch.ao.quantization.QuantStub())
        setattr(decoder_layer, 'dequant', torch.ao.quantization.DeQuantStub())
  
        # self._modules["layers"][0]
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        # motion decoder
        self.vertice_map_r = nn.Linear(args.feature_dim, args.vertice_dim)
        # style embedding
        self.obj_vector = nn.Linear(len(args.train_subjects.split()), args.feature_dim, bias=False)
        self.device = args.device
        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)
        if self.layers_to_quantize:
            self.quant_vertice_out = torch.ao.quantization.QuantStub()
            self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, audio, template, vertice, one_hot, criterion,teacher_forcing=True):
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)#(1, feature_dim)
        frame_num = vertice.shape[1]
        
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        if self.dataset == "BIWI":
            if hidden_states.shape[1]<frame_num*2:
                vertice = vertice[:, :hidden_states.shape[1]//2]
                frame_num = hidden_states.shape[1]//2
        hidden_states = self.audio_feature_map(hidden_states)

        if teacher_forcing:
            vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
            style_emb = vertice_emb  
            vertice_input = torch.cat((template,vertice[:,:-1]), 1) # shift one position
            vertice_input = vertice_input - template
            vertice_input = self.vertice_map(vertice_input)
            vertice_input = vertice_input + style_emb
            vertice_input = self.PPE(vertice_input)
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            vertice_out = self.vertice_map_r(vertice_out)
        else:
            for i in range(frame_num):
                if i==0:
                    vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                    style_emb = vertice_emb
                    vertice_input = self.PPE(style_emb)
                else:
                    vertice_input = self.PPE(vertice_emb)
                
                # Quantisize the vertice input
                # if self.quantize_statically:
                #     vertice_input = self.quant_vertice_out(vertice_input)

                tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
                vertice_out = self.vertice_map_r(vertice_out)
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)

                # Dequantisize the vertice output
                # if self.quantize_statically:
                #     new_output = self.dequant(new_output)
                new_output = new_output + style_emb
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

        vertice_out = vertice_out + template
        loss = criterion(vertice_out, vertice) # (batch, seq_len, V*3)
        loss = torch.mean(loss)
        return loss

    def predict(self, audio, template, one_hot, optimize_last_layer=False):
        template = template.unsqueeze(1) # (1,1, V*3)

        # if self.quantize_statically:
        #     one_hot = self.quant(one_hot)

        obj_embedding = self.obj_vector(one_hot)
        
        # if self.quantize_statically:
        #     audio = self.quant(audio)
    
        hidden_states = self.audio_encoder(audio, self.dataset).last_hidden_state
        all_vertices_out_list = []
        if self.dataset == "BIWI":
            frame_num = hidden_states.shape[1]//2
        elif self.dataset == "vocaset":
            frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)
        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb # The embedding of the speaker
                # The vertice input are already quantized inside the PPE
                vertice_input = self.PPE(style_emb)
            else:
                # Encode the motions vertices with the periodic positional encoder
                # if self.quantize_statically:
                #     vertice_emb = self.quant(vertice_emb)
                vertice_input = self.PPE(vertice_emb)

            # Mask from the paper
            tgt_mask = self.biased_mask[:, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device)
            # Generating the mask of the input vertices for the decoder
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            # One layer of decoder
            vertice_out = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            # Feed forward layer to generate the vertices

            # This is the line that consumes most of the running time
            # The time increases as the input changes
            if optimize_last_layer:
                vertice_out = vertice_out[:,-1,:]
            vertice_out = self.quant_vertice_out(vertice_out)
            vertice_out = self.vertice_map_r(vertice_out)
            
            # Taking into account only the last prediction of the vertices
            if not optimize_last_layer:
                new_output = self.vertice_map(vertice_out[:,-1,:]).unsqueeze(1)
            else:
                new_output = self.vertice_map(vertice_out).unsqueeze(1)
            
            new_output = self.dequant(new_output)
            vertice_out = self.dequant(vertice_out)
                
            new_output = new_output + style_emb

            # If this line is commented the self.vertice_map_r wont be executed longer.
            vertice_emb = torch.cat((vertice_emb, new_output), 1)
            if optimize_last_layer:
                # if self.quantize_statically:
                #     vertice_out = self.dequant(vertice_out)
                all_vertices_out_list.append(vertice_out)

        if optimize_last_layer:
            vertice_out = torch.stack(all_vertices_out_list, dim=1)
        vertice_out = vertice_out + template
        return vertice_out

