import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel, T5ForConditionalGeneration, T5Tokenizer, CLIPVisionModel
import numpy as np

class WeightedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8,  dropout=0.1):
        super(WeightedMultiheadAttention, self).__init__()
        self.n_heads = num_heads
        self.d_model = embed_dim
        self.d_k = embed_dim // num_heads
        self.h = num_heads
        self.dropout = nn.Dropout(dropout)
        self.scale = (self.d_k ** -0.5)

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, weights=None):
        b, t, _ = q.size()

        q = self.q_linear(q).view(b, t, self.h, self.d_k)
        k = self.k_linear(k).view(b, t, self.h, self.d_k)
        v = self.v_linear(v).view(b, t, self.h, self.d_k)

        q = q.permute(0, 2, 1, 3)  # (b, h, t, d_k)
        k = k.permute(0, 2, 3, 1)  # (b, h, t, d_k)
        v = v.permute(0, 2, 1, 3)  # (b, h, t, d_k)

        energy = torch.matmul(q, k) * self.scale
        if weights is not None:
            weights = weights.unsqueeze(1).unsqueeze(2)
            energy = energy * (weights ** 2)
            energy = energy.masked_fill(weights < 1e-5, float('-inf'))
        # print(energy)
        attention = F.softmax(energy, dim=-1)
        # print(attention)
        # print(self.scale)
        
        assert not torch.isinf(attention).any()
        assert not torch.isnan(attention).any()
        # print(attention)
        output = torch.matmul(self.dropout(attention), v)

        output = output.permute(0, 2, 1, 3)  # (b, t, h, d_k)
        output = output.reshape(b, t, -1)  # (b, t, d_model)
        output = self.out_linear(output)

        return output
class CustomTransformerLayer(nn.Module):
    def __init__(self, emb_size, nhead = 8, dim_feedforward = 2048, dropout=0.05):
        super(CustomTransformerLayer, self).__init__()
        self.emb_size = emb_size
        self.attention = WeightedMultiheadAttention(embed_dim=emb_size, num_heads=nhead, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, dim_feedforward),
            nn.LeakyReLU(),
            nn.Linear(dim_feedforward, emb_size)
        )
        self.layer_norm = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, weights):
        # Apply weights to embeddings    
        # Self-attention layer
        attn_output = self.attention(x, x, x, weights)
        x = x + self.dropout(attn_output)
        x = self.layer_norm(x)
        
        # Feed-forward layer
        x = x + self.feed_forward(x)

        return x

class MultiModal(nn.Module):
    def __init__(self, clip_model_name, t5_model, t5_tokenizer, lm_type, select_layer=-1):
        super(MultiModal, self).__init__()
        # vision model
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
        # language model
        clip = CLIPModel.from_pretrained(clip_model_name)
        self.logit_scale = clip.logit_scale
        self.t5_model = t5_model
        self.t5_tokenizer = t5_tokenizer
        self.select_layer = select_layer
        # projector
        self.image_dims = self.clip_model.config.hidden_size
        if 'gpt' in lm_type:
            self.text_dims = self.t5_model.config.n_embd
        else:
            self.text_dims = self.t5_model.config.hidden_size
 
        self.projector = nn.Linear(self.image_dims, self.text_dims)
        self.vocab_size = self.t5_model.config.vocab_size

        self.information_extract = CustomTransformerLayer(self.text_dims)
        self.lm_type = lm_type


    def encode_images_only(self, images):

        image_embeddings = self.clip_model(images, output_hidden_states=True)
        # get the patch image representations (except the cls token)
        image_embeddings = image_embeddings.hidden_states[self.select_layer][:, 1:, :]

        
        image_embeddings = self.projector(image_embeddings)
        bs, l, h = image_embeddings.shape
        return image_embeddings

    def get_text_inputs_embeds(self, text_inputs, device):
        input_ids = text_inputs["input_ids"].to(device)
        input_embeddings = self.t5_model.get_input_embeddings()
        text_inputs_embeds = input_embeddings(input_ids)
        return text_inputs_embeds

    def get_images_with_caption_inputs_embeds(self, images, img_caps, attn_mask, device, only_img):
    
        image_embeddings = self.encode_images_only(images)
        img_special_token_size=image_embeddings.shape[1]

        
        img_caps_input_embs = self.get_text_inputs_embeds(img_caps, device)
        pure_cap_inptut_embs = img_caps_input_embs[:, img_special_token_size+1:, :]
        img_caps_atten = attn_mask
        ### mask should also considered
        pure_attention_mask = img_caps_atten[:, img_special_token_size + 1: ]
        
        cos_matrix = ((F.cosine_similarity(pure_cap_inptut_embs.unsqueeze(2), image_embeddings.unsqueeze(1), dim=-1) + 1.00001)  / 2)
        ### some words are 'pad' tokens, should be mask and the weight should also not be use 
        cos_matrix = (cos_matrix ) * pure_attention_mask.unsqueeze(-1) 
        # print(cos_matrix)
        values , indices = cos_matrix.max(axis=-2)
        image_embeddings = self.information_extract(image_embeddings, 1 - values)

        ### only image is be used so we should only return the image embedding but not concat
        if only_img:
            return image_embeddings, torch.ones((image_embeddings.shape[:2]), device=img_caps_input_embs.device)

        merge_input_embs = torch.cat((img_caps_input_embs[:, 0:1, :], image_embeddings, pure_cap_inptut_embs),
                                     dim=1)
        merge_mask = torch.cat((img_caps_atten[:, 0:1], torch.ones((image_embeddings.shape[:2]), device=img_caps_input_embs.device), pure_attention_mask),
                                     dim=1)
        return merge_input_embs, merge_mask

    def get_rep(self, inputs_embeds, input, device, attention_mask=None):
        if input == None:
            attention_mask = None
        elif attention_mask==None:
            attention_mask = input['attention_mask'].to(device)
        decoder_input_ids = torch.zeros((inputs_embeds.shape[0], 1), dtype=torch.long)
        decoder_input_ids = decoder_input_ids.to(device)
        if 'gpt' in self.lm_type or 'bert' in self.lm_type:
            outputs = self.t5_model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                return_dict=True
            )
            hidden = outputs.last_hidden_state
            rep = hidden[:, -1, :]
        else:

            outputs = self.t5_model(
                input_ids=None,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                decoder_input_ids=decoder_input_ids,
                return_dict=True
            )
            hidden = outputs.last_hidden_state
            rep = hidden[:, 0, :]
        return rep

    def forward(self, images=None, text_inputs=None, device=None,only_img = False):
        if images != None and text_inputs != None:
            merge_embs, attn = self.get_images_with_caption_inputs_embeds(images, text_inputs, text_inputs['attention_mask'].to(device), device, only_img )
            merge_imgs_rep = self.get_rep(merge_embs, text_inputs, device, attn)
            return merge_imgs_rep
        elif images != None and text_inputs == None:
            image_embs = self.encode_images_only(images)
            image_rep = self.get_rep(image_embs, text_inputs, device)
            return image_rep
        elif images == None and text_inputs != None:
            text_embs = self.get_text_inputs_embeds(text_inputs, device)
            text_rep = self.get_rep(text_embs, text_inputs, device)
            return text_rep
        else:
            raise ValueError("the input is error! ")
