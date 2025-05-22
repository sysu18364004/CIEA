import json
import os
from visual import TSVFile
import torch
import random
import base64
from PIL import Image
import io
import numpy as np
from torch.utils.data import Dataset, DataLoader, SequentialSampler, RandomSampler
import torch
import clip
from PIL import ImageFile
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


### no pure txt document in edis
class EdisDataset(Dataset):
    def __init__(self, args, preprocess_fn, tokenizer, data, captions, shuffle,img_special_len=49):
        self.img_neg_num = args.img_neg_num
        self.txt_neg_num = args.txt_neg_num
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn
        self.tokenizer=tokenizer

        self.img_map = {}
        self.img_tsv = []
        self.captions = captions
        self.img_special_len=img_special_len

        self.text_len = args.text_len
        self.cap_len = img_special_len + 2 + self.text_len


        img_feat_path = args.img_feat_path
        img_linelist_path = args.img_linelist_path
        all_img_num = 0
        with open(img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
        self.img_tsv = TSVFile(img_feat_path, all_img_num)
        
        self.data = data


    def __len__(self):        return len(self.data)


    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess_fn(images=Image.open(io.BytesIO(base64.b64decode(img))), return_tensors="pt")["pixel_values"][0]
        if self.captions != None:
            cap = self.captions[idx]
            pre_token= DEFAULT_IM_START_TOKEN+" "+ DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap=pre_token+" "+"caption: "+ cap
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        queries = []
        img_querys = []
        img_inputs = []
        txt_inputs = []
        cap_inputs = []
        txt_labels = []
        img_labels = []
        img_query_labels = []
        iq = 0
        processed_batch = {}
        for qid, example in enumerate(batch):
            queries.append(example['query'])
            
            if 'pos_img' in example:
                img_inputs.append(example['pos_img']['img'])
                if 'cap' in example['pos_img']:
                    cap_inputs.append(example['pos_img']['cap'])
                img_labels.append(qid)
                img_query_labels.append(iq)
                iq += 1
                img_querys.append(example['img_query'])
            
            if 'neg_imgs' in example:
                for instance in example['neg_imgs']:
                    img_inputs.append(instance['img'])
                    if 'cap' in instance:
                        cap_inputs.append(instance['cap'])
                    img_labels.append(-1)
                    

        processed_batch['queries'] = self.tokenizer(queries, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)
        processed_batch['img_queries'] = self.tokenizer(img_querys, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)
        assert len(txt_inputs) != 0 or len(img_inputs) != 0

        if len(img_inputs) != 0:
            processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
            processed_batch['img_labels'] = img_labels
            processed_batch['img_query_labels'] = img_query_labels
            if len(cap_inputs) != 0:
                assert len(cap_inputs) == len(img_inputs)
                processed_batch['img_caps'] = self.tokenizer(cap_inputs, return_tensors='pt',max_length=self.cap_len,padding='max_length',truncation=True)


        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        query = example['Q']
        instance = {'query': query}
        Q  = example['Q']
        Q_ids = self.tokenizer.encode(Q)
        # print(Q)
        for img_id in example['img_posFacts']:
            caption = self.captions[img_id]
            C_ids = self.tokenizer.encode(caption)
            Q_ids = [x for x in Q_ids if x not in C_ids]
        # print(tokenizer.decode(Q_ids))
        img_related_q = self.tokenizer.decode(Q_ids)
        instance['img_query'] = img_related_q
        if len(example['img_posFacts']) != 0:
            if self.shuffle:
                idx = random.choice(example['img_posFacts'])
            else:
                idx = example['img_posFacts'][0]
            instance["pos_img"] = self.encode_img(idx)
        else:
            raise ('No positive instance!')



        if self.img_neg_num > 0:
            neg_imgs = []
            neg_img_idx = example['img_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_img_idx)
            neg_img_idx = neg_img_idx[:self.img_neg_num]
            for idx in neg_img_idx:
                neg_imgs.append(self.encode_img(idx))
            instance["neg_imgs"] = neg_imgs

        return instance

        
class WebQADataset(Dataset):
    def __init__(self, args, preprocess_fn, tokenizer, data, docs, captions, shuffle,img_special_len=49):
        self.img_neg_num = args.img_neg_num
        self.txt_neg_num = args.txt_neg_num
        self.shuffle = shuffle
        self.preprocess_fn = preprocess_fn
        self.tokenizer=tokenizer

        self.img_map = {}
        self.img_tsv = []
        self.docs = docs
        self.captions = captions
        self.img_special_len=img_special_len

        self.text_len = args.text_len
        self.cap_len = img_special_len + 2 + self.text_len


        img_feat_path = args.img_feat_path
        img_linelist_path = args.img_linelist_path
        all_img_num = 0
        with open(img_linelist_path) as fin:
            for line in fin:
                tokens = line.strip().split('\t')
                all_img_num += 1
                self.img_map[tokens[0]] = int(tokens[1])
        self.img_tsv = TSVFile(img_feat_path, all_img_num)
        self.data = data


    def __len__(self):
        return len(self.data)


    def encode_img(self, idx):
        offset = self.img_map[idx]
        img = self.img_tsv[offset][1]
        img = self.preprocess_fn(images=Image.open(io.BytesIO(base64.b64decode(img))), return_tensors="pt")["pixel_values"][0]
        if self.captions != None:
            cap = self.captions[idx]
            pre_token= DEFAULT_IM_START_TOKEN+" "+ DEFAULT_IMAGE_PATCH_TOKEN * self.img_special_len + DEFAULT_IM_END_TOKEN
            cap=pre_token+" "+"caption: "+ cap
            return {'img': img, 'cap':cap}
        return {'img': img}


    def Collector(self, batch):
        queries = []
        img_querys = []
        img_inputs = []
        txt_inputs = []
        cap_inputs = []
        txt_labels = []
        img_labels = []
        img_query_labels = []
        iq = 0
        processed_batch = {}
        for qid, example in enumerate(batch):
            queries.append(example['query'])
            
            if 'pos_img' in example:
                img_inputs.append(example['pos_img']['img'])
                if 'cap' in example['pos_img']:
                    cap_inputs.append(example['pos_img']['cap'])
                img_labels.append(qid)
                img_query_labels.append(iq)
                iq += 1
                img_querys.append(example['img_query'])
            if 'pos_txt' in example:
                txt_inputs.append(example['pos_txt'])
                txt_labels.append(qid)
            if 'neg_imgs' in example:
                for instance in example['neg_imgs']:
                    img_inputs.append(instance['img'])
                    if 'cap' in instance:
                        cap_inputs.append(instance['cap'])
                    img_labels.append(-1)
                    
            if 'neg_txts' in example:
                for instance in example['neg_txts']:
                    txt_inputs.append(instance)
                    txt_labels.append(-1)

        processed_batch['queries'] = self.tokenizer(queries, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)
        processed_batch['img_queries'] = self.tokenizer(img_querys, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)
        assert len(txt_inputs) != 0 or len(img_inputs) != 0

        if len(img_inputs) != 0:
            processed_batch['img_inputs'] = torch.stack(img_inputs, dim=0)
            processed_batch['img_labels'] = img_labels
            processed_batch['img_query_labels'] = img_query_labels
            if len(cap_inputs) != 0:
                assert len(cap_inputs) == len(img_inputs)
                processed_batch['img_caps'] = self.tokenizer(cap_inputs, return_tensors='pt',max_length=self.cap_len,padding='max_length',truncation=True)

        if len(txt_inputs) != 0:
            processed_batch['txt_inputs'] = self.tokenizer(txt_inputs, return_tensors='pt',max_length=self.text_len,padding='max_length',truncation=True)
            processed_batch['txt_labels'] = txt_labels

        return processed_batch

    def __getitem__(self, index):
        example = self.data[index]
        query = example['Q']
        instance = {'query': query}
        Q  = example['Q']
        Q_ids = self.tokenizer.encode(Q)
        # print(Q)
        for img_id in example['img_posFacts']:
            caption = self.captions[img_id]
            C_ids = self.tokenizer.encode(caption)
            Q_ids = [x for x in Q_ids if x not in C_ids]
        # print(tokenizer.decode(Q_ids))
        img_related_q = self.tokenizer.decode(Q_ids)
        instance['img_query'] = img_related_q
        if len(example['img_posFacts']) != 0:
            if self.shuffle:
                idx = random.choice(example['img_posFacts'])
            else:
                idx = example['img_posFacts'][0]
            instance["pos_img"] = self.encode_img(idx)
        elif len(example['txt_posFacts']) != 0:
            if self.shuffle:
                idx = random.choice(example['txt_posFacts'])
            else:
                idx = example['txt_posFacts'][0]
            instance["pos_txt"] = self.docs[idx]
        else:
            raise ('No positive instance!')



        if self.img_neg_num > 0:
            neg_imgs = []
            neg_img_idx = example['img_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_img_idx)
            neg_img_idx = neg_img_idx[:self.img_neg_num]
            for idx in neg_img_idx:
                neg_imgs.append(self.encode_img(idx))
            instance["neg_imgs"] = neg_imgs

        if self.txt_neg_num > 0:
            neg_txts = []
            neg_txt_idx = example['txt_negFacts']
            if self.shuffle:
                np.random.shuffle(neg_txt_idx)
            neg_txt_idx = neg_txt_idx[:self.txt_neg_num]
            for idx in neg_txt_idx:
                neg_txts.append(self.docs[idx])
            instance["neg_txts"] = neg_txts
        return instance



def load_file(path, txt=True, img=True):
    data = []

    with open(path) as fin:
        assert (txt or img)
        for line in fin:
            example = json.loads(line.strip())
            txt_negFacts = example['txt_negFacts']
            np.random.shuffle(txt_negFacts)
            example['txt_negFacts'] = txt_negFacts

            img_negFacts = example['img_negFacts']
            np.random.shuffle(img_negFacts)
            example['img_negFacts'] = img_negFacts

            if txt and len(example['txt_posFacts']) != 0:
                data.append(example)
            if img and len(example['img_posFacts']) != 0:
                data.append(example)

    return data


def load_docs(path):
    data = {}
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            did = str(example['snippet_id'])
            data[did] = example['fact']
    return data

def load_caps(path):
    data = {}
    
    with open(path) as fin:
        for line in fin:
            example = json.loads(line.strip())
            imgid = str(example['image_id'])
            data[imgid] = example['caption']

    return data

