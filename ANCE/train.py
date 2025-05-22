import json
from visual import TSVFile
import logging
import sys
import base64
import os
from typing import Optional
import numpy as np
from torch import nn
from torch.nn import LayerNorm
from tqdm import tqdm
import torch
import argparse
import os.path as op
import time
import pickle
import math
import clip
from torch import optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from data import load_caps, load_docs, load_file, load_img_doc, WebQADataset, EdisDataset
from contextlib import suppress
from tensorboardX import SummaryWriter

logger = logging.getLogger()
import random
import torch.nn.functional as F
from utils import load_model,get_img_patch_token_size

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if hasattr(p.grad, 'data'):
            p.grad.data = p.grad.data.float()

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster

def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def eval_loss(model, valid_reader, device):
    model.eval()
    total_loss = 0.0
    total_corr = 0.0
    counter = 0.0
    loss_function = torch.nn.CrossEntropyLoss()
    for step, batch in tqdm(enumerate(valid_reader)):
        with torch.no_grad():
            query_embedding = model(None, batch['queries'], device)
            bsz, emb_dim = query_embedding.size()
            candidate_embeddings = []
            if 'img_inputs' in batch:
                if 'img_inputs' in batch and 'img_caps' in batch:
                    img_embeddings = model(batch['img_inputs'].cuda(), batch['img_caps'], device)
                elif 'img_inputs' in batch and 'img_caps' not in batch:
                    img_embeddings = model(batch['img_inputs'].cuda(), None, device)
                candidate_embeddings.append(img_embeddings)
            if 'txt_inputs' in batch:
                txt_embeddings = model(None, batch['txt_inputs'], device)
                candidate_embeddings.append(txt_embeddings)
            candidate_embeddings = torch.cat(candidate_embeddings, dim=0)

            query_embedding = F.normalize(query_embedding, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)

            pos_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['pos_idx']).cuda())
            neg_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['neg_idx']).cuda())

            pos_embeddings = pos_embeddings.view(bsz, 1, emb_dim)
            neg_embeddings = neg_embeddings.view(bsz, -1, emb_dim)
            pos_neg_embeddings = torch.cat([pos_embeddings, neg_embeddings], 1)
            pos_neg_embeddings = pos_neg_embeddings.view(-1, emb_dim)
            logit_scale = model.logit_scale.exp()
            pos_neg_score = torch.matmul(query_embedding,
                                         pos_neg_embeddings.t()) * logit_scale
            target = torch.arange(query_embedding.size(0), dtype=torch.long).cuda()
            num_sample = 1+args_dev.neg_num
            target = target * num_sample
            loss = loss_function(pos_neg_score, target)

            max_score, max_idxs = torch.max(pos_neg_score, 1)
            correct_predictions_count = (max_idxs == target).sum() / query_embedding.size(0)
            total_corr += correct_predictions_count.item()
            total_loss += loss.item()
            counter += 1
    if counter == 0:
        return 0.0, 0.0
    return total_loss / counter, total_corr / counter

def train(train_reader, valid_reader, model, device, writer: SummaryWriter):
    t_total = len(train_reader) // args.gradient_accumulation_steps * args.num_train_epochs
    exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.},
            {"params": rest_params, "weight_decay": 0.2},
        ],
        lr=args.learning_rate,
        betas=(0.9, 0.98),
        eps=1.0e-6,
    )
    scheduler = cosine_lr(optimizer, args.learning_rate, args.warmup_steps, t_total)
    loss_function = torch.nn.CrossEntropyLoss()
    tag, global_step, global_loss, best_acc = 0, 0, 0.0, 0.0
    model.zero_grad()
    for epoch in range(int(args.num_train_epochs)):
        for step, batch in enumerate(train_reader):
            model.train()
            query_embedding = model(None, batch['queries'], device)
            bsz, emb_dim = query_embedding.size()
            candidate_embeddings = []
            if 'img_inputs' in batch:
                if 'img_inputs' in batch and 'img_caps' in batch:
                    img_embeddings = model(batch['img_inputs'].cuda(), batch['img_caps'], device)
                elif 'img_inputs' in batch and 'img_caps' not in batch:
                    img_embeddings = model(batch['img_inputs'].cuda(), None, device)
                candidate_embeddings.append(img_embeddings)
            if 'txt_inputs' in batch:
                txt_embeddings = model(None, batch['txt_inputs'], device)
                candidate_embeddings.append(txt_embeddings)
            candidate_embeddings = torch.cat(candidate_embeddings, dim=0)

            query_embedding = F.normalize(query_embedding, dim=-1)
            candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)

            pos_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['pos_idx']).cuda())
            neg_embeddings = torch.index_select(candidate_embeddings, 0, torch.LongTensor(batch['neg_idx']).cuda())

            pos_embeddings = pos_embeddings.view(bsz, 1, emb_dim)
            neg_embeddings = neg_embeddings.view(bsz, -1, emb_dim)
            pos_neg_embeddings = torch.cat([pos_embeddings, neg_embeddings], 1)
            pos_neg_embeddings = pos_neg_embeddings.view(-1, emb_dim)
            logit_scale = model.logit_scale.exp()
            pos_neg_score = torch.matmul(query_embedding, pos_neg_embeddings.t())*logit_scale
            target = torch.arange(query_embedding.size(0), dtype=torch.long).cuda()
            num_sample = 1 + args.img_neg_num + args.txt_neg_num
            target = target * num_sample
            loss = loss_function(pos_neg_score, target)

            max_score, max_idxs = torch.max(pos_neg_score, 1)
            correct_predictions_count = (max_idxs == target).sum() / query_embedding.size(0)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            global_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                global_step += 1
                scheduler(global_step)
                convert_models_to_fp32(model)
                optimizer.step()
                model.zero_grad()
                logger.info("Epoch: {}, global_step: {}, lr: {:.6f}, acc: {:.4f} ({:.4f}), ".format(
                    epoch, global_step,
                    optimizer.param_groups[0]["lr"], correct_predictions_count,
                    global_loss / global_step,
                ))
                writer.add_scalar(
                    "training_acc",
                    correct_predictions_count, global_step)
                writer.add_scalar(
                    "training_loss",
                    global_loss / global_step, global_step)
                if global_step % args.eval_steps == 0 and global_step > 0:
                    logger.info('*********Start eval loss**********')
                    dev_loss, dev_acc = eval_loss(model, valid_reader, device)
                    logger.info("Evaluation at global step {}, average dev loss: {:.4f}, average dev acc: {:.4f}".format(
                        global_step, dev_loss, dev_acc))
                    writer.add_scalar(
                        "dev_acc",
                        dev_acc, global_step)
                    writer.add_scalar(
                        "dev_loss",
                        dev_loss, global_step)
                    if best_acc <= dev_acc:
                        best_acc = dev_acc
                        torch.save({'epoch': epoch,
                                    'model': model.state_dict()}, os.path.join(args.out_path, "model.best.pt"))
                        logger.info("Saved best epoch {0}, best acc {1}".format(epoch, best_acc))
                        tag = 0
                    else:
                        tag += 1
                    if tag >= args.early_stop:
                        logger.info('*********early stop**********')
                        return


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser("")

    parser.add_argument("--out_path", type=str, default='./checkpoint/')
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--t5_model_name", type=str, default='OpenMatch/t5-ance')
    parser.add_argument("--clip_model_name",type=str,default='openai/clip-vit-base-patch32')
    parser.add_argument("--pretrained_model_path", type=str)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--cap_path", type=str)
    parser.add_argument('--img_doc_path',type=str)
    parser.add_argument("--img_feat_path", type=str)
    parser.add_argument("--img_linelist_path", type=str)
    parser.add_argument("--img_root_path", type=str,defaul='')
    parser.add_argument("--text_len",type=int,default=128)

    parser.add_argument("--only_txt", action='store_true', default=False)
    parser.add_argument("--only_img", action='store_true', default=False)
    parser.add_argument("--freeze_language_model",action='store_true',default=False)
    parser.add_argument("--freeze_vision_model", action='store_true', default=False)
    parser.add_argument("--freeze_vision_language_model",action='store_true',default=False)

    parser.add_argument("--num_workers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--img_neg_num", type=int, default=0)
    parser.add_argument("--txt_neg_num", type=int, default=0)
    parser.add_argument("--neg_num", type=int, default=0)

    args = parser.parse_args()
    print(args.img_feat_path)
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    handlers = [logging.FileHandler(os.path.join(args.out_path, 'train_log.txt')), logging.StreamHandler()]
    logging.basicConfig(format='[%(asctime)s] %(levelname)s: %(message)s', level=logging.DEBUG,
                        datefmt='%d-%m-%Y %H:%M:%S', handlers=handlers)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logger.info(args)
    set_seed(args)
    tb_writer = SummaryWriter(log_dir=args.out_path)

    if args.only_txt:
        train_data = load_file(args.train_path, img=False)
        valid_data = load_file(args.valid_path, img=False)
    elif args.only_img:
        train_data = load_file(args.train_path, txt=False)
        valid_data = load_file(args.valid_path, txt=False)
    else:
        train_data = load_file(args.train_path)
        valid_data = load_file(args.valid_path)
    
    captions = None
    if args.cap_path:
        captions = load_caps(args.cap_path)
    args_dev = args

    tokenizer, model, image_processor = load_model(args,device)

    if args.freeze_language_model:
        for n,p in model.t5_model.named_parameters():
            p.requires_grad=False
            if 'shared.weight' in n:
                p.requires_grad=True
    if args.freeze_vision_model:
        for n,p in model.clip_model.named_parameters():
            p.requires_grad=False
    if args.freeze_vision_language_model:
        for n,p in model.named_parameters():
            p.requires_grad = False
            if 'projector' in n:
                p.requires_grad=True

    img_patch_token_size = get_img_patch_token_size(args.clip_model_name)

    if 'edis' in args.train_path.lower():
        img_doc=None
        if args.img_doc_path:
            img_doc=load_img_doc(args.img_doc_path)

        train_data = EdisDataset(args, image_processor, tokenizer, train_data, img_doc,captions=captions, shuffle=True,img_special_len=img_patch_token_size)
        train_sampler = RandomSampler(train_data)
        traindata_reader = DataLoader(dataset=train_data, sampler=train_sampler, num_workers=args.num_workers,
                                      batch_size=args.train_batch_size, collate_fn=train_data.Collector, drop_last=True)
        args_dev.neg_num = 1
        valid_data = EdisDataset(args_dev,image_processor, tokenizer, valid_data, img_doc,captions=captions, shuffle=False,img_special_len=img_patch_token_size)
        valid_sampler = SequentialSampler(valid_data)
        validdata_reader = DataLoader(dataset=valid_data, sampler=valid_sampler, num_workers=args.num_workers,
                                      batch_size=args.valid_batch_size, collate_fn=valid_data.Collector, drop_last=False)
    elif 'webqa' in args.train_path.lower():
        docs = load_docs(args.doc_path)
        train_data = WebQADataset(args, image_processor, tokenizer, train_data, docs, captions=captions, shuffle=True,
                                  img_special_len=img_patch_token_size)
        train_sampler = RandomSampler(train_data)
        traindata_reader = DataLoader(dataset=train_data, sampler=train_sampler, num_workers=args.num_workers,
                                      batch_size=args.train_batch_size, collate_fn=train_data.Collector, drop_last=True)
        args_dev.neg_num = 1
        valid_data = WebQADataset(args_dev, image_processor, tokenizer, valid_data, docs, captions=captions, shuffle=False,
                                  img_special_len=img_patch_token_size)
        valid_sampler = SequentialSampler(valid_data)
        validdata_reader = DataLoader(dataset=valid_data, sampler=valid_sampler, num_workers=args.num_workers,
                                      batch_size=args.valid_batch_size, collate_fn=valid_data.Collector, drop_last=False)
    else:
        raise ("The path of data is error!")

    if args.pretrained_model_path != None:
        logger.info('loading checkpoint from {}'.format(args.pretrained_model_path))
        model.load_state_dict(torch.load(args.pretrained_model_path)['model'])
    model.cuda()
    train(traindata_reader, validdata_reader, model, device, tb_writer)
