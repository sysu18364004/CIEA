export CUDA_VISIBLE_DEVICES=$1

python train.py  --out_path ./checkpoint_multi_hn_webqa/ \
--train_path ../DPR/checkpoint_multi_inb_webqa/train_all.json \
--valid_path ../DPR/checkpoint_multi_inb_webqa/dev_all.json \
--img_linelist_path ../data/WebQA/imgs.lineidx.new \
--img_feat_path ../data/WebQA/imgs.tsv \
--t5_model_name ../llm_models/t5_ance \
--clip_model_name ../llm_models/openaiclip-vit-base-patch32 \
--doc_path ../data/WebQA/all_docs.json \
--cap_path ../data/WebQA/all_imgs.json \
--train_batch_size 64 \
--valid_batch_size 64 \
--text_len 128 \
--pretrained_model_path ../DPR/checkpoint_multi_inb_webqa/model.best.pt \
--gradient_accumulation_steps 1 \
--img_neg_num 1 \
--txt_neg_num 1 \
--num_train_epochs 40 \
--freeze_vision_model



