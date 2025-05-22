export CUDA_VISIBLE_DEVICES=$1

python train.py  --out_path ./checkpoint_multi_inb_webqa/ \
--train_path ../data/WebQA/train.json \
--valid_path ../data/WebQA/dev.json \
--doc_path ../data/WebQA/all_docs.json \
--cap_path ../data/WebQA/all_imgs.json \
--img_feat_path ../data/WebQA/imgs.tsv \
--t5_model_name ../llm_models/t5_ance \
--clip_model_name ../llm_models/openaiclip-vit-base-patch32 \
--pretrained_model_path ../pretrain/checkpoint_pretrain/model.best.pt \
--img_linelist_path ../data/WebQA/imgs.lineidx.new \
--text_len 128 \
--lambda1 $2 \
--num_train_epochs 40 \
--freeze_vision_model