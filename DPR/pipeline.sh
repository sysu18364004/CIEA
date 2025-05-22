bash train_webqa.sh $1 $2
bash gen_embeds_webqa.sh $1
bash get_hn_webqa.sh $1
bash retrieval_webqa.sh $1
cd ../ANCE
bash train_webqa.sh $1
bash gen_embeds_webqa.sh $1
bash retrieval_webqa.sh $1 $2