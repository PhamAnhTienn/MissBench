python train.py \
--project_name=utt_fusion \
--base_model=utt_fusion --data_dir=./data \
--dataset=iemocap --dataset_mode=noncon --batch_first \
--seed=51 --modal_MR 0.0 \
--batch_size=128 --epochs=50 --early_stop=10 --lr=2e-4 --optimizer=adam --lr_policy=plateau \
--embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
--cls_layers=128,128 --dropout_rate=0.5 \
--embd_method_a=maxpool --embd_method_v=maxpool


python train.py \
--project_name=utt_fusion \
--base_model=utt_fusion --data_dir=./data \
--dataset=mosi --dataset_mode=noncon --batch_first --regression \
--seed=51 --modal_MR 0.0 \
--batch_size=128 --epochs=50 --early_stop=10 --lr=2e-4 --optimizer=adam --lr_policy=plateau \
--embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
--cls_layers=128,128 --dropout_rate=0.5 \
--embd_method_a=maxpool --embd_method_v=maxpool


python train.py \
--project_name=utt_fusion \
--base_model=utt_fusion --data_dir=./data \
--dataset=mosei --dataset_mode=noncon --batch_first --regression \
--seed=51 --modal_MR 0.0 \
--batch_size=128 --epochs=50 --early_stop=10 --lr=2e-4 --optimizer=adam --lr_policy=plateau \
--embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
--cls_layers=128,128 --dropout_rate=0.5 \
--embd_method_a=maxpool --embd_method_v=maxpool


python train.py \
--project_name=utt_fusion \
--base_model=utt_fusion --data_dir=./data \
--dataset=sims --dataset_mode=noncon --batch_first --regression \
--seed=51 --modal_MR 0.0 \
--batch_size=128 --epochs=50 --early_stop=10 --lr=2e-4 --optimizer=adam --lr_policy=plateau \
--embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
--cls_layers=128,128 --dropout_rate=0.5 \
--embd_method_a=maxpool --embd_method_v=maxpool