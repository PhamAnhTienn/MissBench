python train.py \
    --project_name=mmin \
    --base_model=mmin --data_dir=./data \
    --dataset=iemocap --dataset_mode=noncon --batch_first \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --embd_method_a=maxpool --embd_method_v=maxpool \
    --task_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --project_name=mmin \
    --base_model=mmin --data_dir=./data \
    --dataset=mosi --dataset_mode=noncon --batch_first --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --embd_method_a=maxpool --embd_method_v=maxpool \
    --task_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --project_name=mmin \
    --base_model=mmin --data_dir=./data \
    --dataset=mosei --dataset_mode=noncon --batch_first --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --embd_method_a=maxpool --embd_method_v=maxpool \
    --task_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --project_name=mmin \
    --base_model=mmin --data_dir=./data \
    --dataset=sims --dataset_mode=noncon --batch_first --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --embd_method_a=maxpool --embd_method_v=maxpool \
    --task_weight=1.0 --mse_weight=4.0 --cycle_weight=2.0 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"