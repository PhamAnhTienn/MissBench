python train.py \
    --base_model=sdrgnn \
    --project_name=sdrgnn --data_dir=./data \
    --base_encoder=lstm --bi_enc \
    --dataset=iemocap \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --hidden=200 --windowp=3 --windowf=3 --loss-recon --task_weight=0.5 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --base_model=sdrgnn \
    --project_name=sdrgnn --data_dir=./data \
    --base_encoder=lstm --bi_enc \
    --dataset=mosi --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --hidden=200 --windowp=3 --windowf=3 --loss-recon --task_weight=0.5 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --base_model=sdrgnn \
    --project_name=sdrgnn --data_dir=./data \
    --base_encoder=lstm --bi_enc \
    --dataset=mosei --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --hidden=200 --windowp=3 --windowf=3 --loss-recon --task_weight=0.5 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --base_model=sdrgnn \
    --project_name=sdrgnn --data_dir=./data \
    --base_encoder=lstm --bi_enc \
    --dataset=sims --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --hidden=200 --windowp=3 --windowf=3 --loss-recon --task_weight=0.5 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
