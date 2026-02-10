python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=iemocap \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=mosi --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=mosei --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=sims --regression \
    --seed=51 --modal_MR 0.4 0.8 0.9 \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
