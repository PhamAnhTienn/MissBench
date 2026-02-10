python train.py \
  --base_model=ada2i \
  --project_name=ada2i --data_dir=./data \
  --dataset=iemocap --batch_first \
  --seed=51 --modal_MR 0.4 0.8 0.9 \
  --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
  --hidden_dim=300 --alpha=0.037 --beta=0.012 --tensor_rank=11 --mmt_nlayers=2 \
  --mmcosine --modulation --normalize \
  --MEI --MLI \
  --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"

python train.py \
  --base_model=ada2i \
  --project_name=ada2i --data_dir=./data \
  --dataset=mosi --batch_first --regression \
  --seed=51 --modal_MR 0.4 0.8 0.9 \
  --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
  --hidden_dim=300 --alpha=0.037 --beta=0.012 --tensor_rank=11 --mmt_nlayers=2 \
  --mmcosine --modulation --normalize \
  --MEI --MLI \
  --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
  --base_model=ada2i \
  --project_name=ada2i --data_dir=./data \
  --dataset=mosei --batch_first --regression \
  --seed=51 --modal_MR 0.4 0.8 0.9 \
  --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
  --hidden_dim=300 --alpha=0.037 --beta=0.012 --tensor_rank=11 --mmt_nlayers=2 \
  --mmcosine --modulation --normalize \
  --MEI --MLI \
  --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"


python train.py \
  --base_model=ada2i \
  --project_name=ada2i --data_dir=./data \
  --dataset=sims --batch_first --regression \
  --seed=51 --modal_MR 0.4 0.8 0.9 \
  --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
  --hidden_dim=300 --alpha=0.037 --beta=0.012 --tensor_rank=11 --mmt_nlayers=2 \
  --mmcosine --modulation --normalize \
  --MEI --MLI \
  --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"