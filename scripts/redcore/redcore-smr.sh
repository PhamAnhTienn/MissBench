for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=redcore \
    --project_name=redcore --data_dir=./data \
    --dataset=iemocap --dataset_mode=noncon --batch_first \
    --seed=51 --modal_MR ${rate} \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --task_weight=1.0 --mse_weight=4 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=redcore \
    --project_name=redcore --data_dir=./data \
    --dataset=mosi --dataset_mode=noncon --batch_first --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --task_weight=1.0 --mse_weight=4 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=redcore \
    --project_name=redcore --data_dir=./data \
    --dataset=mosei --dataset_mode=noncon --batch_first --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --task_weight=1.0 --mse_weight=4 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=redcore \
    --project_name=redcore --data_dir=./data \
    --dataset=sims --dataset_mode=noncon --batch_first --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=128 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --embd_size_a=128 --embd_size_v=128 --embd_size_l=128 \
    --AE_layers=256,128,64 --n_blocks=5 --cls_layers=128,128 --dropout_rate=0.5 \
    --task_weight=1.0 --mse_weight=4 \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done