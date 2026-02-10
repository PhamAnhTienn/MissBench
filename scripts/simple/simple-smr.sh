for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=iemocap \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done

for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=mosi --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=mosei --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=simple \
    --project_name=simple --data_dir=./data \
    --dataset=sims --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --imputation=zero \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done