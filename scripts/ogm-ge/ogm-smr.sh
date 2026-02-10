for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=ogm \
    --project_name=ogm --data_dir=./data \
    --base_encoder=lstm \
    --dataset=iemocap --batch_first \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --method=OGM_GE --alpha=0.1 --fusion_method=concat \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=ogm \
    --project_name=ogm --data_dir=./data \
    --base_encoder=lstm \
    --dataset=mosi --batch_first --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --method=OGM_GE --alpha=0.1 --fusion_method=concat \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=ogm \
    --project_name=ogm --data_dir=./data \
    --base_encoder=lstm \
    --dataset=mosei --batch_first --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --method=OGM_GE --alpha=0.1 --fusion_method=concat \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done


for rate in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7
do
  echo "Running with missing_rate=${rate}"

  python train.py \
    --base_model=ogm \
    --project_name=ogm --data_dir=./data \
    --base_encoder=lstm \
    --dataset=sims --batch_first --regression \
    --seed=51 --modal_MR ${rate} \
    --batch_size=16 --epochs=50 --early_stop=10 --optimizer=adam --lr_policy=plateau \
    --method=OGM_GE --alpha=0.1 --fusion_method=concat \
    --MEI --MLI \
    --comet --comet_api="your-comet-api" --comet_workspace="your-comet-workspace"
done