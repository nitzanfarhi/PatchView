# MSD 
## Runai interface
runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=XXXX --job-name-prefix nitzan 

## Commit Message Model
python msd.py --epochs 10 --batch_size 16 --source_model Message --message_model_type roberta --learning_rate 1e-5 --recreate_cache

## Events Model 
python msd.py --activation=tanh --balance_factor=0.5 --batch_size=512 --dropout=0.3 --epochs=600 --event_l1=83 --event_l2=41 --event_l3=83 --event_l4=80 --event_window_size=41 --folds=10 --learning_rate=0.0001 --run_fold=7 --source_model=Events

## Code Model 
python msd.py --epochs 100 --learning_rate 1e-5 --dropout 0.8  --recreate-cache --folds 10  --source_model Code  --model_type roberta

## Multi Model 
python msd.py --epochs 10 --eval_batch_size 16 --train_batch_size 16 -lr 1e-5 --dropout 0.7  --recreate_cache --code_merge_file --source_model Multi
