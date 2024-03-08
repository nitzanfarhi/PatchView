# Commit Message command (This is fine) 

runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan -- python msd.py --epochs 10 --batch_size 16 --source_model Message --message_model_type roberta --learning_rate 1e-5 --recreate_cache

# Events Command 

runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan -- python msd.py --activation=tanh --balance_factor=0.5 --batch_size=512 --dropout=0.3 --epochs=600 --event_l1=83 --event_l2=41 --event_l3=83 --event_l4=80 --event_window_size=41 --folds=10 --learning_rate=0.0001 --run_fold=7 --source_model=Events


# Code Command 

runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan -- python msd.py --epochs 100 --learning_rate 1e-5 --dropout 0.8  --recreate-cache --folds 10  --source_model Code  --model_type roberta


# all command 
runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan -- python msd.py --epochs 10 --eval_batch_size 16 --train_batch_size 16 -lr 1e-5 --dropout 0.7  --recreate_cache --code_merge_file --source_model Multi


# Interactive hypertune
runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nmcnf -- wandb agent nitzanfarhi/multi5/tsvjpd6z


runai submit --pvc=storage:/storage --preemptible --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nmcnf -- wandb agent nitzanfarhi/multi5/tsvjpd6z


python msd.py --source_model Code --epochs 10 -cet sum --code_merge_file --cache_dir /storage/nitzan/cache_data/ --batch_size 16
python msd.py --activation=tanh --balance_factor=0.8 --batch_size=256 --dropout=0.3 --epochs=600 --event_l1=99 --event_l2=53 --event_l3=50 --event_l4=101 --event_window_size=10 --folds=10 --learning_rate=0.0001 --run_fold=6 --source_model=Events --cache_dir /storage/nitzan/cache_data/


msd.py --activation=tanh --balance_factor=0.6 --batch_size=128 --dropout=0.3 --epochs=100 --event_l1=101 --event_l2=66 --event_l3=81 --event_l4=65 --event_window_size=10 --folds=10 --learning_rate=0.0001 --run_fold=5 --source_model=Events --cache_dir /storage/nitzan/cache_data/


runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan --service-type portforward --port 7777:22


 python msd.py --activation=relu --balance_factor=0.5 --batch_size=32 --dropout=0.2 --epochs=1 --folds=10 --learning_rate=1e-05 --run_fold=8 --source_model=Message --weight_decay=0.0001 --cache_dir /storage/nitzan/cache_data/


runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan

 python msd.py --activation=tanh --balance_factor=0.6 --batch_size=128 --dropout=0.3 --epochs=600 --event_l1=101 --event_l2=66 --event_l3=81 --event_window_size=44 --folds=10 --learning_rate=0.0001 --run_fold=5 --source_model=Events --cache_dir /storage/nitzan/cache_data/ --recreate_cache



 python msd.py --activation=tanh --balance_factor=0.6 --batch_size=128 --dropout=0.3 --epochs=600 --event_l1=101 --event_l2=66 --event_l3=81 --event_window_size=10 --folds=10 --learning_rate=0.0001 --run_fold=5 --source_model=Events --cache_dir /storage/nitzan/cache_data/ --recreate_cache



## SSH
### Patchview
runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan --git-sync source=https://github.com/nitzanfarhi/MultiSourceDetection,branch=main,username=nitzanfarhi,password=REMOVED,target=/app_home --service-type portforward --port 7777:22 -- /usr/sbin/sshd -D


python msd.py --balance_factor=0.7 --code_merge_file --activation=tanh --batch_size=8 --cache_dir=/storage/nitzan/cache_data/ --code_embedding_type=simple_with_comments --dropout=0.2 --run_fold=0 --source_model=Multi --weight_decay=0.0001 --learning_rate=1e-05 --message_model_name=roberta-base --message_model_type=roberta --message_tokenizer_name=roberta-base --dropout=0.2 --early_stop_threshold=100 --epochs=284 --event_l1=883 --event_l2=100 --event_l3=114 --event_window_size_after=5 --event_window_size_before=15 --events_model_type=conv1d --folds=10 --multi_code_model_artifact=nitzanfarhi/MSD4/Code_model_0.bin:v1 --multi_events_model_artifact=nitzanfarhi/MSD4/Events_model_0.bin:v5 --multi_message_model_artifact=nitzanfarhi/MSD4/Message_model_0.bin:v1 --freeze_submodel_layers --cut_layers




runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/lstme:tagname -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan --git-sync source=https://github.com/nitzanfarhi/WWTP,branch=unknown,username=nitzanfarhi,password=REMOVED,target=/app_home --service-type portforward --port 7777:22 -- /usr/sbin/sshd -D





runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/lstme:tagname -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan --git-sync source=https://github.com/nitzanfarhi/WWTP,branch=feature/cve,username=nitzanfarhi,password=REMOVED,target=/app_home --service-type portforward --port 7777:22 -- /usr/sbin/sshd -D



 runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/lstme:tagname -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan --git-sync source=https://github.com/nitzanfarhi/WWTP,branch=feature/cve,username=nitzanfarhi,password=REMOVED,target=/app_home --service-type portforward --port 7777:22 -- /usr/sbin/sshd -D


# For job nitzan-0
 kubectl port-forward nitzan-1-0-0 7777:22




runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/lstme:tagname -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan --git-sync source=https://github.com/nitzanfarhi/WWTP,branch=feature/cve,username=nitzanfarhi,password=REMOVED,target=/app_home --working-dir /app_home/WWTP --  wandb agent nitzanfarhi/rev-9f22ebd42158ebc8f3081daf6606196c1f0a597f/8ouiqm8y


# PatchView
runai submit --pvc=storage:/storage --interactive --attach -g 1 -i nitzan2611/my_repo -e WANDB_API_KEY=[REDACTED] --job-name-prefix nitzan --git-sync source=https://github.com/nitzanfarhi/MultiSourceDetection,branch=feature/tensorflow_dataset_compare,username=nitzanfarhi,password=REMOVED,target=/app_home --service-type portforward --port 7777:22 -- /usr/sbin/sshd -D

