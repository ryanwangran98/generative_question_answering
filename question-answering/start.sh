export XDG_CACHE_HOME=./
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2


# /opt/conda/bin/python -m torch.distributed.run  --nproc_per_node 4 run_seq2seq_qa.py \
#   --model_name_or_path /home/wangran108/code/model_file/randengt5 \
#   --train_file /home/wangran108/code/machine_quality_check/sample7.json\
#   --preprocessing_num_workers 12\
#   --context_column context \
#   --question_column question \
#   --answer_column answer \
#   --do_train \
#   --per_device_train_batch_size 4 \
#   --learning_rate 1e-5 \
#   --num_train_epochs 1 \
#   --max_seq_length 384 \
#   --doc_stride 128 \
#   --output_dir ./output_dir\
#   --overwrite_output_dir \
#   --logging_steps 50 \
#   --logging_nan_inf_filter False\
#   --log_level info \
#   --log_level_replica info \
#   --local_rank 0 \
#   --fp16 > log.log 2>&1 &


accelerate launch /home/wangran108/question-answering/run_seq2seq_qa_no_trainer.py \
    --model_name_or_path /home/wangran108/code/model_file/randengt5 \
    --train_file /home/wangran108/code/machine_quality_check/sample7.json\
    --validation_file  /home/wangran108/code/machine_quality_check/sample2.json\
    --max_source_length 384 \
    --max_target_length 30\
    --text_column context\
    --summary_column answer\
    --preprocessing_num_workers 12\
    --per_device_train_batch_size 4\
    --num_train_epochs 1\
    --num_beams 1\
    --source_prefix "" \
    --with_tracking \
    --ignore_pad_token_for_loss True \
    --output_dir ~/tmp/tst-summarization > log.log 2>&1 &


    # --learning_rate 1e-5\ 初次实验未使用scale learning rate策略，正常运行，没有报错，但是使用了scale_lr之后，报错
    #    --per_device_train_batch_size 8\  batchsize 也影响loss scale
