output_model=/root/epfs/Llama-Chinese/train/sft/save_folder
# 需要修改到自己的输入目录
if [ ! -d ${output_model} ];then
    mkdir ${output_model}
fi
export CUDA_HOME=/usr/local/cuda/
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
# export CUDA_VISIBLE_DEVICES=4

cd /root/epfs/Llama-Chinese/train/sft
torchrun --nproc_per_node 4 --max-restarts 1000  finetune_clm_lora.py \
    --model_name_or_path /root/epfs/Atom-7B-Chat \
    --train_files ../../data/train_sft.csv \
    --validation_files  ../../data/dev_sft.csv \
                         ../../data/dev_sft_sharegpt.csv \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir ${output_model} \
    --evaluation_strategy  steps \
    --max_eval_samples 800 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 10 \
    --warmup_steps 400 \
    --load_in_bits 4 \
    --lora_r 8 \
    --lora_alpha 32 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 10 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 2000 \
    --eval_steps 2000 \
    --save_total_limit 200 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to tensorboard \
    --deepspeed ds_config_zero2.json \
    --overwrite_output_dir \
    --ignore_data_skip true \
    --bf16 \
    --gradient_checkpointing \
    --bf16_full_eval \
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log



    # --resume_from_checkpoint ${output_model}/checkpoint-20400 \