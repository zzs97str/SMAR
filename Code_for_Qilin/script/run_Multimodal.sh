#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

cd /PATH/Code_for_Qilin

pip install torch_optimizer
pip install --upgrade transformers==4.49.0
pip install jieba
pip install --upgrade accelerate==0.26.0
pip install torchvision==0.17.0


CONFIG_FILE=config/search_multimodal_rank_config.yaml

VALID_CUDA_DEVICES="0,1"
echo "VALID_CUDA_DEVICES:$VALID_CUDA_DEVICES"
export CUDA_VISIBLE_DEVICES=${VALID_CUDA_DEVICES}
unset LD_LIBRARY_PATH # 取消cudnn的环境变量，用 pytorch 自带的

NODE_RANK=${NODE_RANK:-0}  # Default to 0 if not provided
num_machines=${num_machines:-1}  # Default to 1 if not provided

output_dir=output/multimodal

echo "output_dir: ${output_dir}"
log_dir=${output_dir}/log
mkdir -p ${output_dir}
mkdir -p ${log_dir}
sh scripts/set_gpu_num.sh ${CONFIG_FILE} ${num_machines}

TIME_STAMP=$(date +'%Y-%m-%d-%H-%M-%S')


# Determine training mode based on whether num_machines is specified
if [[ ${num_machines} -eq 1 ]]; then
    echo "Starting single-node training"
    
    # Single machine mode uses dynamic port allocation
    port=29581
    # Function to check if port is occupied
    is_port_in_use() {
        lsof -i:"$1" > /dev/null 2>&1
        # 返回上一条命令 (lsof) 的退出状态码
        return $?
    }

    # Loop to check if port is occupied
    while is_port_in_use $port; do
        echo "Port $port is in use. Trying next port..."
        port=$((port + 1))
    done
    echo "Using available port: $port"

    # # Single machine training command
    echo "注意python -m accelerate.commands.launch"
    env CUDA_HOME=$CUDA_HOME LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH \
    python -m accelerate.commands.launch \
        --multi_gpu \
        --main_process_port=$port \
        --config_file config/default_config.yaml \
        src/trainer.py ${CONFIG_FILE} ${TIME_STAMP} \
        | tee ${log_dir}/train.log
        # 将训练脚本的输出同时显示在终端并写入日志文件。

else
    # Multi-machine training setup
    MASTER_IP="127.0.0.1"
    MASTER_PORT=8177
    NODE_RANK=${NODE_RANK:-0}

    echo "Starting multi-node training:"
    echo "Master IP: ${MASTER_IP}"
    echo "Master Port: ${MASTER_PORT}"
    echo "Node Rank: ${NODE_RANK}"
    echo "World Size: ${num_machines}"

    # Multi-machine training command
    accelerate launch \
        --multi_gpu \
        --num_machines=${num_machines} \
        --machine_rank=${NODE_RANK} \
        --main_process_ip="${MASTER_IP}" \
        --main_process_port=${MASTER_PORT} \
        --config_file config/default_config.yaml \
        src/trainer.py ${CONFIG_FILE} ${TIME_STAMP} ${NODE_RANK} ${num_machines} \
        | tee ${log_dir}/train_node${NODE_RANK}.log
fi

echo "=================done train=================="
