#!/bin/bash

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:$PATH

cd /path_to_CrossRank

pip install torch_optimizer
pip install --upgrade transformers==4.51.3
pip install jieba
pip install --upgrade accelerate==0.26.0
pip install torchvision==0.17.0
pip install --upgrade pyarrow==14.0.2
echo "Start running Python program"

# pip install flash_attn-2.7.4.post1+cu12torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

CONFIG_FILE=CrossRankExperiment/search_CrossRank.yaml

VALID_CUDA_DEVICES="0,1"
echo "VALID_CUDA_DEVICES:$VALID_CUDA_DEVICES"
export CUDA_VISIBLE_DEVICES=${VALID_CUDA_DEVICES}
unset LD_LIBRARY_PATH # Unset cudnn environment variable, use PyTorch built-in version

NODE_RANK=${NODE_RANK:-0}  # Default to 0 if not provided
num_machines=${num_machines:-1}  # Default to 1 if not provided


output_dir=output/CrossRankEXP

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
    port=36581
    # Function to check if port is occupied
    is_port_in_use() {
        lsof -i:"$1" > /dev/null 2>&1
        # Return the exit status code of the previous command (lsof)
        return $?
    }

    # Loop to check if port is occupied
    while is_port_in_use $port; do
        echo "Port $port is in use. Trying next port..."
        port=$((port + 1))
    done
    echo "Using available port: $port"

    # # Single machine training command
    echo "Notice: python -m accelerate.commands.launch"
    env CUDA_HOME=$CUDA_HOME LD_LIBRARY_PATH=$LD_LIBRARY_PATH PATH=$PATH \
    python -m accelerate.commands.launch \
        --multi_gpu \
        --main_process_port=$port \
        --config_file config/default_config.yaml \
        CrossRankExperiment/trainer.py ${CONFIG_FILE} ${TIME_STAMP} \
        | tee ${log_dir}/train.log
        # Display training output in terminal and write to log file simultaneously

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
        CrossRankExperiment/trainer.py ${CONFIG_FILE} ${TIME_STAMP} ${NODE_RANK} ${num_machines} \
        | tee ${log_dir}/train_node${NODE_RANK}.log
fi

echo "=================done train=================="