CONFIG_FILE=$1
NUM_MACHINES=$2
cat config/default_config.yaml.sample > config/default_config.yaml

# Get CUDA_VISIBLE_DEVICES environment variable
cuda_devices=${CUDA_VISIBLE_DEVICES:-"Not Set"}
echo "cuda_devices: $cuda_devices"

# Calculate the number of GPUs per machine
if [ "$cuda_devices" != "Not Set" ]; then
    num_gpus=$(echo "$cuda_devices" | tr ',' ' ' | wc -w)
else
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
fi

# Calculate the total number of processes (world_size)
WORLD_SIZE=$((num_gpus * NUM_MACHINES))

echo "num_machines: $NUM_MACHINES"
echo "num_machines: $NUM_MACHINES" >> config/default_config.yaml
echo "num_processes: $num_gpus"
echo "num_processes: $num_gpus" >> config/default_config.yaml
echo "WORLD_SIZE: $WORLD_SIZE"

if [[ "$CONFIG_FILE" == *"dcn"* ]]; then
    sed -i 's/mixed_precision:.*$/mixed_precision: "no"/g' config/default_config.yaml
fi