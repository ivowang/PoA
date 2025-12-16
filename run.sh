export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=.
# python src/factories/chat_history_item/offline/construct.py && \

python ./src/distributed_deployment_utils/start_server.py --config_path ./configs/assignments/experiments/qwen25_7b_instruct/instance/db_bench/instance/standard.yaml && \

CUDA_VISIBLE_DEVICES=0 python ./src/run_experiment.py --config_path "configs/assignments/experiments/qwen25_7b_instruct/instance/db_bench/instance/standard.yaml"