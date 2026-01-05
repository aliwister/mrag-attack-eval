export HF_HOME="/data/hub"
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=1
nohup python mia.py --runs 3 --dataset_id pasindu/google_conceptual_captions_20000 --rag_size 2000 --models Qwen/Qwen2.5-VL-7B-Instruct nvidia/Cosmos-Reason1-7B OpenGVLab/InternVL3_5-8B-HF --limit 1000 --exp 1 --rag_first 0 --wrong_rag_first 0 &

export CUDA_VISIBLE_DEVICES=2
nohup python mia.py --runs 3 --dataset_id pasindu/google_conceptual_captions_20000 --rag_size 2000 --models Qwen/Qwen2.5-VL-7B-Instruct nvidia/Cosmos-Reason1-7B OpenGVLab/InternVL3_5-8B-HF --limit 1000 --exp 1 --rag_first 0 --wrong_rag_first 1 &

export CUDA_VISIBLE_DEVICES=3
nohup python mia.py --runs 3 --dataset_id pasindu/google_conceptual_captions_20000 --rag_size 2000 --models Qwen/Qwen2.5-VL-7B-Instruct nvidia/Cosmos-Reason1-7B OpenGVLab/InternVL3_5-8B-HF --limit 1000 --exp 1 --rag_first 1 --wrong_rag_first 0 &