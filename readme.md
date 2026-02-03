# A Comprehensive Evaluation of Membership Inference Attack and Image Caption Extraction Attack on Multimodal RAG

### This repository contains the code to reproduce the results reported in the paper


#### Main Experiment MIA 
```
python mia.py --runs 3 --dataset_id <uclanlp/MRAG-Bench|reach-vb/pokemon-blip-captions|pasindu/google_conceptual_captions_20000|eltorio/ROCOv2-radiology> --exp 1 --rag_size <rag_size> --limit <limit>
```
For `google_conceptual_captions_20000` and `eltorio/ROCOv2-radiology`, `rag_size` to `2000` and `limit` to `1000`. 
For `reach-vb/pokemon-blip-captions` and `uclanlp/MRAG-Bench`, set `rag_size`to `-1` and `limit` to `-1`. 

#### Main Experiment MIA (Transformations)
```
python mia.py --runs 3 --dataset_id <uclanlp/MRAG-Bench|reach-vb/pokemon-blip-captions|pasindu/google_conceptual_captions_20000|eltorio/ROCOv2-radiology> --exp 3 --rag_size <rag_size> --limit <limit>
```
For `google_conceptual_captions_20000` and `eltorio/ROCOv2-radiology`, `rag_size` to `2000` and `limit` to `1000`. 
For `reach-vb/pokemon-blip-captions` and `uclanlp/MRAG-Bench`, set `rag_size`to `-1` and `limit` to `-1`. 

#### RAG-First/RAG-Last Ablation MIA
```
python mia.py --runs 3 --dataset_id pasindu/google_conceptual_captions_20000 --rag_size 2000 --limit 1000 --exp 5
```

#### Other Retrievers 
```
python mia.py --runs 4 --dataset_id reach-vb/pokemon-blip-captions --exp 4
```

#### Main Experiment ICR 
```
python icr.py --runs 3 --models <Qwen/Qwen2.5-VL-7B-Instruct|nvidia/Cosmos-Reason1-7B|OpenGVLab/InternVL3_5-8B-HF> --exp 1 
```
Will run all datasets by default, however can specify `--datasets` to run specific datasets

#### Main Experiment ICR (Transformations)
```
python icr.py --runs 3 --models <Qwen/Qwen2.5-VL-7B-Instruct|nvidia/Cosmos-Reason1-7B|OpenGVLab/InternVL3_5-8B-HF> --exp 3 
```
Will run all datasets by default, however can specify `--datasets` to run specific datasets

#### Experiment ICR Ablation vary N vary k
```
python icr.py --runs 3 --models <Qwen/Qwen2.5-VL-7B-Instruct|nvidia/Cosmos-Reason1-7B|OpenGVLab/InternVL3_5-8B-HF> --exp 4 
```