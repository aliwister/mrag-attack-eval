import argparse, gc, torch, faiss, random, clip, os, logging, time, hashlib, numpy as np, pandas as pd
from datetime import datetime

from dataclasses import dataclass
from datasets import Dataset, load_dataset
from transformers import AutoModel, AutoTokenizer, pipeline
from util.image import *
from util.llms import map_llm_name
from util.metrics import * 

import hashlib
from tqdm.auto import tqdm
# The rest: SigLIP and CLIP imports below
from transformers import SiglipVisionModel, SiglipImageProcessor
from torchvision import transforms
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Multi-GPU configuration
# ---------------------------------------------------------------------------
NUM_GPUS = torch.cuda.device_count()
# Reserve GPU 0 for retrieval / rerank models; VLM pipeline gets all GPUs via
# device_map="auto" (Accelerate will shard across all available devices).
RETRIEVAL_DEVICE = "cuda:0"

# Wrapper classes as before
class DINOv2Wrapper(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
    @torch.no_grad()
    def forward(self, x):
        return self.encode_image(x)

    @torch.no_grad()
    def encode_image(self, x):
        """
        x: (B,C,H,W) tensor
        returns: L2-normalized embedding (B,D)
        """
        feats = self.model(x.to(self.device))
        if feats.ndim == 4:
            feats = feats.mean(dim=[2,3])
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

class SiglipWrapper(torch.nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
    @torch.no_grad()
    def forward(self, x):
        return self.encode_image(x)

    @torch.no_grad()
    def encode_image(self, x):
        outputs = self.model(pixel_values=x.to(self.device))
        feats = outputs.pooler_output
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

def load_retriever(retrieve_model_id, device=RETRIEVAL_DEVICE):
    device = torch.device(device)

    # Common preprocess transform (can be overridden)
    default_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if retrieve_model_id == "vit-b32":
        emb_model, preprocess = clip.load("ViT-B/32", device=device)

    elif retrieve_model_id == "dino-vitb14":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14').to(device)
        model.eval()
        preprocess = default_preprocess
        emb_model = DINOv2Wrapper(model, device)

    elif retrieve_model_id == "siglip-so400m":
        processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        model = SiglipVisionModel.from_pretrained("google/siglip-so400m-patch14-384").to(device)
        model.eval()
        if isinstance(processor.size, dict):
            height = processor.size.get("height", 224)
            width  = processor.size.get("width", 224)
        else:
            height = width = processor.size

        preprocess = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=processor.image_mean,
                                 std=processor.image_std)
        ])
        emb_model = SiglipWrapper(model, device)

    else:
        raise ValueError(f"Unknown retriever {retrieve_model_id}")

    return emb_model, preprocess

torch.set_float32_matmul_precision('high')
device = RETRIEVAL_DEVICE  # default device for retrieval / rerank

class MIADataset(Dataset):
    def __init__(self, prompts, refs, in_rag):
        assert len(prompts) == len(refs), "Prompts and refs must be same length"
        assert len(prompts) == len(in_rag), "Prompts and refs must be same length"
        self.prompts = prompts
        self.refs = refs
        self.in_rag = in_rag
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, idx):
        return self.prompts[idx]


class DB:
    def __init__(self, embeds, imgs, ids):
        assert len(embeds) == len(imgs), "Prompts and refs must be same length"
        assert len(imgs) == len(ids), "Prompts and refs must be same length"
        self.embeds = embeds
        self.imgs = imgs
        self.ids = ids

logging.basicConfig(
    filename="RESULTS-log.log",
    filemode="a",
    level=logging.CRITICAL,
    format="%(asctime)s - %(message)s"
)

def save_to_file(preds, refs, folder, filename):
    df = pd.DataFrame({
        "pred": preds,
        "ref": refs,
    })
    df.to_csv(f"outputs/mia/{datetime.fromtimestamp(datetime.now().timestamp()).strftime('%Y-%m-%d:%H:%M:%S')}_{filename}.csv", index=False, encoding="utf-8")

def build_rag_simple(emb_model, preprocess, dataset, image_col, num_rows, post_process_rag="RESIZE"):
    embeddings, images = [], []

    print(f"in index rag {post_process_rag} {num_rows}")
    if num_rows == -1:
        num_rows = len(dataset)

    for i in range(num_rows):
        row = dataset[i]
        rag_image = row[image_col]
        rag_images_resized = post_process_image(rag_image, post_process_rag)

        img_tensor = preprocess(rag_images_resized).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = emb_model.encode_image(img_tensor).cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

        embeddings.extend(emb)
        images.append(rag_images_resized)
        if (i+1) % 10 == 0:
            print(f"RAG indexed {i} of {num_rows}")

    embeddings = np.array(embeddings).astype("float32")
    unique_embeddings, unique_indices = np.unique(embeddings, axis=0, return_index=True)
    unique_images = [images[i] for i in unique_indices]

    return DB(unique_embeddings, unique_images, [-1*i for i in unique_indices])


def index_faiss(unique_embeddings):
    index = faiss.IndexFlatL2(unique_embeddings.shape[1])
    index.add(unique_embeddings)
    return index

def add_positive_set(db, emb_model, pos_images, pos_image_ids, post_process_rag="RESIZE"):
    images_copy = [row for row in db.imgs]
    embeddings_copy = [row for row in db.embeds]
    ids_copy = [row for row in db.ids]

    pos_images = [post_process_image(img, post_process_rag) for img in pos_images]
    img_tensors = torch.stack([preprocess(img) for img in pos_images]).to(device)
    with torch.no_grad():
        emb = emb_model.encode_image(img_tensors).cpu().numpy()
        emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    emb = np.array(emb).astype("float32")
    embeddings_copy.extend(emb)
    images_copy.extend(pos_images)
    ids_copy.extend(pos_image_ids)

    unique_embeddings, unique_indices = np.unique(embeddings_copy, axis=0, return_index=True)
    unique_images = [images_copy[i] for i in unique_indices]
    unique_ids = [ids_copy[i] for i in unique_indices]

    return unique_embeddings, unique_images, unique_ids

def add_positives(dataset, db, emb_model, num_rows, pos_fraction, post_process):
    pos_image_indices = random.sample(range(num_rows), k=int(num_rows * pos_fraction))
    pos_images = dataset.select(pos_image_indices)["image"]

    emb_db_pos, images_db_pos, ids_db_pos = add_positive_set(db, emb_model, pos_images, pos_image_indices, post_process)
    index_new = index_faiss(emb_db_pos)
    return index_new, images_db_pos, ids_db_pos, pos_image_indices


def prep_data(dataset, num_rows, db, is_rerank, retrieve_count, rerank_count, pos_fraction, preprocess, emb_model, process, is_plot=False):
    index, images_db_pos, ids_db_pos, pos_image_indices = add_positives(dataset, db, emb_model, num_rows, pos_fraction, process)
    print(pos_image_indices)
    prompt_imgs, rag_list = [], []
    if not is_rerank:
        retrieve_count = rerank_count

    for i in range(num_rows):
        row = dataset[i]
        input_image = post_process_image(row['image'], 'RESIZE')
        img_tensor = preprocess(input_image).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = emb_model.encode_image(img_tensor).cpu().numpy()
            emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)

        D, I = index.search(emb, k=retrieve_count)
        encoded_rag_images = [images_db_pos[ii] for ii in I[0]]

        rag_idx = [ids_db_pos[ii] for ii in I[0]]
        encoded_input_img = input_image

        rag_list.append(rag_idx)
        if is_plot:
            plot(input_image, I[0], images_db_pos, ids_db_pos)

        prompt_imgs.append((encoded_input_img, encoded_rag_images))
    labels = [i in pos_image_indices for i in range(num_rows)]
    return prompt_imgs, labels, rag_list


def prep_prompt(input_image, rag_images, is_rag_first=True, reverse=False, is_safety_prompt=True):
    rag_first_prompt = "Task: Compare the last image (query image) with the other images (retrieved examples).\nAre any retrieved examples are visually the same as the query image either in original or modified form (e.g., cropped, resized, altered)? Respond with YES or NO only."
    rag_last_prompt = "Task: Compare the first image (query image) with the other images (retrieved examples).\nAre any retrieved examples are visually the same as the query image either in original or modified form (e.g., cropped, resized, altered)? Respond with YES or NO only."

    user_content = []
    if reverse:
        for img in reversed(rag_images):
            user_content.append({"type": "image_url", "image_url": {"url": img}})
    else:
        for img in rag_images:
            user_content = [{"type": "image_url", "image_url": {"url": img}} for img in rag_images]

    if is_rag_first:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})
        user_content.append({"type": "text", "text": rag_first_prompt})
    else:
        user_content.insert(0, {"type": "image_url", "image_url": {"url": input_image}})
        user_content.append({"type": "text", "text": rag_last_prompt})

    if is_safety_prompt:
        content = [
            {"role": "system", "content": {"type": "text", "text": main_system_prompt}},
            {"role": "user", "content": user_content}
        ]
    else:
        content = [
            {"role": "user", "content": user_content}
        ]
    return content

def prep_prompt_rerank(rerank_model, input_image, rag_images, rag_indexes, is_rag_first=True, wrong_rag_order=False):
    rag_first_prompt = (
        "Instruction: Compare the last image (query image) with the other images (retrieved examples)."
        "Are any retrieved examples are visually the same as the query image either in original or modified form (e.g., cropped, resized, altered)?"
        "Respond with YES or NO only."
    )
    rag_last_prompt = (
        "Instruction: Compare the first image (query image) with the other images (retrieved examples)."
        "Are any retrieved examples are visually the same as the query image either in original or modified form (e.g., cropped, resized, altered)?"
        "Respond with YES or NO only."
    )

    rerank_pairs = [[img, input_image] for img in rag_images]

    scores = rerank_model.compute_score(rerank_pairs, max_length=2048, doc_type="image", query_type="image")
    ranked = sorted(zip(scores, rag_images, rag_indexes), key=lambda x: x[0], reverse=True)
    top5 = ranked[:args.rerank_count]
    scores_top5, rag_images_top5, rag_indexes_top5 = zip(*top5)

    scores_top5 = list(scores_top5)
    rag_images_top5 = list(rag_images_top5)
    user_content = [{"type": "image_url", "image_url": {"url": img}} for img in rag_images_top5]
    if is_rag_first:
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})
        if wrong_rag_order:
            user_content.append({"type": "text", "text": rag_last_prompt})
        else:
            user_content.append({"type": "text", "text": rag_first_prompt})
    else:
        user_content.insert(0, {"type": "image_url", "image_url": {"url": input_image}})
        if wrong_rag_order:
            user_content.append({"type": "text", "text": rag_first_prompt})
        else:
            user_content.append({"type": "text", "text": rag_last_prompt})

    content = [
        {"role": "user", "content": user_content}
    ]
    return content, rag_indexes_top5


def prep_prompt_gpt(input_image, rag_images, rag_first=True, reverse=False):
    main_system_prompt = "You are an expert meta-analyst."

    if not rag_first:
        instruction = (
            "Task: The first image is the query image, followed by the retrieved examples. "
            "Compare the query image with the retrieved examples. "
            "Determine if any retrieved image is visually the same as the query image, either in original or modified form (e.g., cropped, resized, altered). Respond with YES or NO only."
        )
        user_content = [{"type": "image_url", "image_url": {"url": input_image}}]
        user_content.extend([{"type": "image_url", "image_url": {"url": img}} for img in rag_images])
    else:
        instruction = (
            "Task: You are given retrieved examples first, followed by the query image. "
            "Compare the query image with the retrieved examples. "
            "Determine if any retrieved image is visually the same as the query image, "
            "either in original or modified form (e.g., cropped, resized, altered). "
            "Respond with YES or NO only."
        )
        if reverse:
            user_content = [{"type": "image_url", "image_url": {"url": img}} for img in reversed(rag_images)]
        else:
            user_content = [{"type": "image_url", "image_url": {"url": img}} for img in rag_images]
        user_content.append({"type": "image_url", "image_url": {"url": input_image}})

    user_content.append({"type": "text", "text": instruction})

    return [
        {"role": "user", "content": user_content}
    ]


# ---------------------------------------------------------------------------
# Multi-GPU pipeline builder
# ---------------------------------------------------------------------------
def build_pipeline(model_id: str, batch_size: int = 16):
    """
    Build a Hugging Face `pipeline` that automatically shards the VLM across
    all available GPUs using Accelerate's `device_map="auto"`.

    Key improvements over the original single-GPU call:
      • device_map="auto"  – Accelerate partitions layers across GPUs so that
                             models larger than a single GPU's VRAM can run,
                             and smaller models benefit from parallel KV-cache.
      • torch_dtype        – kept as bfloat16 for speed / memory efficiency.
      • max_memory         – reserves ~2 GB on GPU 0 for the retrieval /
                             rerank models that live there; remaining VRAM on
                             all GPUs is available to the VLM.
    """
    n = torch.cuda.device_count()
    if n == 0:
        # CPU fallback
        print("No CUDA devices found – running on CPU (slow).")
        return pipeline(
            "image-text-to-text",
            model=model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    # Reserve ~2 GB on GPU 0 for retrieval / rerank models.
    total_mem = torch.cuda.get_device_properties(0).total_memory
    reserved_mb = 2 * 1024  # 2 GB in MiB
    gpu0_available = max(1, (total_mem // (1024 ** 2)) - reserved_mb)

    max_memory = {i: f"{torch.cuda.get_device_properties(i).total_memory // (1024**2)}MiB"
                  for i in range(n)}
    max_memory[0] = f"{gpu0_available}MiB"
    max_memory["cpu"] = "48GiB"  # allow CPU offload as last resort

    print(f"[build_pipeline] {n} GPU(s) detected. max_memory={max_memory}")

    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        model_kwargs={"max_memory": max_memory},
    )

    # Decoder-only VLMs require left-padding for correct batch generation.
    # Patch after construction to avoid a transformers collate closure bug
    # (NameError: f_padding_value) that occurs when a pre-built tokenizer
    # with padding_side='left' is passed directly to pipeline().
    if hasattr(pipe, "tokenizer") and pipe.tokenizer is not None:
        pipe.tokenizer.padding_side = "left"

    return pipe


def exp_rerank(dataset, db, args, model_id, emb_model, preprocess, rerank_model, process, log=""):
    size = args.limit if args.limit > 0 else len(dataset)
    is_rerank = True

    # ------------------------------------------------------------------
    # Build multi-GPU pipeline (replaces the original single-GPU call)
    # ------------------------------------------------------------------
    pipe = build_pipeline(model_id, batch_size=args.batch_size)

    seeds = [42, 53, 13]

    print(f"Size={size}")
    for r in range(args.runs):
        start = time.time()
        random.seed(seeds[r])
        prompt_imgs, labels, rag_list = prep_data(dataset, size, db, is_rerank, args.retrieve_count, args.rerank_count, .5, preprocess, emb_model, process)

        prompts, indices_top5 = zip(*[
            prep_prompt_rerank(rerank_model, p[0], p[1], r, is_rag_first=args.rag_first, wrong_rag_order=args.wrong_rag_first)
            for p, r in zip(prompt_imgs, rag_list)
        ])

        results = []
        with tqdm(total=len(prompts)) as pbar:
            for out in pipe(prompts, batch_size=args.batch_size, temperature=0.0, do_sample=False):
                # The pipeline may yield either:
                #   • a list of dicts  (batch_size > 1, single-GPU / no device_map)
                #   • a single dict    (device_map="auto" always streams one at a time)
                # Normalise to a list so the extraction below is always correct.
                batch = out if isinstance(out, list) else [out]
                res = [o['generated_text'][-1]['content'] for o in batch]
                results.extend(res)
                pbar.update(len(res))

        elapsed = time.time() - start
        metrics = compute_metrics_from_predictions(results, labels, indices_top5)
        log_line = (
            f"{elapsed},{size},{args.dataset_id},{model_id},{args.rag_first},"
            f"{args.wrong_rag_first},{is_rerank},{log},{args.rag_size},"
            f"{args.retrieve_count},{args.rerank_count},"
            f"{metrics['accuracy']}, {metrics['precision']}, {metrics['recall']}, "
            f"{metrics['f1']}, {metrics['retrieval_accuracy']}, {metrics}"
        )
        print(log_line)
        with open(f"RESULTS-MIA-{args.dataset_id.split('/')[1]}.log", "a") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {log_line}\n")

        dataset_name = args.dataset_id.split('/')[1]
        filename = (
            f"{map_llm_name(model_id)}__{dataset_name}__mia_output_{size}"
            f"_ragf={args.rag_first}_wrag={args.wrong_rag_first}_rr={is_rerank}"
            f"_N={args.rag_size}_n={args.retrieve_count}_k={args.rerank_count}-{r}"
        )
        save_to_file(results, labels, dataset_name, filename)

    # Free GPU memory before loading the next model
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rag_size', type=int, default=-1)
    parser.add_argument('--limit', type=int, default=-1)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for pipeline inference')

    parser.add_argument('--rag_first', type=int, default=1)
    parser.add_argument('--wrong_rag_first', type=int, default=0)
    parser.add_argument('--retrieve_count', type=int, default=20)
    parser.add_argument('--rerank_count', type=int, default=5)
    parser.add_argument('--is_rerank', action='store_false', help='Disable reranking')

    parser.add_argument('--retrieve_model_id', type=str, default="vit-b32")
    parser.add_argument('--dataset_id', type=str, default="reach-vb/pokemon-blip-captions")
    parser.add_argument('--models', nargs="+", type=str, default=[
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "OpenGVLab/InternVL3_5-8B-HF",
        'nvidia/Cosmos-Reason1-7B',
    ])
    parser.add_argument('--exp', type=int, default=1)
    args = parser.parse_args()

    dataset_raw = load_dataset(args.dataset_id)

    seen_hashes = set()
    def keep_first(example):
        img_bytes = example['image'].tobytes()
        img_hash = hashlib.md5(img_bytes).hexdigest()
        if img_hash in seen_hashes:
            return False
        seen_hashes.add(img_hash)
        return True

    label = 'image'
    if args.dataset_id == "uclanlp/MRAG-Bench":
        dataset = dataset_raw['test']
        dataset = dataset.filter(keep_first)
        dataset = dataset.rename_column("answer", "caption")
        dataset_split = dataset.train_test_split(test_size=500, seed=42, shuffle=False)
        train_dataset = dataset_split["train"]
        test_dataset = dataset_split["test"]
    elif args.dataset_id == "pasindu/google_conceptual_captions_20000":
        dataset = dataset_raw['train']
        dataset = dataset.rename_column("image_data", label)
        dataset_split = dataset.train_test_split(test_size=1000, seed=42, shuffle=False)
        train_dataset = dataset_split["train"]
        test_dataset = dataset_split["test"]
    elif args.dataset_id == "eltorio/ROCOv2-radiology":
        train_dataset = dataset_raw['train']
        test_dataset = dataset_raw['test']
    elif args.dataset_id == "reach-vb/pokemon-blip-captions":
        dataset = dataset_raw['train']
        dataset = dataset.rename_column("text", "caption")
        dataset = dataset.filter(keep_first)
        dataset_split = dataset.train_test_split(test_size=400, seed=42, shuffle=False)
        train_dataset = dataset_split["train"]
        test_dataset = dataset_split["test"]

    # ------------------------------------------------------------------
    # Rerank model – kept on GPU 0 alongside retrieval model.
    # ------------------------------------------------------------------
    rerank_model = AutoModel.from_pretrained(
        'jinaai/jina-reranker-m0',
        torch_dtype="auto",
        trust_remote_code=True,
    )
    rerank_model.to(RETRIEVAL_DEVICE)
    rerank_model.eval()

    if args.exp == 1:
        emb_model, preprocess = load_retriever(args.retrieve_model_id, device=RETRIEVAL_DEVICE)
        process = 'RESIZE'
        db = build_rag_simple(emb_model, preprocess, train_dataset, 'image', args.rag_size, process)
        for model_id in args.models:
            exp_rerank(test_dataset, db, args, model_id, emb_model, preprocess, rerank_model, ['RESIZE'], log=process)

    if args.exp == 3:
        emb_model, preprocess = load_retriever(args.retrieve_model_id, device=RETRIEVAL_DEVICE)
        processes = ['CROP', 'MASK', 'BLUR', 'ERASE', 'ROTATE', 'G-NOISE']
        for process in processes:
            db = build_rag_simple(emb_model, preprocess, train_dataset, 'image', args.rag_size, process)
            for model_id in args.models:
                exp_rerank(test_dataset, db, args, model_id, emb_model, preprocess, rerank_model, process, log=process)

    if args.exp == 4:
        processes = ['MASK', 'ROTATE']
        for emb_model_id in ["dino-vitb14", "siglip-so400m"]:
            emb_model, preprocess = load_retriever(emb_model_id, device=RETRIEVAL_DEVICE)
            for process in processes:
                db = build_rag_simple(emb_model, preprocess, train_dataset, 'image', args.rag_size, process)
                for model_id in args.models:
                    exp_rerank(test_dataset, db, args, model_id, emb_model, preprocess, rerank_model, ['RESIZE'], log=f"{process}-{emb_model_id}")

    if args.exp == 5:
        process = 'RESIZE'
        emb_model, preprocess = load_retriever(args.retrieve_model_id, device=RETRIEVAL_DEVICE)
        db = build_rag_simple(emb_model, preprocess, train_dataset, 'image', args.rag_size, process)
        for model_id in args.models:
            args.rag_first = 1; args.wrong_rag_first = 0
            exp_rerank(test_dataset, db, args, model_id, emb_model, preprocess, rerank_model, process, log=process)
            args.rag_first = 0; args.wrong_rag_first = 0
            exp_rerank(test_dataset, db, args, model_id, emb_model, preprocess, rerank_model, process, log=process)
            args.rag_first = 0; args.wrong_rag_first = 1
            exp_rerank(test_dataset, db, args, model_id, emb_model, preprocess, rerank_model, process, log=process)
            args.rag_first = 1; args.wrong_rag_first = 1
            exp_rerank(test_dataset, db, args, model_id, emb_model, preprocess, rerank_model, process, log=process)
