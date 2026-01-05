
rerank_user_prompt = "Compare two input images and return a similarity score from 0 (completely different) to 1 (identical) based on overall visual appearance, including content, structure, and style. Just return a number!"
rerank_system_prompt = "You are a scoring expert for image-to-image similarity"

def rerank(vlm_client, model_id, input_image, rag_images, k):
    result = []
    for img in rag_images:
        #display(img)
        content = [
            {"role": "system", "content": rerank_system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": rerank_user_prompt},
                {"type": "image_url", "image_url": { "url": input_image }},
                {"type": "image_url", "image_url": { "url": img }}
            ]}
        ]
        #print(content)
        res = prompt_vllm(vlm_client, model_id, content)
        #print(res)
        result.append(res)

    #print(result)
    top_k_indices = np.argsort(result)[-k:][::-1]
    return [rag_images[i] for i in top_k_indices]