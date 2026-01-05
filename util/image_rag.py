def search_images(image_store, query_embedding, index, k=5):
    # Search in FAISS index
    distances, indices = index.search(query_embedding, k)
    
    # Get image IDs from indices
    image_ids = list(image_store.image_map.keys())
    results = [image_ids[idx] for idx in indices[0]]
    
    return results, distances[0]