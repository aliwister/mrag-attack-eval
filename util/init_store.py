import os
import faiss
from torch.utils.data import DataLoader, Subset
from sentence_transformers import SentenceTransformer
import uuid
import numpy as np


def init_image_store(dataset, image_store, batch_size=4, collate_fn=None, limit=None, update_indices=None):
    # Initialize image embeddings model
    image_embeddings = SentenceTransformer('clip-ViT-B-32')
    
    # Initialize FAISS index with Inner Product (cosine similarity)
    dimension = 512  # CLIP ViT-B-32 produces 512-dimensional embeddings
    index = faiss.IndexFlatIP(dimension)
    
    # Set to track unique embeddings
    seen_embeddings = set()
    
    def default_collate_fn(batch):
        """Default collate function for DataLoader"""
        images = [item['image'] for item in batch]
        return {
            'images': images
        }
    
    def add_to_store(images):
        """Add images to FAISS index and image store"""
        # Get image embeddings
        image_embeds = image_embeddings.encode(images, convert_to_numpy=True)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(image_embeds)
        
        # Filter out duplicates
        unique_indices = []
        for i, embed in enumerate(image_embeds):
            # Convert embedding to tuple for hashing
            embed_tuple = tuple(embed.round(decimals=3))
            if embed_tuple not in seen_embeddings:
                seen_embeddings.add(embed_tuple)
                unique_indices.append(i)
        
        if not unique_indices:
            return []
            
        # Keep only unique images and their embeddings
        unique_images = [images[i] for i in unique_indices]
        unique_embeds = image_embeds[unique_indices]
        
        # Generate unique IDs for images
        image_ids = [str(uuid.uuid4()) for _ in unique_images]
        
        # Add to FAISS index
        index.add(unique_embeds)
        
        # Store images
        image_store.add_images(image_ids, unique_images)
        
        return image_ids
    
    # Create directory for storage
    os.makedirs("./faiss_db", exist_ok=True)
    
    # Process the dataset in batches
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn if collate_fn is not None else default_collate_fn
    )
    
    i = 1
    for batch in dataloader:
        add_to_store(batch['images'])           
        print(f"Processed batch {i} of {len(dataloader)}")
        i += 1
        if limit and i > limit:
            break
    
    # If update_indices is provided, process only those indices
    if update_indices is not None:
        subset_dataset = Subset(dataset, update_indices)
        update_dataloader = DataLoader(
            subset_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn if collate_fn is not None else default_collate_fn
        )
        
        print("\nProcessing update indices...")
        for batch in update_dataloader:
            add_to_store(batch['images'])  
    
    # Save FAISS index
    faiss.write_index(index, f"./faiss_db/{image_store.file_name}.faiss")
    
    return index, image_embeddings
