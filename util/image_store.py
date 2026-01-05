import json
import os
from util.image import resize_base64_image


class ImageStore:
    def __init__(self, file_name):
        self.directory = '/home/ali.lawati/mllm-rag/image_store'
        self.file_name = file_name
        self.image_map = {}
        self._load_store()
    
    def _load_store(self):
        """Load existing data from files"""
        if os.path.exists(os.path.join(self.directory, f"{self.file_name}.json")):
            with open(os.path.join(self.directory, f"{self.file_name}.json"), "r") as f:
                self.image_map = json.load(f)
    
    def _save_store(self):
        """Save data to files"""
        with open(os.path.join(self.directory, f"{self.file_name}.json"), "w") as f:
            json.dump(self.image_map, f)
    
    def add_images(self, image_ids, images):
        """Store images with their IDs"""
        for img_id, img in zip(image_ids, images):
            self.image_map[img_id] = resize_base64_image(img)
        self._save_store()
    
    def get_image(self, image_id):
        """Retrieve image by ID"""
        return self.image_map.get(image_id)