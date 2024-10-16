import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from deepface import DeepFace
import cv2
from typing import Dict, Tuple, Union
import logging
from pathlib import Path
import json

class AgeDataset(Dataset):
    """Enhanced dataset class that handles both face crops and full images"""
    
    def __init__(
        self,
        csv_file: str,
        root_dir: str,
        transform=None,
        cache_dir: str = None,
        min_face_size: int = 20,
        use_cache: bool = True
    ):
        """
        Args:
            csv_file (str): Path to the CSV file with image filenames and ages
            root_dir (str): Directory with all the images
            transform: Albumentations transforms to be applied
            cache_dir (str): Directory to cache face detections
            min_face_size (int): Minimum face size to detect
            use_cache (bool): Whether to cache face detections
        """
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.min_face_size = min_face_size
        self.use_cache = use_cache
        
        # Setup caching
        if cache_dir and use_cache:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "face_detections.json"
            self.face_cache = self._load_cache()
        else:
            self.face_cache = {}
            
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate the dataset
        self._validate_dataset()
        
    def _validate_dataset(self):
        """Validate all images in the dataset"""
        valid_indices = []
        for idx in range(len(self.data)):
            img_path = self._get_image_path(idx)
            if os.path.exists(img_path):
                valid_indices.append(idx)
            else:
                self.logger.warning(f"Image not found: {img_path}")
        
        # Update dataframe to only include valid images
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        
    def _load_cache(self) -> Dict:
        """Load face detection cache from disk"""
        if self.cache_file.exists():
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Save face detection cache to disk"""
        if self.use_cache:
            with open(self.cache_file, 'w') as f:
                json.dump(self.face_cache, f)
                
    def _get_image_path(self, idx: int) -> str:
        """Get the full path to an image"""
        img_name = self.data.iloc[idx, 0]
        return os.path.join(self.root_dir, img_name)
    
    def _detect_face(self, image: np.ndarray, img_path: str) -> Union[Dict, None]:
        """Detect face from image and return detection info"""
        try:
            # Use DeepFace for face detection
            face_obj = DeepFace.extract_faces(
                img_path=image if isinstance(image, str) else image,
                detector_backend='retinaface',
                enforce_detection=True,
                align=True
            )
            
            if face_obj and len(face_obj) > 0:
                # Get the face with highest confidence
                best_face = max(face_obj, key=lambda x: x['confidence'])
                facial_area = best_face['facial_area']
                
                return {
                    'x': facial_area['x'],
                    'y': facial_area['y'],
                    'w': facial_area['w'],
                    'h': facial_area['h']
                }
            
        except Exception as e:
            self.logger.warning(f"Face detection failed for {img_path}: {str(e)}")
        
        return None
    
    def _process_image(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Process an image and return both face crop and full image"""
        img_path = self._get_image_path(idx)
        
        # Load full image
        full_image = cv2.imread(img_path)
        full_image = cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB)
        
        # Check cache for face detection
        if self.use_cache and img_path in self.face_cache:
            face_coords = self.face_cache[img_path]
            if face_coords:
                x, y, w, h = face_coords
                face_crop = full_image[y:y+h, x:x+w]
            else:
                face_crop = None
        else:
            # Detect face
            face_detection = self._detect_face(full_image, img_path)
            
            if face_detection:
                x, y, w, h = (
                    face_detection['x'],
                    face_detection['y'],
                    face_detection['w'],
                    face_detection['h']
                )
                face_crop = full_image[y:y+h, x:x+w]
                
                # Cache the detection
                if self.use_cache:
                    self.face_cache[img_path] = [int(x), int(y), int(w), int(h)]
            else:
                face_crop = None
                if self.use_cache:
                    self.face_cache[img_path] = None
            
            # Periodically save cache
            if self.use_cache and len(self.face_cache) % 100 == 0:
                self._save_cache()
        
        # If face detection failed, use center crop
        if face_crop is None:
            h, w = full_image.shape[:2]
            min_dim = min(h, w)
            start_h = (h - min_dim) // 2
            start_w = (w - min_dim) // 2
            face_crop = full_image[start_h:start_h+min_dim, start_w:start_w+min_dim]
        
        print(f"{img_path} image is processed.")
        return face_crop, full_image
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        # Get the image filename and age from the CSV
        age = float(self.data.iloc[idx, 1])
        
        # Get face crop and full image
        face_crop, full_image = self._process_image(idx)
        
        # Apply transformations if provided
        if self.transform:
            face_crop = self.transform(image=face_crop)["image"]
            full_image = self.transform(image=full_image)["image"]
        
        return face_crop, full_image, age
    
    def __del__(self):
        """Save cache when the dataset is destroyed"""
        if hasattr(self, 'use_cache') and self.use_cache:
            self._save_cache()