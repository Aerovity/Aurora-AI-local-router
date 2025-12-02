"""
V2 Embedder - Nomic embeddings for CACTUS router.

Uses nomic-ai/nomic-embed-text-v1.5 (768 dimensions).
Better semantic understanding, but larger model size.
Supports Matryoshka embeddings for dimension reduction.
"""

import numpy as np
from typing import List, Optional

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class NomicEmbedder:
    """Nomic embedding model for V2 CACTUS router.
    
    Uses nomic-ai/nomic-embed-text-v1.5 (768 dimensions, ~500MB).
    Supports Matryoshka embeddings for flexible dimension reduction.
    
    Features:
    - 768-dimensional embeddings (full)
    - Can truncate to 64/128/256/512 dims (Matryoshka)
    - Better semantic understanding than MiniLM
    - Requires trust_remote_code=True
    """
    
    DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
    DIMENSIONS = 768
    MATRYOSHKA_DIMS = [64, 128, 256, 512, 768]
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        target_dims: int = 768,
        device: Optional[str] = None
    ):
        """Initialize embedder.
        
        Args:
            model_name: Override model name
            target_dims: Target embedding dimensions (for Matryoshka truncation)
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch not installed. "
                "Install with: pip install transformers torch"
            )
            
        self.model_name = model_name or self.DEFAULT_MODEL
        self.target_dims = target_dims
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer = None
        self.model = None
        self._loaded = False
        
    def load(self):
        """Load the embedding model."""
        if not self._loaded:
            print(f"  Loading Nomic embeddings on {self.device}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            self.model.eval()
            
            self._loaded = True
            print(f"  ✅ Nomic loaded ({self.DIMENSIONS} dims, target={self.target_dims})")
            
    def embed(self, text: str, add_prefix: bool = True) -> np.ndarray:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to embed
            add_prefix: Whether to add 'search_query:' prefix
            
        Returns:
            numpy array of shape (target_dims,)
        """
        if not self._loaded:
            self.load()
            
        # Add search prefix for queries
        if add_prefix:
            text = f"search_query: {text}"
            
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / \
                       torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
        embedding = embedding.cpu().numpy().squeeze().astype(np.float32)
        
        # Matryoshka truncation if needed
        if self.target_dims < self.DIMENSIONS:
            embedding = embedding[:self.target_dims]
            
        return embedding
    
    def embed_batch(
        self, 
        texts: List[str], 
        batch_size: int = 32,
        add_prefix: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            add_prefix: Whether to add 'search_query:' prefix
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of shape (n_texts, target_dims)
        """
        if not self._loaded:
            self.load()
            
        embeddings = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            if add_prefix:
                batch = [f"search_query: {t}" for t in batch]
                
            inputs = self.tokenizer(
                batch,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / \
                                  torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
            embeddings.append(batch_embeddings.cpu().numpy())
            
            if show_progress and (i // batch_size + 1) % 10 == 0:
                print(f"    Embedded {min(i+batch_size, len(texts))}/{len(texts)} samples...")
                
        result = np.vstack(embeddings).astype(np.float32)
        
        # Matryoshka truncation if needed
        if self.target_dims < self.DIMENSIONS:
            result = result[:, :self.target_dims]
            
        return result
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def to_mobile_dims(self, dims: int = 128) -> 'NomicEmbedder':
        """Create a new embedder with reduced dimensions for mobile.
        
        Args:
            dims: Target dimensions (must be in MATRYOSHKA_DIMS)
            
        Returns:
            New NomicEmbedder with truncated dimensions
        """
        if dims not in self.MATRYOSHKA_DIMS:
            raise ValueError(f"dims must be one of {self.MATRYOSHKA_DIMS}")
            
        mobile_embedder = NomicEmbedder(
            model_name=self.model_name,
            target_dims=dims,
            device=self.device
        )
        
        # Share the model if already loaded
        if self._loaded:
            mobile_embedder.tokenizer = self.tokenizer
            mobile_embedder.model = self.model
            mobile_embedder._loaded = True
            
        return mobile_embedder
