

from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Extra, Field

from transformers import CLIPVisionModelWithProjection, AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection
from PIL import Image

import machine_learning.utils.constants as CONST;

DEFAULT_MODEL_NAME = CONST.IMAGE_MODEL_NAME

class CLIPEmbeddings(BaseModel, Embeddings):
    """
    HuggingFace sentence_transformers embedding models.
    """

    client: Any  #: :meta private:
    model_name: str = DEFAULT_MODEL_NAME
    """Model name to use."""
    cache_folder: Optional[str] = None
    """Path to store models. 
    Can be also set by SENTENCE_TRANSFORMERS_HOME environment variable."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""
    encode_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass when calling the `encode` method of the model."""
    multi_process: bool = False
    """Run encode() on multiple GPUs."""
    show_progress: bool = False
    """Whether to show a progress bar."""

    def __init__(self, **kwargs: Any):
        """Initialize the sentence_transformer."""
        super().__init__(**kwargs)

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Compute doc embeddings using a HuggingFace transformer model.
        Args:
            texts: The list of texts to embed.
        Returns:
            List of embeddings, one for each text.
        """
        processor =  AutoProcessor.from_pretrained(self.model_name)
        vision_model = CLIPVisionModelWithProjection.from_pretrained(self.model_name)
        
        photo_image_url = texts[0]
        print('Embedding URL: ' + photo_image_url)
        
        image = Image.open(photo_image_url)        
        image_inputs = processor(images=image, return_tensors="pt")              
        embeddings = vision_model(**image_inputs)
                              
        return embeddings.image_embeds.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a HuggingFace transformer model.
        Args:
            text: The text to embed.
        Returns:
            Embeddings for the text.
        """
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        text_model = CLIPTextModelWithProjection.from_pretrained(self.model_name)
        
        inputs = tokenizer([text], padding=True, return_tensors="np")
              
        return text_model(**inputs).text_embeds
