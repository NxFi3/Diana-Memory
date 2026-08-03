from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class MemoryItem:
    """ item for Memory """
    id: int
    graph: np.ndarray
    mem_type: str
    value: str
    embedding: np.ndarray
    created_at: int        
    last_access: int     
    count: int
    importance: float
    deleted: int           
    raw_score: Optional[float] =None
    normalized_score: Optional[float] =None
    similarity:Optional[float]=None
    rank:Optional[float]=None
    rrf_score:Optional[float] =None
    bm25_rank: Optional[int] = None 
    embedding_rank: Optional[int] = None 
    embedding_distance:Optional[float] = None
