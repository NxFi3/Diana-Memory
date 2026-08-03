from collections import deque
from Engine.Generator import Generation
from Utils.logger import get_logger
from uuid import uuid4
import numpy as np
import pickle
import time 
import faiss
import re
import os


logger = get_logger('STM')

def clean_text(text):
    try: 
        logger.info("[CLR] Cleaning Query")
        cleaned = re.sub(r'[^a-zA-Z0-9\s\u0600-\u06FF]', '', text)

        cleaned = re.sub(r'\s+', ' ', cleaned)
    except Exception as e :
        logger.error(f"[CLR] Unexpected Error {e}")
        return text
    return cleaned.strip()



class STM:
    def __init__(self, generator:Generation,STM_path: str = None, STM_Size: int = 100) -> None:
        self.stm_size = STM_Size
        self.Memory = {}
        self.Generator = generator
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.Generator.Encoder_dim))
        self.mode = 'normal'
        self.buffers = deque(maxlen=10)
        if STM_path is not None:
            self.Path = STM_path
        else:
            current_dir = os.getcwd()
            self.Path = os.path.join(current_dir, "STM_memory.pkl")
        
        if os.path.exists(self.Path):
            self.load()
        else:
            self.save()
            logger.info(f"[STM] Created new memory file at {self.Path}")
    

    def search(self, query,search_mode=True):
        try: 
            if self.mode.lower() =='normal':
                mode_str = "Retrieval" if search_mode else "Save"
                logger.info(f'[SEARCH] Searching Memory Layer Mode {mode_str}')
                Cleaned_Query = clean_text(query)
                Query_embedding = self.Generator.Get_Embedding(Cleaned_Query)
                distances, indices = self.index.search(Query_embedding, k=5)

                results = []
                TimeOfUse = time.time()
            
                for i, ids in enumerate(indices[0]):
                    if ids != -1:
                
                        item = self.Memory[ids].copy()
                        item['similarity'] = float(distances[0][i])
                        item['id'] = int(ids)
                        results.append(item)
                        
                        if search_mode:
                            self.Memory[ids]['count'] += 1
                            self.Memory[ids]['last_access'] = TimeOfUse

                results.sort(key=lambda x: x['similarity'], reverse=True)
            else:
                return list(reversed(self.buffers)), [], ('', np.zeros(self.Generator.Encoder_dim, dtype=np.float32).reshape(1, -1))
        except Exception as e:
            logger.error(f"[SEARCH] Unexpected Error {e}")
            return [] , [] , ('',np.zeros(self.Generator.Encoder_dim, dtype=np.float32).reshape(1,-1))
        
        return results, distances , (Cleaned_Query,Query_embedding)

    def add(self, query: str,tool_result:str=''):
        try:
            logger.info("[ADD] Adding New Memory")
            if self.mode.lower() == 'normal':
                    
                results, distances, (cleaned_query, query_embedding) = self.search(query,False)
                
                if len(distances) > 0 and distances[0][0] >= 0.99:
                    logger.info(f"[STM] Duplicate detected (sim={distances[0]:.3f}), not adding")
                    return False
                
                ID = uuid4().int & 0x7FFFFFFFFFFFFFFF
                SetTime = time.time()
                
                self.Memory[ID] = {
                    'value': query,
                    'embedding': query_embedding,
                    'timestamp': SetTime,
                    'count': 1,
                    'last_access': SetTime
                }
                self.index.add_with_ids(query_embedding, np.array([ID], dtype=np.int64))
                
                if len(self.Memory) > self.stm_size * 1.2:
                    self.forgetting()
            else:
                self.buffers.append({'value': query, 'tool_result': tool_result, 'timestamp': time.asctime()})
        except Exception as e:
            logger.error(f"[ADD] Adding Failed because of Error {e}")
            return False
        return True


    def save(self):
        logger.info(f'[SAVE] Saving STM on Disk')
        try:
            with open(self.Path, 'wb') as f:
                pickle.dump(self.Memory, f)
            faiss.write_index(self.index, f"{self.Path}_index.faiss")
        except Exception as e:
            logger.error(f'[SAVE] Save Failed Error  {e}')
            return False
        return True
    

    def load(self):
        logger.info('[LOADER] Loading Memory From Disk')
        try:
            with open(self.Path, 'rb') as f:
                self.Memory = pickle.load(f)
            self.index = faiss.read_index(f"{self.Path}_index.faiss")
            logger.info(f"[LOADER] Loaded {len(self.Memory)} items from {self.Path}")
        except Exception as e:
            logger.error(f"[LOADER] Unexpected Error {e}")
            return False
        return True

    def clear(self):

        self.Memory = {}
        self.save()
        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.Generator.Encoder_dim))
        return True
    
    def forgetting(self, keep_count=None):
        try:
            if keep_count is None:
                keep_count = self.stm_size

            if len(self.Memory) <= keep_count * 1.1:  
                logger.info(f"[FORGETTING] Memory only {len(self.Memory)}/{keep_count}, skipping")
                return
            
            if len(self.Memory) <= keep_count:
                return
            
            items_with_scores = []
            logger.info("[FORGETTING] Computing Ranks...")
            for item_id, item in self.Memory.items():
                frequency = 1 - np.exp(-0.1 * item['count'])
                time_diff = time.time() - item['last_access']
                recency = np.exp(-1e-3 * time_diff)
                score = frequency + (recency * 0.2)
                items_with_scores.append((score, item_id))
            
            logger.info(f'[FORGETTING] Computing What To keep')
            items_with_scores.sort(key=lambda x: x[0], reverse=True)
            

            keep_ids = set()
            for score, item_id in items_with_scores[:keep_count]:
                keep_ids.add(item_id)

            removed_count = 0
            for item_id in list(self.Memory.keys()):
                if item_id not in keep_ids:
                    del self.Memory[item_id]
                    self.index.remove_ids(np.array([item_id], dtype=np.int64))
                    removed_count += 1
            
            self.save()
            logger.info(f"[FORGETTING] Removed {removed_count} items, kept {len(self.Memory)}")
            
        except Exception as e:
            logger.error(f"[FORGETTING] ERROR {e}")
            return False


