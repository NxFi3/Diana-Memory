from Memory.MemoryItems import MemoryItem
from Memory.DatabaseManager import DBManager
from Engine.Generator import Generation
from Engine.prompts import Duplicate_prompt
from Utils.LLM_handler import parse_duplicate_response
from Utils.logger import get_logger
import numpy as np
import time 
import faiss
import re
from typing import List

logger = get_logger('RTR') #RETRIEVAL


def Compute_items_similarity(query_embedding: np.ndarray, DB_results: List[MemoryItem], 
                             Generator_DIM: int, k: int = 10):

    top_similarity = []
    similarity_score = []
    indices = np.array([[]], dtype=np.int64)
    
    try:
        if not DB_results:
            logger.debug("[SIM] No DB results to compare")
            return top_similarity, similarity_score, indices
        
        valid_embeddings = []
        valid_results = []
        for r in DB_results:
            if r.embedding is not None and len(r.embedding) > 0:
                valid_embeddings.append(r.embedding)
                valid_results.append(r)
        
        if not valid_embeddings:
            logger.warning("[SIM] No valid embeddings in DB_results")
            return top_similarity, similarity_score, indices
        
        db_results_embedding = np.array(valid_embeddings).astype(np.float32)
        
        if db_results_embedding.shape[1] != Generator_DIM:
            logger.error(f"[SIM] Dimension mismatch: {db_results_embedding.shape[1]} vs {Generator_DIM}")
            return top_similarity, similarity_score, indices
        

        
        index = faiss.IndexFlatIP(Generator_DIM)
        index.add(db_results_embedding)
        
        
        all_distance, indices = index.search(query_embedding, min(k, len(valid_results)))
        
        for i, ids in enumerate(indices[0]):
            if ids != -1 and ids < len(valid_results):
                item = valid_results[ids]
                # Create a copy with similarity
                item_copy = MemoryItem(
                    id=item.id,
                    graph=item.graph,
                    mem_type=item.mem_type,
                    value=item.value,
                    embedding=item.embedding,
                    created_at=item.created_at,
                    last_access=item.last_access,
                    count=item.count,
                    importance=item.importance,
                    deleted=item.deleted,
                    similarity=float(all_distance[0][i])
                )
                top_similarity.append(item_copy)
        
        if top_similarity:
            similarity_score = [item.similarity for item in top_similarity]
        
        logger.debug(f"[SIM] Found {len(top_similarity)} similar items")
        
    except Exception as e:
        logger.error(f"[SIM] Error in similarity computation: {e}")
        return [], [], np.array([[]], dtype=np.int64)
    
    return top_similarity, similarity_score, indices


def Compute_Rank(DB_search_results: List[MemoryItem],
                 frequency_decay: float = 0.1, 
                 recency_decay: float = 1e-5,
                 alpha: float = 0.4, 
                 beta: float = 0.3, 
                 gamma: float = 0.4) -> np.ndarray:

    try:
        if not DB_search_results:
            logger.debug("[RANK] Empty results")
            return np.array([])

        I_i = np.array([r.importance if r.importance else 0.5 for r in DB_search_results])
        L_i = np.array([r.last_access if r.last_access else time.time() for r in DB_search_results])
        C_i = np.array([r.count if r.count else 1 for r in DB_search_results])

        frequency = 1 - np.exp(-frequency_decay * C_i)
        current_time = time.time()
        time_diff = current_time - L_i
        recency = np.exp(-recency_decay * time_diff)
        
        rank = (alpha * I_i) + (beta * recency) + (gamma * frequency)
        
        return rank
        
    except Exception as e:
        logger.error(f"[RANK] Error in rank computation: {e}")
        return np.array([])

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def mmr_select(items: List[MemoryItem], k: int = 10, lambda_: float = 0.7) -> List[MemoryItem]:
    if not items:
        return []

 
    valid_items = []
    for item in items:
        if item is not None and item.rank is not None and item.embedding is not None:
            valid_items.append(item)
    
    
    if not valid_items:
        return items[:min(k, len(items))]
    

    ranks = []
    for item in valid_items:
        if item.rank is not None:
            ranks.append(item.rank)
    
    if not ranks:
        return valid_items[:min(k, len(valid_items))]
    
    min_rank = min(ranks)
    max_rank = max(ranks)
    range_rank = max_rank - min_rank if max_rank > min_rank else 1.0

    rank_norm_dict = {}
    for item in valid_items:
        if item.rank is not None:
            rank_norm_dict[item.id] = (item.rank - min_rank) / range_rank
        else:
            rank_norm_dict[item.id] = 0.0

    selected = []
    candidates = valid_items.copy()
    

    if candidates:
        first = max(candidates, key=lambda x: rank_norm_dict.get(x.id, 0.0))
        selected.append(first)
        candidates.remove(first)
    
    selected_embeddings = []
    for item in selected:
        if item.embedding is not None:
            selected_embeddings.append(item.embedding)

    while len(selected) < k and candidates:
        best_item = None
        best_score = -1e9

        for c in candidates:
            if c.embedding is None:
                continue
                
     
            max_sim = 0.0
            if selected_embeddings:
                for s_emb in selected_embeddings:
                    if s_emb is not None:
                        sim = cosine_sim(c.embedding, s_emb)
                        if sim > max_sim:
                            max_sim = sim
            

            mmr_score = lambda_ * rank_norm_dict.get(c.id, 0.0) - (1 - lambda_) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_item = c

        if best_item:
            selected.append(best_item)
            if best_item.embedding is not None:
                selected_embeddings.append(best_item.embedding)
            candidates.remove(best_item)

    return selected


def Compute_RRF(db_result: List[MemoryItem], 
               embedding_query: np.ndarray, 
               Encoder_dim: int = 768, 
               k_rrf: int = 60, 
               limit: int = 10) -> List[MemoryItem]:

    # Build BM25 list
    bm25_list = []
    for r in db_result:
        bm25_list.append({
            'id': r.id,
            'value': r.value,
            'graph': r.graph,
            'bm25_score': r.normalized_score if r.normalized_score is not None else 0.5,
            'last_access': r.last_access,
            'count': r.count,
            'importance': r.importance,
            'created_at': r.created_at
        })
    
    if bm25_list:
        embeddings = np.array([r.embedding for r in db_result]).astype(np.float32)
        

        
        index = faiss.IndexFlatIP(Encoder_dim)
        index.add(embeddings)
        D, I = index.search(embedding_query, k=len(bm25_list))
        
        embedding_ranked = [bm25_list[idx] for idx in I[0]]
        embedding_distances = D[0]
    else:
        embedding_ranked = []
        embedding_distances = []
    
    # RRF fusion
    rrf_scores = {}
    for rank, item in enumerate(bm25_list, start=1):
        rrf_scores[item['id']] = {
            **item,
            'rrf_score': 1 / (k_rrf + rank),
            'bm25_rank': rank,
            'embedding_rank': None,
            'embedding_distance': None
        }
    
    for rank, (item, dist) in enumerate(zip(embedding_ranked, embedding_distances), start=1):
        if item['id'] in rrf_scores:
            rrf_scores[item['id']]['rrf_score'] += 1 / (k_rrf + rank)
            rrf_scores[item['id']]['embedding_rank'] = rank
            rrf_scores[item['id']]['embedding_distance'] = float(dist)
    
    # Convert to MemoryItem list
    final_items = []
    for item_data in sorted(rrf_scores.values(), key=lambda x: x['rrf_score'], reverse=True)[:limit]:

        original = next((r for r in db_result if r.id == item_data['id']), None)
        if original:
            memory_item = MemoryItem(
                id=original.id,
                graph=original.graph,
                mem_type=original.mem_type,
                value=original.value,
                embedding=original.embedding,
                created_at=original.created_at,
                last_access=original.last_access,
                count=original.count,
                importance=original.importance,
                deleted=original.deleted,
                raw_score=original.raw_score,
                normalized_score=original.normalized_score,
                rrf_score=item_data['rrf_score'],
                bm25_rank=item_data['bm25_rank'],
                embedding_rank=item_data['embedding_rank'],
                embedding_distance=item_data['embedding_distance']
            )
            final_items.append(memory_item)
    
    return final_items

class Retrieval:
    def __init__(self, DataBase: DBManager, Generator: Generation) -> None:
        self.Generator = Generator
        self.DataBase = DataBase
        self.graph = None
    def LTM_Search(self, query: str, Mode: str = 'normal') -> List[MemoryItem]:
        try:
            query = re.sub(r'[^\w\s]', '', query)
            
            if Mode.lower() == 'save':
                DB_results = self.DataBase.Search(query, Limit=100)
                query_embedding = self.Generator.Get_Embedding(f'query: {query}')
                return self._EfficientRetRieval(DB_results, query_embedding, k=50)
            elif Mode.lower() == 'graph':
                DB_results = self.DataBase.Search(query, Limit=40)
                query_embedding = self.Generator.Get_Embedding(f'query: {query}')
                return self._EfficientRetRieval(DB_results, query_embedding, k=3)
            elif Mode.lower() == 'normal':
                results = self._NormalRetRieval(query,10)
                return results
            else:
                logger.warning(f'Mode "{Mode}" not created yet')
                return []
                
        except Exception as e:
            logger.error(f"[RTR] LTM_Search error: {e}")
            return []

    def _EfficientRetRieval(self, DB_results: List[MemoryItem], 
                            query_embedding: np.ndarray, 
                            k: int) -> List[MemoryItem]:
    
        Top_similarity, distances, indices = Compute_items_similarity(
            query_embedding, DB_results, self.Generator.Encoder_dim, k=k
        )
        if not Top_similarity or not distances:
            return []
        valid_indices = [idx for idx in indices[0] if idx != -1]
        matched_results = [DB_results[idx] for idx in valid_indices]
        rank_scores = Compute_Rank(matched_results)
        for ids, item in enumerate(Top_similarity):
            item.rank = float(rank_scores[ids]) if ids < len(rank_scores) else 0.0
        top_results = sorted(Top_similarity, key=lambda x: x.rank, reverse=True)[:k]
        return top_results
    
    def _NormalRetRieval(self, query: str, k: int = 10) -> List[MemoryItem]:
        try:
            query_cleaned = re.sub(r'[^\w\s]', '', query)
            query_embedding = self.Generator.Get_Embedding(f'query: {query_cleaned}')
            DBResults = self.DataBase.Search(query, Limit=100)
            Encoder_dim = self.Generator.Encoder_dim
            
            # RRF
            RRFResults = Compute_RRF(
                db_result=DBResults,
                embedding_query=query_embedding,
                Encoder_dim=Encoder_dim,
                k_rrf=60,
                limit=60
            )
            
            # MMR
            MMRResult = mmr_select(items=RRFResults, k=20, lambda_=0.8)
            
            # Graph expansion (neighbors)

            if self.graph is None:
                from Memory.Graph_Manager import GraphManager
                self.graph = GraphManager(self.Generator, self.DataBase)
            graphExpand = self.graph.get_neighbors(MMRResult)
            
            # DeDup
            combined = MMRResult + graphExpand
            unique_items = {item.id: item for item in combined}.values()
            Full = list(unique_items)
            
            # Compute Rank
            rank_scores = Compute_Rank(Full)  # np.ndarray
            

            for ids, item in enumerate(Full):
                item.rank = float(rank_scores[ids]) if ids < len(rank_scores) else 0.0
            
            # sort rank
            sorted_items = sorted(Full, key=lambda x: x.rank, reverse=True)
            top_values = [item.value for item in sorted_items[:50]]
            
            # 8. Reranking CrossEncoder
            rerank_scores = self.Generator.RankOutPut(query, top_values)

            for item, score in zip(sorted_items[:50], rerank_scores):
                item.rank = float(score)
            
            final_results = sorted(sorted_items[:50], key=lambda x: x.rank, reverse=True)
            
            return final_results[:k]

        except Exception as e:
            logger.error(f'[RTR Normal] Unexpected Error: {e}')
            return []