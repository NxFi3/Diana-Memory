from Engine.Generator import Generation
from Memory.DatabaseManager import DBManager
from Memory.MemoryItems import MemoryItem
from Memory.Retrievals import Compute_items_similarity , Compute_Rank
from Utils.logger import get_logger
from typing import List
import numpy as np 


logger = get_logger("GRH")




class GraphManager:
    def __init__(self,generator:Generation,dbmanager:DBManager) -> None:
     
            self.min_sim_for_graph = 0.78
            self.DataBase = dbmanager
            self.Generator = generator
    
    
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
    
    def _Graph_Search(self, query):

            try:
                    db_results = self.DataBase.Search(query)
                    query_embedding = self.Generator.Get_Embedding(f'query: {query}')
                    DB_results = self._EfficientRetRieval(DB_results=db_results,query_embedding=query_embedding,k=3)
                    
                    graph_results = sorted(DB_results, key=lambda x: x.similarity, reverse=True)
                    
                    logger.info(f"[GRAPH] Found {len(graph_results)} connections for query")
                    return graph_results
                    
            except Exception as e:
                logger.error(f"[GRAPH] Graph_Search error: {e}")
                return []
       
    def BuildGraphForItem(self,query:str,top:int=3):
        logger.info(f"BuildingGraph For {query[:50]}...")
        try:
            results = self._Graph_Search(query)
            tops = results[:top]
            ids = [i.id for i in tops]
            logger.debug(f"Graph result: {len(ids)} IDs found")
            return np.array(ids)
        except Exception as e:
            logger.error(f'[GRAPH] unexpected Error {e}')
            return np.array([])
    def get_neighbors(self, MMRResults: list) -> list:

        results = []
        try:
            logger.info(f'[GRAPH] Getting neighbors for {len(MMRResults)} items')
            
            for item_id in MMRResults:
                item = self.DataBase.get_by_id(item_id.id)
                if item is not None:
    
                    results.append(item)
                else:
                    logger.warning(f"[GRAPH] Item with id {item_id.id} not found")
            
            logger.info(f"[GRAPH] Retrieved {len(results)}/{len(MMRResults)} neighbors")
            return results
        except Exception as e:
            logger.error(f"[GRAPH] Unexpected Error: {e}")
            return results
    