from Memory.DatabaseManager import DBManager
from Memory.Retrievals import Retrieval
from Engine.Generator import Generation
from Engine.prompts import  Duplicate_prompt
from Utils.LLM_handler import parse_duplicate_response
from Utils.logger import get_logger
import numpy as np
import time 
import faiss
import re

logger = get_logger('LTM')



class LTM:
    def __init__(self,Generator:Generation,DataBaseManager:DBManager,surprise_history:list=[]) -> None:
        self.alpha = 0.4
        self.beta = 0.3
        self.gamma = 0.4
        self.tetha = 1.5
        self.recency_decay = 1e-5
        self.frequency_decay = 0.1
        self.DataBase = DataBaseManager
        self.Generator = Generator
        from Memory.Graph_Manager import GraphManager 
        self.graphManager = GraphManager(generator=self.Generator,dbmanager=self.DataBase)
        self.RetriEval = Retrieval(self.DataBase,self.Generator) 
        self.graphManager.retrieval = self.RetriEval 
        if surprise_history:
            self.surprise_threshold = np.percentile(surprise_history, 80) # NOT USING FOR NOW
        else:
            self.surprise_threshold = 0.12
    

    def Save_LTM(self, query: str, importance: float):
        try:
            if not 0 <= importance <= 1:
                logger.warning(f"[SAVE] Invalid importance: {importance}, using 0.5")
                importance = 0.5
            
            query_embedding = self.Generator.Get_Embedding(f'passage: {query}')
            memory_result = self.RetriEval.LTM_Search(query, Mode='save')
            
            if not memory_result:
                logger.info("[SAVE] empty DB → direct save")
                graph_ids = self.graphManager.BuildGraphForItem(query) 
                self.DataBase.add_item(query, query_embedding, 'TMode', importance,graph=graph_ids)
                return True


            similarities = [m.similarity for m in memory_result if m.similarity is not None]
            max_sim = max(similarities) if similarities else 0
            best_match = memory_result[0] if memory_result else None

            logger.info(f"[SAVE] max similarity = {max_sim:.3f}")

            # STRICT DUPLICATE
            if max_sim > 0.989 and best_match:
                new_importance = (importance + best_match.importance) / 2  
                self.DataBase.update_item(
                    best_match.id,  
                    new_value=query,
                    new_importance=new_importance,
                    increment_count=True
                )
                logger.info(f"[SAVE] merged duplicate (sim={max_sim:.3f}), no LLM")
                return True

            # LLM-based merge
            if max_sim > 0.96:
                top_5_duplicate = memory_result[:5]
                context = "\n".join([f"ID: {r.id}\nVALUE: {r.value}\n" for r in top_5_duplicate])  
                
                is_duplicate = self.Generator.Generate_text(Duplicate_prompt(query, context))
                duplicate = parse_duplicate_response(is_duplicate)
                
                if duplicate:
                    duplicates_embedding = self.Generator.Get_Embedding(duplicate.get('new_value', query))
                    self.DataBase.update_item(
                        duplicate['id'],
                        new_value=duplicate.get('new_value', query),
                        new_embedding=duplicates_embedding,
                        increment_count=True
                    )
                    logger.info("[SAVE] merged duplicate memory (LLM)")
                    return True

            # Surprise-based save
            weighted_sim = sum(similarities) / len(similarities) if similarities else 0
            surprise = importance * (1 - weighted_sim)
            save_score = (0.7 * importance) + (0.3 * surprise)

            logger.info(f"[SAVE] surprise = {surprise:.3f}, save_score = {save_score:.3f}")

            if save_score > self.surprise_threshold:
                graph_ids = self.graphManager.BuildGraphForItem(query) 
                self.DataBase.add_item(query, query_embedding, 'TMode', importance,graph=graph_ids)
                logger.info("[SAVE] stored in memory")
                return True

            logger.info("[SAVE] rejected")
            return False
            
        except Exception as e:
            logger.error(f"[SAVE] Save_LTM error: {e}")
            return False
        
        