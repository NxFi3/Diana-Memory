
from sentence_transformers import SentenceTransformer , CrossEncoder
import ollama 
import numpy as np
from Utils.logger import get_logger
import json
import os

logger = get_logger("GEN")

class Generation:
    def __init__(self, config_path: str = ''):
        self.LLM_Name = "llama3.1:8b-instruct-q4_K_M" 
        self.Encoder_Name = "intfloat/multilingual-e5-base" 
        self.Reranker_Name = "BAAI/bge-reranker-v2-m3"
        self.Encoder = None
        self.Encoder_dim = 768 
        self.Reranker = None 
        if config_path:
            self._ReadConfig(config_path)
        else:
            self._load_models()
    
    def _ReadConfig(self, config_path: str):
        try:
            logger.info(f"Reading Config File From {config_path}")
            with open(config_path, 'r', encoding='utf-8') as file:
                config = json.load(file)
            

            LLM_Name = config.get('LLM', self.LLM_Name)
            encoder_name = config.get('encoder', self.Encoder_Name)
            reranker_name = config.get('reranker',self.Reranker_Name)
            success = self._load_models(LLM_Name=LLM_Name, Encoder_Name=encoder_name,reranker_name=reranker_name)
            if not success:
                logger.error("Problem While Loading Models from config, trying defaults")
                self._load_models()  # try defaults
                
        except FileNotFoundError:
            logger.error(f"Config file not found at {config_path}, loading default models")
            self._load_models()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}, loading default models")
            self._load_models()
        except Exception as e:
            logger.error(f"Error while reading config: {e}, loading default models")
            self._load_models()
    
    def _load_models(self, LLM_Name: str = '', Encoder_Name: str = '',reranker_name:str=''):
        try:
            if LLM_Name:
                self.LLM_Name = LLM_Name
            if Encoder_Name:
                self.Encoder_Name = Encoder_Name
            if reranker_name:
                self.Reranker_Name = reranker_name
            logger.info(f"Loading Models: LLM={self.LLM_Name}, Encoder={self.Encoder_Name}, Reranker={self.Reranker_Name}")

            try:
                ollama.show(self.LLM_Name)
            except Exception:
                logger.warning(f"LLM model '{self.LLM_Name}' not found in ollama. Pulling might be needed")

            self.Encoder = SentenceTransformer(self.Encoder_Name)
            self.Encoder_dim = self.Encoder.get_sentence_embedding_dimension()
            self.Reranker = CrossEncoder(self.Reranker_Name)
            
            logger.info(f"Models Loaded Successfully. Encoder dimension: {self.Encoder_dim}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False
    
    def Generate_text(self, 
                      prompt: str,
                      temperature: float = 0.6,
                      do_sample: bool = False,
                      repetition_penalty: float = 1.1,
                      stream: bool = False,max_new_tokens:int=0) -> str:
        """Generate text using LLM"""
        try:
            logger.info(f"[LLM] Generating text")
            if max_new_tokens == 0:
                options = {
                    "temperature": temperature if do_sample else 0.0,
                    "do_sample": do_sample,"stream":stream,
                    "repetition_penalty": repetition_penalty,
                }
            else:
                options = {
                    "temperature": temperature if do_sample else 0.0,
                    "do_sample": do_sample,"stream":stream,
                    "repetition_penalty": repetition_penalty,"num_predict": max_new_tokens
                }
            result = ollama.generate(
                model=self.LLM_Name,
                prompt=prompt,
                
                options=options
            )
            
            return result.get('response', 'NO_RESPONSE')
            
        except Exception as e:
            logger.error(f"[LLM] Generation error: {e}")
            return 'GENERATION ERROR.'
    
    def Get_Embedding(self, query: str, normalize: bool = True) -> np.ndarray:
        """Get embedding vector for text"""
        if self.Encoder is None:
            logger.error("[ENC] Encoder not loaded")
            return np.zeros(self.Encoder_dim, dtype=np.float32).reshape(1,-1)
        
        try:
            logger.debug(f"[ENC] Encoding Query: {query[:50]}...")
            result = self.Encoder.encode(query, normalize_embeddings=normalize)
            return result.reshape(1,-1)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return np.zeros(self.Encoder_dim, dtype=np.float32).reshape(1,-1)
    
    def RankOutPut(self,query,documents):

        try:
            if not documents:
                logger.warning("[RERANKER] No documents to rank")
                return []
            logger.info(f'[RERANKER] Ranking {len(documents)} sentences for query {query[:50]}...')
            scores = self.Reranker.predict([[query, doc] for doc in documents])
            return scores
        
        except Exception as e:
            logger.error(f"[RERANKER] Unexpected Error {e}")
            return []

