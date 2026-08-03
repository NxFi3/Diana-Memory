from Utils.logger import get_logger
from Memory.MemoryItems import MemoryItem
import numpy as np
import sqlite3
import time
import re
logger = get_logger("DBM")


class DBManager:
    def __init__(self, DB_path: str = '') -> None:
        logger.info("Setting DataBase")
        self.database_path = 'Long_Term_Memory.db'
        if DB_path != '':
            logger.info(f"Using custom DB: {DB_path}")
            self.database_path = DB_path
        else:
            logger.info(f"Using Default DB: {self.database_path}")

    def create_database(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    graph BLOB,
                    mem_type TEXT,
                    value TEXT,
                    embedding BLOB,
                    created_at INTEGER,
                    last_access INTEGER,
                    count INTEGER,
                    importance REAL,
                    deleted INTEGER DEFAULT 0
                )
            """)
            conn.commit()

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts USING fts5(value)
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_id ON items(id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_importance ON items(importance)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_deleted ON items(deleted)") 

            conn.commit()
            logger.info("Database + FTS5 created successfully")

        except Exception as e:
            logger.error(f"DB creation error: {e}")
        finally:
            conn.close()

    def Search(self, query, Limit=50, Embedding_dtype=np.float32, include_deleted=False):

        results_and = self._search_with_operator(query, Limit, 'AND', Embedding_dtype, include_deleted)
        if results_and:
            return results_and
        logger.info(f"No results for AND, trying OR for: {query}")
        results_or = self._search_with_operator(query, Limit, 'OR', Embedding_dtype, include_deleted)
        return results_or

    def _search_with_operator(self, query, Limit=50, operator='AND', Embedding_dtype=np.float32, include_deleted=False):
        query = re.sub(r'[^\w\s]', '', query)
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            if operator == 'AND':
                words = query.split()
                if len(words) == 1:
                    fts_query = f'"{query}"'
                else:
                    fts_query = ' AND '.join([f'"{w}"' for w in words])
            else:
                words = query.split()
                if len(words) == 1:
                    fts_query = f'"{query}"'
                else:
                    fts_query = ' OR '.join([f'"{w}"' for w in words])
            
            if include_deleted:
                deleted_filter = ""
            else:
                deleted_filter = " AND items.deleted = 0"
            
            cursor.execute(f"""
                SELECT 
                    items.id,
                    items.graph,
                    items.mem_type,
                    items.value,
                    items.embedding,
                    items.created_at,
                    items.last_access,
                    items.count,
                    items.importance,
                    items.deleted,
                    bm25(items_fts) as raw_score
                FROM items_fts
                JOIN items ON items.id = items_fts.rowid
                WHERE items_fts MATCH ? {deleted_filter}
                ORDER BY raw_score
                LIMIT ?
            """, (fts_query, Limit))
            
            rows = cursor.fetchall()
            
            if not rows:
                return []
            
            scores = [r[10] for r in rows]  
            min_score = min(scores)
            max_score = max(scores)
            
            results = []
            for r in rows:
                if max_score > min_score:
                    normalized = (r[10] - min_score) / (max_score - min_score)
                else:
                    normalized = 0.5
                
                graph_data = np.frombuffer(r[1], dtype=Embedding_dtype) if r[1] is not None else np.array([])
                embedding_data = np.frombuffer(r[4], dtype=Embedding_dtype) if r[4] is not None else np.array([])
                
            
                item = MemoryItem(id=r[0],graph=graph_data,mem_type=r[2],
                              value=r[3],embedding=embedding_data,
                              created_at=r[5],last_access=r[6],count=r[7],
                              importance=r[8],raw_score=r[10],
                              normalized_score=normalized,deleted=r[9])
                
                results.append(item)
            
            results.sort(key=lambda x: x.normalized_score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"Search with {operator} error: {e}")
            return []
        finally:
            conn.close()
                


    def get_by_id(self, item_id: int, Embedding_dtype=np.float32):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT 
                    items.id,
                    items.graph,
                    items.mem_type,
                    items.value,
                    items.embedding,
                    items.created_at,
                    items.last_access,
                    items.count,
                    items.importance,
                    items.deleted
                FROM items
                WHERE items.id = ?
            """, (item_id,))
            
            row = cursor.fetchone()
            
            if not row:
                logger.warning(f"Item with id {item_id} not found")
                return None
            
            graph_data = np.frombuffer(row[1], dtype=Embedding_dtype) if row[1] is not None else np.array([])
            embedding_data = np.frombuffer(row[4], dtype=Embedding_dtype) if row[4] is not None else np.array([])
            

            result = MemoryItem(id=row[0],graph=graph_data,mem_type=row[2],
                              value=row[3],embedding=embedding_data,
                              created_at=row[5],last_access=row[6],count=row[7],
                              importance=row[8],deleted=row[9])
                
       
            
            logger.info(f"Retrieved item {item_id}")
            return result
            
        except Exception as e:
            logger.error(f"Get by id error: {e}")
            return None
        finally:
            conn.close()

    def add_item(self, value: str, embedding: np.ndarray,
                 mem_type: str = 'TMode', importance: float = 0.3, graph: np.ndarray = np.array([])):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        try:
            now = time.time()
            cursor.execute("""
                INSERT INTO items (graph, mem_type, value, embedding, created_at, last_access, count, importance)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (graph.tobytes(), mem_type, value, embedding.tobytes(), now, now, 1, importance))

            item_id = cursor.lastrowid
            cursor.execute("INSERT INTO items_fts(rowid, value) VALUES (?, ?)", (item_id, value))

            conn.commit()
            logger.info(f"Item added: {item_id}")
            return item_id

        except Exception as e:
            logger.error(f"Add item error: {e}")
            return None
        finally:
            conn.close()

    def update_item(self, item_id, new_value=None, new_graph=None, new_mem_type=None, 
                    new_importance=None, new_embedding=None, increment_count=False, 
                    soft_delete: bool = None): 
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        try:
            now = time.time()
            updates = []
            params = []
            

            if soft_delete is True:
                updates.append('deleted = ?')
                params.append(1)
                logger.debug(f"Soft deleting item {item_id}")
            elif soft_delete is False:
                updates.append('deleted = ?')
                params.append(0)
                logger.debug(f"Restoring item {item_id}")
    
            
            if new_value is not None:
                updates.append("value = ?")
                params.append(new_value)
                cursor.execute("UPDATE items_fts SET value = ? WHERE rowid = ?", (new_value, item_id))

            if new_importance is not None:
                updates.append("importance = ?")
                params.append(new_importance)

            if new_embedding is not None:
                updates.append("embedding = ?")
                params.append(new_embedding.tobytes())
            if new_graph is not None:
                updates.append('graph = ?')
                params.append(new_graph.tobytes())
            if new_mem_type is not None:
                updates.append('mem_type = ?')
                params.append(new_mem_type)

            if updates or increment_count:
                updates.append("last_access = ?")
                params.append(now)
            if increment_count:
                updates.append("count = count + 1")

            if not updates:
                return False

            query = f"UPDATE items SET {', '.join(updates)} WHERE id = ?"
            params.append(item_id)

            cursor.execute(query, params)
            conn.commit()
            logger.info(f"Item updated: {item_id}")
            return True

        except Exception as e:
            logger.error(f"Update error: {e}")
            return False
        finally:
            conn.close()

    def restore_by_id(self, item_id: int):

        return self.update_item(item_id, soft_delete=False)
    def delete_item(self, item_id: int):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM items WHERE id = ?", (item_id,))
            cursor.execute("DELETE FROM items_fts WHERE rowid = ?", (item_id,))
            conn.commit()
            logger.info(f"Deleted item {item_id}")
            return True

        except Exception as e:
            logger.error(f"Delete error: {e}")
            return False
        finally:
            conn.close()
    
    def DataBase_summary(self):
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM items")
            total_items = cursor.fetchone()[0]
            
            cursor.execute("SELECT mem_type, COUNT(*) FROM items GROUP BY mem_type")
            type_counts = dict(cursor.fetchall())
            
            cursor.execute("SELECT AVG(importance) FROM items")
            avg_importance = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT COUNT(*) FROM items WHERE graph IS NOT NULL AND length(graph) > 0")
            items_with_graph = cursor.fetchone()[0]
            
            cursor.execute("SELECT MAX(last_access) FROM items")
            last_update = cursor.fetchone()[0]
            
            summary = {
                'total_items': total_items,
                'by_type': type_counts,
                'avg_importance': round(avg_importance, 3),
                'items_with_graph': items_with_graph,
                'last_update': last_update
            }
            
            logger.info(f"Database Summary: {summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Summary error: {e}")
            return {}
        finally:
            conn.close()