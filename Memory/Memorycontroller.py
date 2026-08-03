from Engine.prompts import Save_Decision_Prompt, summary_prompt, Main_prompt
from Utils.LLM_handler import parse_save_decision
from Utils.logger import get_logger
from Engine.Generator import Generation
from Memory.DatabaseManager import DBManager 
from Memory.LTM_manager import LTM 
from Memory.STM_manager import STM 
import json
import asyncio

logger = get_logger('MCTR')  # MemoryController


class MemoryController:
    def __init__(self, config_path: str, context_threshold: int = 4096) -> None:
        logger.info(f'Loading Config From {config_path} Context Len threshold: {context_threshold}')
        try:
            self.generator = Generation(config_path)
            config = self._readConfig(config_path)
            if config:
                self.dbmanager = DBManager(config.get('LTM_DATABASE_PATH', ''))
                self.dbmanager.create_database()
                self.LTM = LTM(self.generator, self.dbmanager)
                self.STM = STM(self.generator, config.get('STM_PATH', ''), config.get("STM_SIZE", 100))
                self.ContextThreshold = context_threshold - 100  # SAFE ZONE
            else:
                raise ValueError("CONFIG ERROR")
        except Exception as e:
            logger.info(f"Unwanted Error Check Config first {e}")
            logger.warning("Trying to load Memory Defaults")
            self.generator = Generation()
            self.dbmanager = DBManager()
            self.dbmanager.create_database()
            self.LTM = LTM(self.generator, self.dbmanager)
            self.STM = STM(self.generator)

    def load_stm(self):
        self.STM.load()

    def _readConfig(self, path):
        try:
            logger.info('Reading Config File...')
            with open(path, 'r', encoding='utf-8') as file:
                config = json.load(file)
        except Exception as e:
            logger.info(f"Read Error {e}")
            return {}
        return config

    async def BuildWorkingContext(self, query):
        try:
            logger.info('[BWC] Searching Memory')
            
            ltm_results = self.LTM.RetriEval.LTM_Search(query)  
            ltm_results = ltm_results[:10] if ltm_results else []

            stm_results, _, _ = self.STM.search(query, search_mode=True)  
            stm_results = stm_results[:5] if stm_results else []

            stm_top = ""
            if stm_results:
                stm_lines = []
                for s in stm_results[:5]:
                    timestamp = s.get('timestamp', 'N/A')
                    value = s.get('value', '')
                    stm_lines.append(f'[{timestamp}] {value}')
                stm_top = "\n".join(stm_lines)

            ltm_top = ""
            if ltm_results:
                ltm_lines = []
                for l in ltm_results[:5]:
                    ltm_lines.append(f"ID: {l.id}\n{l.value}")
                ltm_top = "\n".join(ltm_lines)

            total_items = len(stm_results) + len(ltm_results)
            logger.info(f"[BWC] Found {total_items} items")

            summary_result = ""
            if len(ltm_results) > 5:
                logger.info('[BWC] Creating Summary')
                extra_items = ltm_results[5:8]
                extra_content = "\n".join([f"{r.value}" for r in extra_items if r.value])
                if extra_content:
                    summary_result = self.generator.Generate_text(summary_prompt(extra_content))

            policy_retrieval = f"[LTM]\n{ltm_top}\n\n[STM]\n{stm_top}"
            if summary_result:
                policy_retrieval += f"\n\n[SUMMARY OF LOW RANKS]\n{summary_result}"

            working_context = Main_prompt(query, stm_top, ltm_top, summary_result)

            if len(working_context) >= self.ContextThreshold:
                logger.warning('[BWC] WorkingContext Under Pressure... Making Smaller')
           
                stm_top = ""
                if stm_results:
                    stm_lines = []
                    for s in stm_results[:3]:
                        timestamp = s.get('timestamp', 'N/A')
                        value = s.get('value', '')
                        stm_lines.append(f'[{timestamp}] {value}')
                    stm_top = "\n".join(stm_lines)

                ltm_top = ""
                if ltm_results:
                    ltm_lines = []
                    for l in ltm_results[:3]:
                        ltm_lines.append(f"ID: {l.id}\n{l.value}")
                    ltm_top = "\n".join(ltm_lines)

                summary_result = ""
                if len(ltm_results) > 3:
                    extra_items = ltm_results[3:6]
                    extra_content = "\n".join([f"{r.value}" for r in extra_items if r.value])
                    if extra_content:
                        summary_result = self.generator.Generate_text(summary_prompt(extra_content))

                policy_retrieval = f"[LTM]\n{ltm_top}\n\n[STM]\n{stm_top}"
                if summary_result:
                    policy_retrieval += f"\n\n[SUMMARY OF LOW RANKS]\n{summary_result}"
                
                working_context = Main_prompt(query, stm_top, ltm_top, summary_result)

            return working_context, policy_retrieval

        except Exception as e:
            logger.error(f"[BWC] WorkingContext Error {e}")
            import traceback
            logger.error(traceback.format_exc())
            return "NO Information", "NO Information"

    async def Memory_Policy_Layer(self, query, Assistant_output, policy_retrieval):
        try:
            logger.info(f"[MPL] Getting Policy Result")
            LlmPolicy = self.generator.Generate_text(
                Save_Decision_Prompt(query, Assistant_output, policy_retrieval)
            )
            Policy = parse_save_decision(LlmPolicy)

            if Policy:
                should_save = Policy.get('should_save', False)
                new_value = Policy.get('new_value', '')
                importance = Policy.get('importance', 0.0)
                logger.debug(f'Should_save: {should_save} Value: {new_value} Importance: {importance}')
                
                if not should_save:
                    return False

                value_to_save = new_value if new_value else query
                result = self.LTM.Save_LTM(value_to_save, importance)
                logger.info(f"{'[MPL] Requesting Save' if result else '[MPL] Not Saving Any Memory'}")
                return result if result else False

            return False

        except Exception as e:
            logger.error(f'[MPL] Unexpected Error: {e}')
            return False


    async def Agent_process(self, sender: str, query: str, user_id: str = None):
        try:
            logger.info(f"Processing Query from {sender}: {query[:50]}...")
            
            working_context, policy_retrieval = await self.BuildWorkingContext(query)
            
            Assistant_output = self.generator.Generate_text(working_context)

            self.STM.add(f"[USER_ID: {user_id} USERNAME: @{sender}]\n{query}\n[Diana]\n{Assistant_output}")
            # =================================
            
            await self.Memory_Policy_Layer(query, Assistant_output, policy_retrieval)
            
            return Assistant_output

        except Exception as e:
            logger.error(f'Generation Error {e}')
            import traceback
            logger.error(traceback.format_exc())
            return "SYSTEM ERROR"