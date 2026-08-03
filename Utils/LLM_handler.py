import ast
import re
import json

def parse_llm_list(response):
    """  
    Parse LLM output into a list of numbers.
    (RETRIEVAL LLM RELEVANT CHECK)
    """
    try:
   
        if '[' in response and ']' in response:
            start = response.find('[')
            end = response.find(']') + 1
            list_str = response[start:end]
            return ast.literal_eval(list_str)
    except:
        pass

    numbers = re.findall(r'\d+', response)
    return [int(n) for n in numbers] if numbers else []
def get_importance_from_response(response):
    """Extract a float in LLM output.
    (FOR SETTING MEMORY IMPORTANCE)
    """

    match = re.search(r'\b(0(?:\.\d{1,2})?|1(?:\.0{1,2})?)\b', response)
    if match:
        return float(match.group(1))
    return 0.7  

def parse_merge_response(response):

    import json
    import re
    
    response = response.strip()

    if response == '{}' or response == '':
        return {}

    try:

        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            if 'id' in result and result['id'] is not None:
                return {
                    'id': int(result['id']),
                    'new_value': result.get('new_value', ''),
                    'new_importance': float(result.get('new_importance', 0.5))
                }
    except:
        pass

    id_match = re.search(r'id["\s:]+(\d+)', response, re.IGNORECASE)
    if id_match:
        item_id = int(id_match.group(1))

        value_match = re.search(r'new_value["\s:]+"([^"]+)"', response)
        new_value = value_match.group(1) if value_match else ''

        imp_match = re.search(r'new_importance["\s:]+([\d.]+)', response)
        new_imp = float(imp_match.group(1)) if imp_match else 0.5
        
        return {
            'id': item_id,
            'new_value': new_value,
            'new_importance': new_imp
        }
    
    return {}

def parse_duplicate_response(response):
    """
    Parse duplicate detector response
    
    Args:
        response: LLM response string
    
    Returns:
        dict: {'id': int, 'new_value': str} or None
    """
    response = response.strip()

    if response == '{}' or response == '':
        return None
    
    try:

        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            if 'id' in result and result['id']:
                return {
                    'id': int(result['id']),
                    'new_value': result.get('new_value', '')
                }
    except:
        pass
    
    id_match = re.search(r'id["\s:]+(\d+)', response, re.IGNORECASE)
    if id_match:
        item_id = int(id_match.group(1))
        value_match = re.search(r'new_value["\s:]+"([^"]+)"', response)
        new_value = value_match.group(1) if value_match else ''
        return {'id': item_id, 'new_value': new_value}
    
    return None


def parse_save_decision(response):
    """Parse save decision response from LLM"""
    import json
    import re
    
    response = response.strip()
    

    if response == '{}' or response == '':
        return {'should_save': False, 'new_value': '', 'importance': 0.0}
    
    try:

        json_match = re.search(r'\{[^{}]*\}', response)
        if json_match:
            result = json.loads(json_match.group())
            

            value_field = result.get('new_value', result.get('value', ''))
            
            return {
                'should_save': result.get('should_save', False),
                'new_value': value_field,
                'importance': float(result.get('importance', 0.0))
            }
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        pass
    

    response_lower = response.lower()
    if 'true' in response_lower or 'save' in response_lower:

        value_match = re.search(r'new_value["\s:]+"([^"]+)"', response)
        if value_match:
            return {
                'should_save': True,
                'new_value': value_match.group(1),
                'importance': 0.7
            }
        return {'should_save': True, 'new_value': '', 'importance': 0.5}
    
    return {'should_save': False, 'new_value': '', 'importance': 0.0}