import numpy as np
from itertools import combinations, permutations

def create_fewshot_primer(prompt_data=None):
    """Creates the primer string for GPT in-context learning"""
    if prompt_data is None:
        raise ValueError("Please Specify Prompt Data")
        
    prompt = ''
    prompt += prompt_data['prefixes']['instructions'] + prompt_data['instructions'] + prompt_data['separator']
    
    for example in prompt_data['examples']:
        
        prompt += prompt_data['prefixes']['input'] + example['input'] + prompt_data['separator']
        prompt += prompt_data['prefixes']['output'] + example['output'] + prompt_data['separator']
        
    return prompt
    
def create_prompt(sentence, prompt_data=None):
    """Creates a prompt using the specified sentence for GPT in-context learning"""
    if prompt_data is None:
        raise ValueError("Please Specify Prompt Data")
        
    prompt_init = create_fewshot_primer(prompt_data)
    
    prompt = prompt_init + prompt_data['prefixes']['input'] + sentence + prompt_data['separator']
    prompt += prompt_data['prefixes']['output']
    
    return prompt   

# Partial primer & prompt functions
def create_partial_fewshot_primer(include = np.arange(8), prompt_data=None):
    """Creates the primer string for GPT in-context learning, filtering to include a subset of specified priming strings"""
    if prompt_data is None:
        raise ValueError("Please Specify Prompt Data")
    
    prompt = ''
    prompt += prompt_data['prefixes']['instructions'] + prompt_data['instructions'] + prompt_data['separator']
    
    # Grab each priming example in the specified order.
    for i in include:
        example = prompt_data['examples'][i]
        prompt += prompt_data['prefixes']['input'] + example['input'] + prompt_data['separator']
        prompt += prompt_data['prefixes']['output'] + example['output'] + prompt_data['separator']
        
    return prompt

def create_partial_prompt(sentence, include=np.arange(8), prompt_data=None):
    """Creates a prompt using the specified sentence and partial list of in-context primer sentences"""
    if prompt_data is None:
        raise ValueError("Please Specify Prompt Data")
        
    prompt_init = create_partial_fewshot_primer(include, prompt_data)
    
    prompt = prompt_init + prompt_data['prefixes']['input'] + sentence + prompt_data['separator']
    prompt += prompt_data['prefixes']['output']
    
    return prompt 