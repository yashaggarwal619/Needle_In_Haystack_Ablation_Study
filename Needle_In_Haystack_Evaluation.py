import transformer_lens
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AutoTokenizer
from transformer_lens import HookedTransformer, utils
import os
import json
import pickle
import functools
import random
from langchain.evaluation import load_evaluator
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import random
import re


torch.cuda.empty_cache()



token=os.getenv("HF_KEY")


def extract_answer(text):
    match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return "No answer found."
def head_ablation_hook(attn_scores, hook, head):
    """
    Hook function to ablate a specific attention head.
    Sets the attention scores of the given head to negative infinity.
    """
    attn_scores[:, head, :, :] = -torch.inf
    return attn_scores





def ablate_random_heads(model, num_heads=100):
    """
    Randomly selects and applies ablation to a set of heads across layers.

    Args:
        model: The transformer model.
        num_heads: Number of heads to ablate.

    Returns:
        hooks: A list of ablation hooks for the selected heads.
    """
    total_layers = model.cfg.n_layers  # Assuming model has a config specifying layers
    total_heads_per_layer = model.cfg.n_heads  # Assuming each layer has equal heads

    # Randomly select (layer, head) pairs
    random_heads = [
        (random.randint(0, total_layers - 1), random.randint(0, total_heads_per_layer - 1))
        for _ in range(num_heads)
    ]

    hooks = [
        (
            utils.get_act_name("attn_scores", layer),
            functools.partial(head_ablation_hook, head=head)
        )
        for layer, head in random_heads
    ]

    return hooks, random_heads  # Return selected heads for reference

def generate_text_with_random_ablation(model, input_text, tokenizer,hooks,num_heads=30, max_length=50):
    """
    Generates text from the model with multiple randomly ablated attention heads.

    Args:
        model: The transformer model.
        input_text: The input string.
        tokenizer: The tokenizer to convert text to tokens.
        num_heads: Number of random heads to ablate.
        max_length: The maximum length of the output text.

    Returns:
        output_text: The generated output string.
        ablated_heads: The list of ablated (layer, head) pairs.
    """
    # Tokenize input text
    input_tokens = tokenizer(input_text, return_tensors="pt")["input_ids"]

   

    # Store generated tokens
    generated_tokens = input_tokens.tolist()[0]  # Convert to list for appending
    
    # Generate iteratively
    with torch.no_grad():
        for _ in range(max_length):
            logits = model.run_with_hooks(
                torch.tensor([generated_tokens]),  # Convert to tensor
                return_type="logits",
                fwd_hooks=hooks,
               
            )
    
            # Get next token (argmax or sampling)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    
            # Append and stop if EOS token is generated
            generated_tokens.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    # Decode output tokens into text
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return extract_answer(output_text)

def ablate_specific_heads(model, df, num_heads=30):
    """
    Ablates a given number of specific heads from a dataframe.

    Args:
        model: The transformer model.
        df: Pandas dataframe containing `layer` and `head` columns sorted by importance.
        num_heads: Number of heads to ablate.

    Returns:
        hooks: A list of ablation hooks for the selected heads.
        ablated_heads: The list of (layer, head) pairs ablated.
    """
    # Select the top `num_heads` heads to ablate
    selected_heads = df.head(num_heads)[["layer", "head"]].values.tolist()

    hooks = [
        (
            utils.get_act_name("attn_scores", layer),
            functools.partial(head_ablation_hook, head=head)
        )
        for layer, head in selected_heads
    ]

    return hooks, selected_heads 

def generate_text_with_specific_ablation(model, input_text, tokenizer, num_heads=30, df=None, max_length=50,hooks=None):
    """
    Generates text from the model with multiple randomly ablated attention heads.

    Args:
        model: The transformer model.
        input_text: The input string.
        tokenizer: The tokenizer to convert text to tokens.
        num_heads: Number of random heads to ablate.
        max_length: The maximum length of the output text.

    Returns:
        output_text: The generated output string.
        ablated_heads: The list of ablated (layer, head) pairs.
    """
    # Tokenize input text
    input_tokens = tokenizer(input_text, return_tensors="pt")["input_ids"]

    # Create ablation hooks for random heads
    # hooks, ablated_heads = ablate_specific_heads(model, df,num_heads=num_heads)

    # Store generated tokens
    generated_tokens = input_tokens.tolist()[0]  # Convert to list for appending
    
    # Generate iteratively
    with torch.no_grad():
        for _ in range(max_length):
            logits = model.run_with_hooks(
                torch.tensor([generated_tokens]),  # Convert to tensor
                return_type="logits",
                fwd_hooks=hooks,
               
            )
    
            # Get next token (argmax or sampling)
            next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    
            # Append and stop if EOS token is generated
            generated_tokens.append(next_token)
            if next_token == tokenizer.eos_token_id:
                break

    # Decode output tokens into text
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return extract_answer(output_text)



def insert_needle_in_haystack(needle: str, haystack: str, depth: float, max_token_length: int) -> str:
    """
    Inserts the needle into the haystack at a specified depth while ensuring
    the total length does not exceed max_token_length.
    
    :param needle: The string to insert into the haystack.
    :param haystack: The main text where the needle will be inserted.
    :param depth: A float between 0 and 1 indicating the insertion depth.
    :param max_token_length: The maximum length of the final output.
    :return: The modified haystack with the needle inserted.
    """
    
    # Tokenize by words (assuming words roughly represent tokens)
    haystack_tokens = haystack.split()
    needle_tokens = needle.split()
    
    # Ensure we have space for the needle
    available_space = max_token_length - len(needle_tokens)
    
    if available_space <= 0:
        raise ValueError("Max token length is too small to fit the needle.")
    
    # Trim haystack to fit within the max token length
    haystack_tokens = haystack_tokens[:available_space]
    
    # Compute insertion index based on depth
    insert_index = int(len(haystack_tokens) * depth)
    
    # Insert the needle at the calculated position
    modified_tokens = haystack_tokens[:insert_index] + needle_tokens + haystack_tokens[insert_index:]
    
    # Ensure final length does not exceed max_token_length
    modified_tokens = modified_tokens[:max_token_length]
    
    return " ".join(modified_tokens)


def load_process_text(file_path):
    complete_haystack=''
    for file in os.listdir(file_path):
        with open(f"Haysteack_text/{file}", "r") as file:
            content = file.read()
        complete_haystack+='\n\n'+content
    return complete_haystack


def generate_final_prompt(question, complete_context):
    input_text = f'''Given the context answer the following question:
Context: {complete_context}

Question: {question}
just provide the answer and nothing else.
Answer:'''

    return input_text

def run_needle_in_haystack_induction_heads(model_hf,tokenizer,df,needle, question, haystack):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # needle="The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    # question='What is the best thing to do in San Francisco?'
    
    
    num_heads_ablation=[1,5,10,20,30,40,50,60,80,90,100,110,120,130,140,150]
    needle_depth=[0,0.5,1]

    
    
    final_dict={
        'heads_ablated':[],
        'needle':[],
        'needle_depth':[],
        'complete_context':[],
        'question':[],
        'answer':[],
        'ablated_heads':[]
        
    }

   
    for num_head in num_heads_ablation:
        # Create ablation hooks for random heads
        hooks, ablated_heads = ablate_specific_heads(model_hf, df,num_heads=num_head)
        for depth in needle_depth:

            haystack_with_needle=insert_needle_in_haystack(needle, haystack, depth, 1000)
            
            final_prompt=generate_final_prompt(question, haystack_with_needle)

            # final_answer=generate_text_with_random_ablation(model_hf, final_prompt, tokenizer,hooks,num_head, max_length=50)
            final_answer= generate_text_with_specific_ablation(model_hf, final_prompt, tokenizer, num_head, df=df, max_length=50,hooks=hooks)
            final_dict['heads_ablated'].append(num_head)
            final_dict['needle'].append(needle)
            final_dict['needle_depth'].append(depth)
            final_dict['complete_context'].append(haystack_with_needle)
            # final_dict['needle'].append(needle)
            final_dict['question'].append(question)
            final_dict['answer'].append(final_answer)
            final_dict['ablated_heads'].append(str(ablated_heads))
           
            
        

    return final_dict



def run_needle_in_haystack_random(model_hf, tokenizer,needle, question, haystack):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # needle="The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    # question='What is the best thing to do in San Francisco?'

    
    num_heads_ablation=[1,5,10,20,30,40,50,60,80,90,100,110,120,130,140,150]
    needle_depth=[0,0.5,1]

    

    
    final_dict={
        'heads_ablated':[],
        'needle':[],
        'needle_depth':[],
        'complete_context':[],
        'question':[],
        'answer':[],
        'ablated_heads':[]
        
    }
    for num_head in num_heads_ablation:
        # Create ablation hooks for random heads
        hooks, ablated_heads = ablate_random_heads(model_hf, num_heads=num_head)
        for depth in needle_depth:

            haystack_with_needle=insert_needle_in_haystack(needle, haystack, depth, 500)
            
            final_prompt=generate_final_prompt(question, haystack_with_needle)

            final_answer=generate_text_with_random_ablation(model_hf, final_prompt, tokenizer,hooks,num_head, max_length=50)

            final_dict['heads_ablated'].append(num_head)
            final_dict['needle'].append(needle)
            final_dict['needle_depth'].append(depth)
            final_dict['complete_context'].append(haystack_with_needle)
            # final_dict['needle'].append(needle)
            final_dict['question'].append(question)
            final_dict['answer'].append(final_answer)
            final_dict['ablated_heads'].append(str(ablated_heads))
         
            
        

    return final_dict


def main(model='meta-llama/Llama-3.2-1B-Instruct',induction_map_path='induction_score_heat_map/meta-llama-Llama-3.2-1B-Instruct induction heads.csv',text_file_path='Haysteack_text/'):

    needle="The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day."
    question='What is the best thing to do in San Francisco?'

    
    tokenizer = AutoTokenizer.from_pretrained(
           model,
            use_auth_token=token,
   
    # cache_dir="/nfs/nfs8/home/scratch/yaggarw"
        )

    model_hf = HookedTransformer.from_pretrained(
    model,
    use_auth_token=token,
    n_devices=4, 
    torch_dtype= torch.bfloat16,
    device_map="cuda"
# cache_dir="nfs/nfs8/home/scratch/yaggarw",
    )

    df=pd.read_csv(induction_map_path)

    haystack=load_process_text(text_file_path)
    
    final_dict_induction=run_needle_in_haystack_induction_heads(model_hf,tokenizer,df,needle, question, haystack)
    final_dict_random=run_needle_in_haystack_random(model_hf,tokenizer,needle, question, haystack)

    filemodel_name=model.split('/')[-1]

    complete_file_name_induction=f'{filemodel_name}_1000_induction.csv'
    complete_file_name_random=f'{filemodel_name}_1000_random.csv'

    pd.DataFrame(final_dict_induction).to_csv(complete_file_name_induction)
    pd.DataFrame(final_dict_random).to_csv(complete_file_name_random)
    return final_dict_induction,final_dict_random

main()
