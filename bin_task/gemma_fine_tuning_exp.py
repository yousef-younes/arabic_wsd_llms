
#This file contains the code necessary to do the gemma fine tuning on binary classification task using hugging face

from tqdm import tqdm
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os


import pandas as pd

from huggingface_hub import login
login(token="your_token")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use only GPUs 0, 1, 2


dev= pd.read_json("../data/wsd/bin_dev.json")

def preprocess_text(data):
  for i in range(len(data)):
    data.loc[i,'context']=data.loc[i,'context']
    data.loc[i,'sense']=data.loc[i,'sense']
    word=data.loc[i,'word']
    data.loc[i,'context'] = data.loc[i,'context'].replace(word, f"<token>{word}</token>")
    # break
  return data

dev=preprocess_text(dev)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use only GPUs 0, 1, 2

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-27b")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-27b",
    device_map="auto",
    torch_dtype="auto",  # Use mixed precision if available
    #device_map="balanced",  # <--- splits across all available GPUs
)



results = []
for i in tqdm(range(len(dev))):

    prompt = f"""
        You are a model specialized in Arabic word sense disambiguation.
        Given a context, a target word, and a candidate sense (all provided in Arabic),
        determine if the candidate sense correctly matches the meaning of the target word in the given context.

        The target word appears between <token> and </token> tags in the context.

        You MUST respond with exactly one character:
        - Output "1" if the candidate sense matches the meaning of the word in the context.
        - Otherwise, output "0".
        Do NOT output anything else.
        Do NOT output \n or whitespaces.

        Word: {dev['word'][i]}
        Context: {dev['context'][i]}
        Sense: {dev['sense'][i]}
        """

    #start_time = time.time()  # Start timer
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=32)
    results.append(tokenizer.decode(outputs[0]))
    #end_time = time.time()
    #elapsed = end_time - start_time
    
    #print(f"Elapsed time: {elapsed:.4f} seconds")  # Output: ~1.5005 seconds
    #print(tokenizer.decode(outputs[0]))
    

dev['Dev-gemma2-27b'] = results
dev.to_csv('./Dev-gemma2:27b', index=False)
