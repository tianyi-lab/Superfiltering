from tqdm import tqdm 
import numpy as np
import time
import json
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import math
from submodlib.functions.facilityLocation import FacilityLocationFunction

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def do_fla(X, number_all, number_select):
    start_time = time.time()

    Y = X
    obj = FacilityLocationFunction(n=number_all, mode="dense", data=Y, metric="euclidean")
    greedyList = obj.maximize(budget=number_select, optimizer='LazyGreedy', stopIfZeroGain=False, stopIfNegativeGain=False, verbose=False)
    idx_list = [tuple_i[0] for tuple_i in greedyList]

    print('FLA time used:',(time.time()-start_time)/60,'(min)')
    return idx_list

def do_ifd(args, json_data):
    def sort_key(x):
        # Check if the value is nan
        if math.isnan(x[args.key_name]):
            return (0, 0) 
        return (1, x[args.key_name]) 
    
    # The types for the ifd filtering
    # 1. Given the ifd_num, choose the highese ifd_num samples
    # 2. Not given the ifd_num, choose between bounds

    # Both the above needs to define the ifd_lupper
    if args.ifd_lupper != 0:
        filtered_data = [x for x in json_data if (isinstance(x[args.key_name], (int, float)) and x[args.key_name] < args.ifd_lupper)]
    else:
        filtered_data = json_data

    if args.ifd_num != 0:
        new_data = sorted(filtered_data, key=sort_key, reverse=True)
        new_data = new_data[:args.ifd_num]
    else:
        new_data = [x for x in filtered_data if (isinstance(x[args.key_name], (int, float)) and x[args.key_name] > args.ifd_lower)]
    
    print('Data length after ifd:', len(new_data))
    return new_data

def combine_sentences(args, filtered_data):
    sent_all = []
    for dict_i in filtered_data:

        if 'input' in dict_i.keys():
            instruction_i = dict_i['instruction'] + '\n' + dict_i['input'] + '\n'
        else:
            instruction_i = dict_i['instruction'] + '\n'

        if args.emb_type == 'whole':
            sent_all.append(instruction_i+dict_i['output'])
        elif args.emb_type == 'instruction':
            sent_all.append(instruction_i)
        elif args.emb_type == 'response':
            sent_all.append(dict_i['output'])

    return sent_all

# Function to get embeddings for a list of sentences
def get_sentence_embeddings(sentences, model_name, batch_size):

    #Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Initialize the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    all_sentence_embeddings = []

    for i in tqdm(range(0, len(sentences), batch_size)):
        # Process sentences in batches
        batch_sentences = sentences[i:i + batch_size]

        # Tokenize sentences and convert to input format expected by the model
        encoded_input = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors='pt').to(device)

        # Move encoded input to the same device as the model
        encoded_input = {k: v.to(model.device) for k, v in encoded_input.items()}

        # Get model's output (without any specific head)
        with torch.no_grad():
            model_output = model(**encoded_input)

        # Perform pooling
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        all_sentence_embeddings.append(sentence_embeddings)

    # Concatenate all embeddings from each batch
    all_sentence_embeddings = torch.cat(all_sentence_embeddings, dim=0)

    return all_sentence_embeddings.cpu()

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--json_data_path", type=str, default='')
    parser.add_argument("--json_save_path", type=str, default='')

    parser.add_argument("--ifd_num", type=int, default=520)
    parser.add_argument("--ifd_lower", type=float, default=0.9, help='the lower bound for ifd filtering')
    parser.add_argument("--ifd_lupper", type=float, default=1.0, help='the upper bound for ifd filtering')
    parser.add_argument("--key_name", type=str, default='ifd_ppl',help='ifd_ppl')

    parser.add_argument("--emb_model", type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    parser.add_argument("--emb_type", type=str, default='whole', help='whole, instruction, response')
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--fla_num", type=int, default=100)
    args = parser.parse_args()

    with open(args.json_data_path, 'r') as f:
        json_data = json.load(f)

    # filter by ifd
    filtered_data = do_ifd(args,json_data)

    # get the embedding
    sentences = combine_sentences(args, filtered_data)
    embeddings = get_sentence_embeddings(sentences, args.emb_model, args.batch_size)

    # do fla
    X = embeddings.numpy()
    fla_idxs = do_fla(X, len(sentences),args.fla_num)

    final_json_data_ori = [filtered_data[i] for i in fla_idxs]
    with open(args.json_save_path, 'w') as file:
        json.dump(final_json_data_ori, file, indent=4)

if __name__ == '__main__':
    main()