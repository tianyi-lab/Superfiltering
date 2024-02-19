import os
import json
import torch
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

PROMPT_DICT_NONE = {
    "prompt_input": (
        "{instruction}\n{input}\n"
    ),
    "prompt_no_input": (
        "{instruction}\n"
    ),
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='data/alpaca_data/alpaca_data.json')
    parser.add_argument("--save_path", type=str, default='debug.jsonl')
    parser.add_argument("--model_name_or_path", type=str, default='gpt2')
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--prompt", type=str, default='none', help='none')
    args = parser.parse_args()
    return args

# Used to get the ppl and emb for the whole input
def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        with torch.no_grad(): 
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()

    except:
        return 0, 0

# Used to get the ppl and emb for part of input, used in conditional version, and token-wise loss
def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):

    try:
        input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

        start_index = text.rfind(target_span)
        start_token = len(tokenizer.encode(text[:start_index]))
        end_token = input_ids.shape[1]

        labels = input_ids.clone()
        labels[0, :start_token] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=labels)

        loss = outputs.loss
        perplexity = torch.exp(loss)

        return perplexity.to('cpu').item(), loss.to('cpu').item()
    
    except:
        return 0, 0


def main():

    args = parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto", cache_dir='../cache', output_hidden_states=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir='../cache')

    model.eval()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx != -1 else len(data)
    sampled_data = data[start_idx:end_idx]

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    with open(args.save_path, "r") as file:
        exsisting_num =  sum(1 for _ in file)
    sampled_data = sampled_data[exsisting_num:]


    if args.prompt == 'none':
        prompt_no_input = PROMPT_DICT_NONE["prompt_no_input"]
        prompt_input = PROMPT_DICT_NONE["prompt_input"]

    for i in tqdm(range(len(sampled_data))):

        data_i = sampled_data[i]
        instruct_i = data_i['instruction']
        output_i = data_i['output']

        input_i = data_i['input'] if 'input' in data_i.keys() else ''
        if input_i == '':
            temp_dict = {'instruction':instruct_i}
            promt_to_use = prompt_no_input.format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use

        else:
            temp_dict = {'instruction':instruct_i,'input':input_i}
            promt_to_use = prompt_input.format_map(temp_dict)
            whole_text = promt_to_use + output_i
            instruct_i = promt_to_use


        instruct_i_input_ids = tokenizer.encode(instruct_i, return_tensors="pt", truncation=True, max_length=args.max_length).to(device)
        instruct_i_len = instruct_i_input_ids.shape[1] 

        if output_i == '':
            temp_data_i = {}
        else:
            ppl_out_alone, loss_out_alone = get_perplexity_and_embedding_whole_text(tokenizer, model, output_i, args.max_length-instruct_i_len+1)
            ppl_out_condition, loss_out_condition = get_perplexity_and_embedding_part_text(tokenizer, model, whole_text, output_i, args.max_length)

            temp_data_i = {}
            temp_data_i['ppl'] = [0,ppl_out_alone,0,ppl_out_condition]
            temp_data_i['loss'] = [0,loss_out_alone,0,loss_out_condition]

        with open(args.save_path, "a") as file:
            file.write(json.dumps(temp_data_i) + '\n')
    
    print('Done: Data Analysis:',args.data_path)

if __name__ == "__main__":
    main()