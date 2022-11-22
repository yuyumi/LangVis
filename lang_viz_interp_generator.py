import torch
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

import numpy as np
import argparse

import json

from config import getConfig

use_mingpt = True # Let's just use mingpt now. We can switch to huggingface's GPT model later if we think it's necessary


def generate(prompt, tokenizer, model, num_samples=1, steps=20, do_sample=False, saliency=False):    
    x = tokenizer(prompt).to(args.device)
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    # For consistency reason, do not sample output logits. We want results to be reproducible
    output = model.generate(x, max_new_tokens=steps, do_sample=do_sample, 
                            return_logits=saliency, save_grad=saliency)
    
    y = output[0]
    interm_output = output[1:]
    
    out = tokenizer.decode(y[0].cpu().squeeze())

    return y, out, interm_output


def main(args):
    set_seed(args.random_state)
    
    output_dict = {"input_prompt": args.prompt}
    
    out = args.prompt
    
    if args.saliency_metric == "IG":
        model.use_ig = True
    else:
        model.use_ig = False

    input_tokens = tokenizer(args.prompt)
    input_tokens = [tokenizer.decode(tok.unsqueeze(0)) for tok in input_tokens[0]]
    output_dict["input_tokens"] = input_tokens
    for i in range(args.num_tokens):
        y, out, interm_output = generate(prompt=out,
                                         tokenizer=tokenizer, 
                                         model=model,
                                         num_samples=1, 
                                         steps=1,
                                         do_sample=args.do_sample,
                                         saliency=args.saliency)

        token = tokenizer.decode(y[0][-1].unsqueeze(0))
        
        if "tokens" not in output_dict.keys():
            output_dict["tokens"] = []
        output_dict["tokens"].append(token)

        if args.saliency:
            if args.saliency_metric == "IG":
                aggregated_grads = interm_output[2].unsqueeze(0)
            else:
                logits = interm_output[1]
                tok_emb = interm_output[2]
                grads = torch.autograd.grad(tuple(torch.flatten(logits[..., y[0][-1]])), tok_emb, retain_graph=True)
                aggregated_grads = []
                for grad, inp in zip(grads, tok_emb):
                    assert args.saliency_metric in ["mean", "inputXGrad"]
                    if args.saliency_metric == "inputXGrad":
                        aggregated_grads.append(torch.norm(grad * inp, dim=-1))
                    elif args.saliency_metric == "mean":
                        aggregated_grads.append(grad.abs().mean(-1))

            if "saliency" not in output_dict.keys():
                output_dict["saliency"] = []
            output_dict["saliency"].append(aggregated_grads[0].cpu().detach()[0].tolist())
        torch.cuda.empty_cache()

    output_dict["output_full_text"] = out
    output_dict["output_token_list"] = [tokenizer.decode(token.unsqueeze(0)) for token in y[0]]
    
    if args.attn_pairs:
        y, out, interm_output = generate(prompt=out,
                                         tokenizer=tokenizer, 
                                         model=model,
                                         num_samples=1, 
                                         steps=1,
                                         do_sample=args.do_sample,
                                         saliency=args.saliency)
        attn = interm_output[0]
        output_dict["key_attn_pairs"] = {}
        output_dict["key_attn_pairs_ind"] = []
        if args.agg_method == "max":
            agg_attn = torch.amax(attn[args.attn_layer_sel][0], dim=0).cpu().detach()[1:, 1:]
        elif args.agg_method == "mean":
            agg_attn = torch.mean(attn[args.attn_layer_sel][0], dim=0).cpu().detach()[1:, 1:]
        elif args.agg_method.isnumeric():
            agg_attn = attn[args.attn_layer_sel][0][int(args.agg_method) - 1].cpu().detach()[1:, 1:]
            
        for i in range(agg_attn.shape[1]):
            key_ind = torch.argmax(agg_attn[i])
            # if key_ind == i or i < args.num_tokens_buffed:
            #     continue
            if i < args.num_tokens_buffed:
                continue
            # iqr = torch.quantile(agg_attn[i][:i+1], 0.75) - torch.quantile(agg_attn[i][:i+1], 0.25)
            # if 1.5 * iqr + torch.quantile(agg_attn[i][:i+1], 0.75) > agg_attn[i][key_ind]:
            # if 2 * torch.mean(agg_attn[i][:i+1]) > agg_attn[i][key_ind]:
                # continue
            
            key_word = output_dict["output_token_list"][key_ind+1].strip(" ")
            cur_word = output_dict["output_token_list"][i+1].strip(" ")
            
            if not (cur_word.isalpha() and key_word.isalpha()):
                continue
  
            key_word_ind = key_word + f"_{int(key_ind.cpu()+1)}"
            cur_word_ind = cur_word + f"_{i+1}"
            if key_word_ind not in output_dict["key_attn_pairs"].keys():
                output_dict["key_attn_pairs"][key_word_ind] = []
            output_dict["key_attn_pairs"][key_word_ind].append(cur_word_ind)
            
            output_dict["key_attn_pairs_ind"].append((key_word, cur_word, 
                                                  int(key_ind.cpu() + 1), i + 1))
        
    if args.return_output:
        return output_dict
    else:
        with open(args.out_path, "w") as outfile:
            json.dump(output_dict, outfile)

if __name__ == '__main__':
    args = getConfig()
    if args.device == "Default":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)    
else:
    args = argparse.Namespace()
    args.lang_model = "gpt2"
    args.random_state = 0
    args.saliency = True
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.saliency_metric = "mean"
    args.attn_layer_sel = "attn_layer_12"
    args.agg_method = "mean"
    args.num_tokens = 10
    args.num_tokens_buffed = 5
    args.attn_pairs = True
    args.return_output = True
    args.do_sample = False
    model = GPT.from_pretrained(args.lang_model)

    # ship model to device and set to eval mode
    model.to(args.device)
    model.eval()
    tokenizer = BPETokenizer()
    