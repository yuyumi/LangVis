import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', required=True, type=str, help='Prompt for generation')
    parser.add_argument('--num_tokens', default=1, type=int, help='How many new tokens to generate')
    parser.add_argument('--lang_model', default="gpt2", type=str, help="gpt2, gpt2-xl, gpt-nano")
    parser.add_argument('--random_state', default=0, type=int, help="Set a fixed random state to have reproducibility")
    parser.add_argument('--device', default="Default", type=str, help="Device for computations")
    parser.add_argument('--saliency',  default=True, action='store_true',
                        help="Return the saliency score for each new generated tokens")
    parser.add_argument('--saliency_metric', default="mean", type=str, help="mean or inputXGrad")
    parser.add_argument('--out_path', default="output.json", type=str, help="Path to the output json file")
    parser.add_argument('--attn_pairs', default=True, action='store_true',
                        help="Return key word pairs extracted from the attention output")
    parser.add_argument('--attn_layer_sel', default="attn_layer_11", type=str, help="Use which layer of attention to extract key word pairs")
    parser.add_argument('--agg_method', default="mean", type=str, help="Aggregation method for getting a summary attention from all attention heads")
    parser.add_argument('--num_tokens_buffed', default=5, type=int, help="How many tokens to ignore before starting to extract key word pairs")
    parser.add_argument('--return_output', default=False, action='store_true',)
    parser.add_argument('--do_sample', default=False, action='store_true',)

    cfg = parser.parse_args()

    return cfg