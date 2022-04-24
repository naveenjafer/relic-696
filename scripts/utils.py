from turtle import right
import numpy as np
import pickle

NUM_SENTS = 6
quote_window = 5

from transformers import RobertaModel, RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
from retriever_train.dataset_config import DATASET_CONFIG, BASE_CONFIG

def build_lit_instance(quotes, left_sents, right_sents, all_sents, append_quote=True, mask_token="<mask>"):

    """Make a context using left / right sentences. Use <mask> token for quote."""
    inst = []
    for quote in quotes:
        if left_sents > 0:
            left_context = " ".join(quote[0][-left_sents:])
        else:
            left_context = ""
        if right_sents > 0:
            right_context = " ".join(quote[3][:right_sents])
        else:
            right_context = ""
        full_context = left_context + f" {mask_token} " + right_context

        if append_quote:
            #print(quote)
            #quit(1)
            left_start = quote[1]-quote_window
            if left_start < 0:
                left_start = 0
            left_quote = all_sents[left_start : quote[1]]
            right_quote = all_sents[quote[1]+quote[2] : quote[1]+quote[2]+quote_window]



            left_quote, left_quote = truncateQuoteBoundaries(left_quote, quote[-1], left_quote)

            left_quote_context = " ".join(left_quote)
            right_quote_context = " ".join(right_quote)

            actual_quote = left_quote_context + ' </s> ' + quote[-1] + ' </s> ' + right_quote_context

            inst.append(
                [" ".join(full_context.split()), " ".join(actual_quote.split())]
            )
        else:
            inst.append(
                " ".join(full_context.split())
            )
    #print(inst[0])
    #quit(1)
    return inst

def getTokensLen(text):
    tokenized = tokenizer(text)
    #print(len(tokenized["input_ids"]))
    return len(tokenized["input_ids"])

def truncateQuoteBoundaries(left_quote, quote, right_quote):
    quoteTokenLen = getTokensLen(" ".join(left_quote) + quote + " ".join(right_quote))
    flipDigit = -1
    counter = 0
    while quoteTokenLen > BASE_CONFIG["max_suffix_length"]:
        counter += 1
        if flipDigit < 0:
            # remove from left
            left_quote = left_quote[1:]
        else:
            # remove from right
            right_quote = right_quote[:-1]
        flipDigit *= -1
        if len(left_quote) == 0 and len(right_quote) == 0:
            break
        if counter == 11:
            break
        quoteTokenLen = getTokensLen(" ".join(left_quote) + quote + " ".join(right_quote))
    return left_quote, left_quote

def pickle_load(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data


def pickle_dump(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)


def build_candidates(book_data):
    candidates = {}
    all_sentences = book_data["sentences"]
    for ns in range(1, NUM_SENTS):
        candidates[ns] = [" ".join([x.strip() for x in all_sentences[idx:idx + ns]])
                          for idx in book_data["candidates"][f"{ns}_sentence"]]
        assert all([ii == xx for ii, xx in enumerate(book_data["candidates"][f"{ns}_sentence"])])
    return candidates


def print_results(results):
    keys = ["mean_rank", "recall@1", "recall@3", "recall@5", "recall@10", "recall@50", "recall@100", "num_candidates"]
    if len(results[1]["recall@1"]) > 0:
        for ns in range(1, NUM_SENTS):
            print(f"\nResults with {ns} sentence quotes ({len(results[ns]['mean_rank'])} instances):")
            for key in keys:
                print(f"{key} = {np.mean(results[ns][key]):.4f}", end=', ')
            print("")

        # print overall results
        total_instances = sum([len(results[ns]['mean_rank']) for ns in range(1, NUM_SENTS)])
        print(f"\nResults with all quotes ({total_instances} instances):")
        for key in keys:
            all_results = [x for ns in range(1, NUM_SENTS) for x in results[ns][key]]
            print(f"{key} = {np.mean(all_results):.4f}", end=', ')
        print("")
    else:
        print("Not printing results since the answers are hidden for this split...")
