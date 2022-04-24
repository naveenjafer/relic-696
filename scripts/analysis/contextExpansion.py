# Measures how much more content can be incorporated in the datasets.
import os
import json

from transformers import RobertaModel, RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

dataset = "../../RELiC"

def calculateContextExpansion():
    train_file = os.path.join(dataset, "train.json")
    book_wise_stats = {}
    with open(train_file) as file:
        train_data = json.load(file)
    for book in train_file:
        sentences = train_file[book]["sentences"]
        for quoteId in train_file[book]["quotes"]:
            quoteStart = train_file[book]["quotes"][quoteId][1]
            quoteLen = train_file[book]["quotes"][quoteId][2]
            quote = sentences[quoteStart:quoteStart+quoteLen]

            




if __name__ == "__main__":
    calculateContextExpansion()