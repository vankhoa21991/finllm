import json
import datasets
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import datasets
import json
from tqdm import tqdm

dic = {
    0:"negative",
    1:'neutral',
    2:'positive',
}

fpb_datasets = load_dataset("financial_phrasebank", "sentences_50agree")
fpb_datasets = fpb_datasets["train"]
fpb_datasets = fpb_datasets.to_pandas()
fpb_datasets.columns = ["input", "output"]
fpb_datasets["output"] = fpb_datasets["output"].apply(lambda x:dic[x])
fpb_datasets["instruction"]  = "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."
fpb_datasets = datasets.Dataset.from_pandas(fpb_datasets)
fpb_datasets = fpb_datasets.train_test_split(seed = 42)['train']

train_dataset = datasets.concatenate_datasets([fpb_datasets]*6)   # we want each data source have similar number of samples

def make_label(x):
    if x < - 0.1: return "negative"
    elif x >=-0.1 and x < 0.1: return "neutral"
    elif x >= 0.1: return "positive"

def add_instructions(x):
    if x == "post":
        return "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}."
    else:
        return "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}."

dataset = load_dataset('pauri32/fiqa-2018')
dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"] ,dataset["test"] ])
dataset = dataset.to_pandas()
dataset["output"] = dataset.sentiment_score.apply(make_label)
dataset["instruction"] = dataset.format.apply(add_instructions)
dataset = dataset[['sentence', 'output',"instruction"]]
dataset.columns = ["input", "output","instruction"]
dataset = datasets.Dataset.from_pandas(dataset)
dataset = dataset.train_test_split(0.226, seed = 42)['train']

tmp_dataset = datasets.concatenate_datasets([dataset]*21)
train_dataset = datasets.concatenate_datasets([train_dataset, tmp_dataset])
print(tmp_dataset.num_rows)


social_media_dataset = load_dataset('zeroshot/twitter-financial-news-sentiment')
social_media_dataset = social_media_dataset['train']
social_media_dataset = social_media_dataset.to_pandas()
social_media_dataset['label'] = social_media_dataset['label'].apply(lambda x:dic[x])
social_media_dataset['instruction'] = 'What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.'
social_media_dataset.columns = ['input', 'output', 'instruction']
social_media_dataset = datasets.Dataset.from_pandas(social_media_dataset)

tmp_dataset = datasets.concatenate_datasets([social_media_dataset]*2)
train_dataset = datasets.concatenate_datasets([train_dataset,tmp_dataset])
print(tmp_dataset.num_rows)

finance_dataset = load_dataset('oliverwang15/news_with_gpt_instructions')
finance_dataset = finance_dataset['train'].to_pandas()
finance_dataset['output'] = finance_dataset['label']
finance_dataset["input"] = finance_dataset["news"]
finance_dataset["instruction"] = 'What is the sentiment of this news? Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/mildly positive/moderately positive/strong positive}, then provide some short reasons.'
finance_dataset = finance_dataset[['input', 'output', 'instruction']]
finance_dataset = datasets.Dataset.from_pandas(finance_dataset)

train_dataset = datasets.concatenate_datasets([train_dataset, finance_dataset])
all_dataset = train_dataset.shuffle(seed = 42)
print(all_dataset.shape)

def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

data_list = []
for item in all_dataset.to_pandas().itertuples():
    tmp = {}
    tmp["instruction"] = item.instruction
    tmp["input"] = item.input
    tmp["output"] = item.output
    data_list.append(tmp)
import pandas as pd
df = pd.DataFrame(data_list)
df.to_csv("data/dataset_fin.csv", index=False)
with open("data/dataset_fin.jsonl", 'w') as f:
    for example in tqdm(data_list, desc="formatting.."):
        f.write(json.dumps(format_example(example)) + '\n')

