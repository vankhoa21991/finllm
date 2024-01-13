import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import Dataset
from peft import LoraConfig, PeftConfig
from trl import SFTTrainer
from transformers import (AutoModelForCausalLM,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          TrainingArguments,
                          pipeline,
                          logging)
from sklearn.metrics import (accuracy_score,
                             classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from peft.utils import TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
import datasets
from finbot.example_16_evaluate import evaluate, predict
from peft import PeftModel


def prepare_data():
    filename = "data/dataset_fin.csv"

    df = pd.read_csv(filename)
    df['text'] = df['input']
    df['sentiment'] = df['output'].map({'moderately positive': 'positive',
                                        'moderately negative': 'negative',
                                        'strong positive': 'positive',
                                        'strong negative': 'negative',
                                        'mildly positive': 'positive',
                                        'mildly negative': 'negative',
                                        'neutral': 'neutral',
                                        'negative': 'negative',
                                        'positive': 'positive'
                                        })

    X_train = list()
    X_test = list()
    for sentiment in ["positive", "neutral", "negative"]:
        train, test = train_test_split(df[df.sentiment == sentiment],
                                       train_size=300,
                                       test_size=300,
                                       random_state=42)
        X_train.append(train)
        X_test.append(test)

    X_train = pd.concat(X_train).sample(frac=1, random_state=10)
    X_test = pd.concat(X_test)

    eval_idx = [idx for idx in df.index if idx not in list(train.index) + list(test.index)]
    X_eval = df[df.index.isin(eval_idx)]
    X_eval = (X_eval
              .groupby('sentiment', group_keys=False)
              .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
    X_train = X_train.reset_index(drop=True)

    X_train = pd.DataFrame(X_train.apply(generate_prompt, axis=1),
                           columns=["text"])
    X_eval = pd.DataFrame(X_eval.apply(generate_prompt, axis=1),
                          columns=["text"])

    y_true = X_test.sentiment
    X_test = pd.DataFrame(X_test.apply(generate_test_prompt, axis=1), columns=["text"])

    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)

    return train_data, eval_data, X_test, y_true
def generate_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative".

            [{data_point["text"]}] = """.strip()



def init_model(model_name = "meta-llama/Llama-2-7b-chat-hf"):
    compute_dtype = getattr(torch, "float16")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True
    )

    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                              trust_remote_code=True,
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

def train(model, tokenizer, train_data, eval_data):
    # target_modules = TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING['phi']
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules=target_modules
    )

    training_arguments = TrainingArguments(
        output_dir="logs",
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8, # 4
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        evaluation_strategy="epoch"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=1024,
    )

    # Train model
    trainer.train()

    # Save trained model
    output_dir ="results/trained-model-ex11"
    trainer.save_model(output_dir)

def init_model_from_checkpoint(peft_model = "results/trained-model-ex11"):
    compute_dtype = getattr(torch, "float16")

    config = PeftConfig.from_pretrained(peft_model)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        quantization_config=bnb_config,
        use_auth_token=False,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    # model = AutoPeftModelForCausalLM.from_pretrained(peft_model)

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path,
                                              trust_remote_code=True,
                                              )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # config.init_lora_weights = False
    #
    # model = get_peft_model(model, config)
    # model.add_adapter("lora", peft_config=config)
    # model.enable_adapters()
    model = PeftModel.from_pretrained(model, peft_model, device_map="auto")
    # model = model.merge_and_unload()
    # model = model.to(DEV)
    model.eval()
    return model, tokenizer

def validate(X_test, y_true, model, tokenizer):
    y_pred = predict(X_test, model, tokenizer)
    evaluate(y_true, y_pred)
    return y_pred

def main():
    train_data, eval_data, X_test, y_true = prepare_data()
    model, tokenizer = init_model()
    validate(X_test, y_true, model, tokenizer)
    train(model, tokenizer, train_data, eval_data)

    y_pred = validate(X_test, y_true, model, tokenizer)
    evaluation = pd.DataFrame({'text': X_test["text"],
                               'y_true': y_true,
                               'y_pred': y_pred},
                              )
    evaluation.to_csv("results/test_predictions_ex15.csv", index=False)

def load_and_validate():
    train_data, eval_data, X_test, y_true = prepare_data()
    model, tokenizer = init_model_from_checkpoint(peft_model="results/trained-model-ex11/")
    validate(X_test, y_true, model, tokenizer)

if __name__=='__main__':
    load_and_validate()
    # main()