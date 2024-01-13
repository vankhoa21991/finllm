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
from peft import PeftModel
def evaluate(y_true, y_pred):
	mapping = {'positive': 2, 'neutral': 1, 'none': 1, 'negative': 0}

	def map_func(x):
		return mapping.get(x, 1)

	y_true = np.vectorize(map_func)(y_true)
	y_pred = np.vectorize(map_func)(y_pred)

	# Calculate accuracy
	accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
	print(f'Accuracy: {accuracy:.3f}')

	# Generate accuracy report
	unique_labels = set(y_true)  # Get unique labels

	for label in unique_labels:
		label_indices = [i for i in range(len(y_true))
						 if y_true[i] == label]
		label_y_true = [y_true[i] for i in label_indices]
		label_y_pred = [y_pred[i] for i in label_indices]
		accuracy = accuracy_score(label_y_true, label_y_pred)
		print(f'Accuracy for label {label}: {accuracy:.3f}')

	# Generate classification report
	class_report = classification_report(y_true=y_true, y_pred=y_pred)
	print('\nClassification Report:')
	print(class_report)

	# Generate confusion matrix
	conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
	print('\nConfusion Matrix:')
	print(conf_matrix)
def inference(pipe, prompt):
    result = pipe(prompt)
    answer = result[0]['generated_text'].split("=")[-1]
    return answer

def predict(X_test, model, tokenizer):
    y_pred = []
    pipe = pipeline(task="text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=1,
                    temperature=0.01,
                    # device='cuda'
                    )
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        answer = inference(pipe, prompt)

        if "positive" in answer:
            y_pred.append("positive")
        elif "negative" in answer:
            y_pred.append("negative")
        elif "neutral" in answer:
            y_pred.append("neutral")
        else:
            y_pred.append("none")
    return y_pred

