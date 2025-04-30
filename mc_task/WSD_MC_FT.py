#this module contains the code to fine tune small language models like mBert to do the WSD as a multichoice task

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"


os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import transformers
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer


from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))
    #!nvidia-smi

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


model_checkpoint = "CAMeL-Lab/bert-base-arabic-camelbert-msa"

#"UBC-NLP/ARBERTv2" 

#"CAMeL-Lab/bert-base-arabic-camelbert-msa" #"bert-base-multilingual-uncased" #"aubmindlab/bert-base-arabertv2"  #"CAMeL-Lab/bert-base-arabic-camelbert-ca" #"bert-base-multilingual-uncased"
batch_size = 2 


datasets = load_dataset('json', data_files={'train': '../data/wsd/new_train_43.json','dev':'../data/wsd/new_dev_43.json'}) #{'train':'../data/wsd/camel_train4.json','dev':'../data/wsd/camel_dev4.json'})


#number of max possible glosses
max_num_choices = 43


tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)


ending_names = []
for i in range(max_num_choices):
    ending_names.append("gloss"+str(i))

print(ending_names)

def preprocess_function(examples):
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[context] * max_num_choices for context in examples["context"]]
    # Grab all second sentences possible for each context.
    question_headers = examples["word"]
    second_sentences = [[f"{header}: {examples[end][i]}" for end in ending_names] for i, header in enumerate(question_headers)]

    # Flatten everything
    first_sentences = sum(first_sentences, [])
    second_sentences = sum(second_sentences, [])

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, max_length=512, truncation=True, padding=True)
    # Un-flatten
    return {k: [v[i:i+max_num_choices] for i in range(0, len(v), max_num_choices)] for k, v in tokenized_examples.items()}


encoded_datasets = datasets.map(preprocess_function, batched=True)


model = AutoModelForMultipleChoice.from_pretrained(model_checkpoint)

model_name = model_checkpoint.split("/")[-1]
args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-on-43-choices",
    learning_rate=5e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=False,
    save_safetensors=False,
    seed= 42,
    #max_grad_norm = 0.5
    #push_to_hub=True,
)



@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        #print(len([features[i].keys() for i in range(len(features)) if len(features[i])<2]))
        #print("**************************************************")
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [[{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


accepted_keys = ["input_ids", "attention_mask", "label"]
features = [{k: v for k, v in encoded_datasets["train"][i].items() if k in accepted_keys} for i in range(10)]
batch = DataCollatorForMultipleChoice(tokenizer)(features)
import numpy as np
import numpy as np
import evaluate

# Load the metrics from 'evaluate' library
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_predictions):
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)

    # Calculate all metrics
    accuracy = accuracy_metric.compute(predictions=preds, references=label_ids)["accuracy"]
    precision = precision_metric.compute(predictions=preds, references=label_ids, average="macro")["precision"]
    recall = recall_metric.compute(predictions=preds, references=label_ids, average="macro")["recall"]
    f1 = f1_metric.compute(predictions=preds, references=label_ids, average="macro")["f1"]

    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

def compute_metrics_old(eval_predictions):
    print(eval_predictions)
    predictions, label_ids = eval_predictions
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}


trainer = Trainer(
    model,
    args,
    train_dataset=encoded_datasets["train"],
    eval_dataset=encoded_datasets["dev"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForMultipleChoice(tokenizer),
    compute_metrics=compute_metrics,
)

trainer.train()
