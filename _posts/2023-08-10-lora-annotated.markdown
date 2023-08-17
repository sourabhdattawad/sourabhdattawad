---
layout: post
title:  "Implementing LoRA: Low-Rank Adaptation of Large Language Models from scratch (Annotated paper)"
date:   2023-08-10 14:45:45 +0200
categories: Machine Learning Design
comments: true
---
# Implementing LoRA: Low-Rank Adaptation of Large Language Models from scratch (Annotated paper) :heart_eyes:

# Overview

We are aware that fine-tuning large language models poses challenges due to two primary factors:

- It demands a significant quantity of GPUs, which might not always be accessible or financially feasible :money_mouth_face:.
- Pre-trained models experience [catastrophic forgetting](https://arxiv.org/abs/1612.00796) of acquired parameters when trained for a particular task, causing the initial model's performance to decline across different tasks :roll_eyes:. 

Therefore, are there any approaches to enhance the efficiency of model fine-tuning :question: 

- Certainly, [adapter](https://arxiv.org/pdf/1902.00751.pdf) methods involving sequentially trainable parameters. However, it's important to note that these techniques introduce extra inference latency due to the inability to parallelize sequential computations :broken_heart:. 

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685v2.pdf) introduces a novel, parameter-efficient approach for fine-tuning. It integrates trainable parameters for fine-tuning purposes, all while avoiding any supplementary inference latency :heart_eyes:.

I have annotated the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685v2.pdf), covering all important aspects. You can enlarge the picture by clicking on it :grin:.

<a href="{{ site.base_path }}/assets/images/posts/annotated_lora.png" class="test-popup-link">
   <img src="{{ site.base_path }}/assets/images/posts/annotated_lora.png" width="400" height="500px"/>
</a>

In this blog, we're going to create a complete implementation of `LoRA` using Python 🐍. While there's a fantastic implementation by [HuggingFace](https://github.com/huggingface/peft/tree/main) that you might find interesting, here we'll focus on building everything from scratch.

# Implementation

We'll make use of the `bert-base-cased` model provided by [HuggingFace](https://huggingface.co/) to apply the `LoRa` technique. To provide a brief understanding of `BERT`, it's a model primarily focused on encoding, designed to predict missing words within input text. It employs a structure built on the `transformer` architecture. In this demonstration, we'll fine-tune `BERT` for sentiment analysis on movie reviews by adding a classification head onto `bert-base-cased`. Furthermore, we'll introduce a mechanism to integrate `LoRA` layers into the encoder modules :muscle:.

Let us install a few libraries first,
{%highlight bash%}
pip install torch datasets transformers evaluate
{%endhighlight%}


# Load IMDB dataset

{%highlight python%}

from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import torch.nn as nn
from torch.nn import Linear
import copy
import evaluate
import numpy as np
import math

model_type = "bert-base-cased"
imdb = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained(model_type)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

tokenized_imdb = imdb.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}

{%endhighlight%}

# Load pre-trained dataset

{%highlight python%}
model = AutoModelForSequenceClassification.from_pretrained(model_type, num_labels=2, id2label=id2label, label2id=label2id)
print(model)
{%endhighlight%}

The model architecture,

{%highlight python%}
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(28996, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
{%endhighlight%}

Examining the encoder module, we observe a total of 12 BertLayers. Within each layer's self-attention mechanism, there exist weights for query (WQ), value (WV), and key (WK).

{%highlight python%}
(encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
        ...       
        )
      )
)
{%endhighlight%}

# The LoRA Concept

According to the research paper, the training process focuses exclusively on weight matrices A and B, while keeping all other weights in a frozen state.

`WQ_new = WQ_old * x + B1A1 * x`

`WV_new = WV_old * x + B2A2 * x`

`WK_new = WK_old * x + B3A3 * x`

What should the configuration of these weight matrices resemble :question: 

- Matrix `A` possesses dimensions of `d*r`, where `d` signifies the dimension of weight matrix `W`, and `r` represents the new rank, which can be `8` or `16`. You can consider this as a hyperparamter to be tuned. We initiate `A` with Gaussian initialization.
- Matrix `B` holds dimensions of `r*d`, where `d` indicates the dimension of weight matrix `W`, and `r` corresponds to the projected dimension of Matrix `A`. Initially, all elements in `B` matrix are established as `0`.

# Implementing LoRA module

Below is the `PyTorch` code that puts into practice the concepts I just explained :statue_of_liberty:.

{% highlight python%}
class LoRAModule(nn.Module):
    def __init__(self, layer, r=8, alpha=16):
        super().__init__()
        # Store the original layer
        self.W = layer
        # Initialize LoRa parameters
        self.LoRA_A = Linear(in_features=768, out_features=r, bias=False)
        self.LoRA_B = Linear(in_features=r, out_features=768, bias=False)
        # Initialize LoRa parameters' weights
        self.reset_params()
        # Store the scaling factor for LoRa
        self.scaling_factor = alpha / r

    def reset_params(self):
        # Initialize LoRA_A weights using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.LoRA_A.weight, a=math.sqrt(5))
        # Initialize LoRA_B weights with zeros
        nn.init.zeros_(self.LoRA_B.weight)

    def forward(self, x):
        # Apply LoRa transformation: W(x) + LoRA_B(LoRA_A(x)) * scaling_factor
        x = self.W(x) + self.LoRA_B(self.LoRA_A(x)) * self.scaling_factor
        return x
{%endhighlight%}

# Updating the original BERT model with LoRA module

Before we add the LoRA module, we make a copy of the original model. This copy helps us merge weights to avoid inference latency, which I'll explain more about in the later part of this article :monkey: :monkey:.

{% highlight python%}
def update_model_with_lora_weights(model):
    # Loop through the 12 BertLayer instances in the encoder
    for i in range(12):
        # Replace the query component of self-attention with LoRAModule by retaining weights
        model.bert.encoder.layer[i].attention.self.query = LoRAModule(
            model.bert.encoder.layer[i].attention.self.query
        )
        # Replace the value component of self-attention with LoRAModule by retaining weights
        model.bert.encoder.layer[i].attention.self.value = LoRAModule(
            model.bert.encoder.layer[i].attention.self.value
        )
    # Return the updated model with LoRA weights
    return model


model_copy = copy.deepcopy(model)
model_with_lora = update_model_with_lora_weights(model_copy)
print(model_with_lora)
{%endhighlight%}

Now, we observe that the original BERT model has been integrated with the LoRA Module.

{% highlight python%}
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(28996, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0-11): 12 x BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): LoRAModule(
                (W): Linear(in_features=768, out_features=768, bias=True)
                (LoRA_A): Linear(in_features=768, out_features=8, bias=False)
                (LoRA_B): Linear(in_features=8, out_features=768, bias=False)
              )
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): LoRAModule(
                (W): Linear(in_features=768, out_features=768, bias=True)
                (LoRA_A): Linear(in_features=768, out_features=8, bias=False)
                (LoRA_B): Linear(in_features=8, out_features=768, bias=False)
              )
              (dropout): Dropout(p=0.1, inplace=False)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
            (intermediate_act_fn): GELUActivation()
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)
{%endhighlight%}


# Setting trainable parameters

{% highlight python%}

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )
print_trainable_parameters(model_with_lora)
{%endhighlight%}

{% highlight python%}
trainable params: 108606722 || all params: 108606722 || trainable%: 100.0
{%endhighlight%}

As we see, all the weights in the model are configured to be trainable. However, we only need to designate the weights of the `LoRA` and `classification head` components as trainable :nerd_face:. 

{% highlight python%}
def set_trainable_params(model):
    for name, param in model.named_parameters():
        if not "LoRA" in name and not "classifier" in name:
            param.requires_grad = False

set_trainable_params(model_with_lora)
print_trainable_parameters(model_with_lora)

{%endhighlight%}
{% highlight python%}
trainable params: 296450 || all params: 108606722 || trainable%: 0.27295732210755796
{%endhighlight%}

We only need to train `0.27%` of the total parameters :heart_eyes:.

# Setup evaluation metrics

{% highlight python%}
accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
{%endhighlight%}

# Setup training procedure

{% highlight python%}
training_args = TrainingArguments(
    output_dir="bert-cased-lora-imdb",
    learning_rate=1e-3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
)
trainer = Trainer(
    model=model_with_lora,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
{%endhighlight%}

Now we're fully prepared to train the model :sunglasses:!

# Setup training procedure

{% highlight python%}
trainer.train()
{%endhighlight%}

# Results
{% highlight python%}
Epoch	Training Loss	Validation Loss	    Accuracy
0	     0.719200	      0.751393	    0.500000
1	     0.267300	      0.209935	    0.917600
2	     0.213700	      0.203721	    0.925000
3	     0.153000	      0.209234	    0.929640

{%endhighlight%}

The loss is going down after each epoch. I trained it for 3 epochs, but you can train it more until the model converges.

We are not done yet :upside_down_face:...

# Merging the weights to reduce inference latency

Let us look back at the equation, 

`WQ_new = WQ_old * x + B1A1 * x`

After learning the parameters A1 and B1, we can merge the two computations into a single process :wink:.

`WQ_new = (WQ_old + B1A1) * x`

This straightforward technique enables `LoRA` Model inference without any additional latency.

Let's modify the code to achieve the same objective. We will proceed to combine the weights within the original `model` not `model_with_lora`.

{% highlight python%}
def merge_lora_weights(model, model_with_lora):
    for i in range(12):
        # Get the LoRA and original self-attention modules for the i-th layer
        lora_module = model_with_lora.bert.encoder.layer[i].attention.self
        org_module = model.bert.encoder.layer[i].attention.self
        
        # Compute merged weights for the query component
        lora_query_weights = torch.matmul(
            lora_module.query.LoRa_B.weight, lora_module.query.LoRa_A.weight
        )
        merged_query_weights = torch.add(lora_module.query.W.weight, lora_query_weights)
        org_module.query.weights = merged_query_weights
        
        # Compute merged weights for the value component
        lora_value_weights = torch.matmul(
            lora_module.value.LoRa_B.weight, lora_module.value.LoRa_A.weight
        )
        merged_value_weights = torch.add(lora_module.value.W.weight, lora_value_weights)
        org_module.value.weights = merged_value_weights

merge_lora_weights(model, model_with_lora)

{%endhighlight%}

The weights in the `model` have been modified. You can now employ this updated `model` for your inference tasks.

Congratulations! You have now successfully implemeted LoRA from scratch :trophy:.

