```python
%load_ext autoreload
%autoreload 2
```

# Finetuning large language models


```python
#!pip install langchain langchain-community transformers bitsandbytes accelerate langchain-openai langchain evaluate langchain-together
```


```python
#importing libraries

import dotenv, os
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_together import ChatTogether
from langchain_openai import ChatOpenAI
from enum import Enum
from langchain.output_parsers.enum import EnumOutputParser
from utils import debug_llm, extract_json



dotenv.load_dotenv()
```




    True



In the previous chapter, we explored the inner workings of large language models and how they can be leveraged for various tasks, such as text generation and sequence classification, through effective prompting and zero-shot capabilities. We also delved into the vast array of pre-trained models available, courtesy of the vibrant community.

However, will these pre-trained models exhibit remarkable versatility, their general purpose training may not always be optimized for specific tasks or domains. Fine-tuning emerges as a crucial techique to adapt and refine a language model's understanding to the nuances of a particular dataset or a task

Consider the field of medical research, where a language model pre-trained solely on general web text may struggle to perform effectively out-of-the-box. By fine-tuning the model on a corpus of medical literature, its ability to generate relevant medical text or assist in information extraction from healthcare documents can be significantly enhanced.

Conversational models present another compelling use case. As discussed earlier, large pre-trained models are primarily trained to predict the next token, which may not seamlessly translate to engaging, conversational interactions. By fine-tuning these models on datasets containing everyday conversations and informal language structures, we can adapt their outputs to emulate the natural flow and nuances of interfaces like ChatGPT.

The primary objective of this chapter is to establish a solid foundation in fine-tuning large language models (LLMs). Consequently, we will delve into the following key areas:

## Chapter Outline
- Classifying the topic of a text using LLAMA 3
- Finetuning the classification using prompt engineering
- Finetuning LLAMA 3 with retraining on the dataset
- Parameter-efficient fine-tuning techniques that enable training on smaller GPUs


Through this comprehensive exploration, you will gain insights into tailoring language models to excel in specific tasks and domains, unleashing their true potential for a wide range of applications.

## Text Classification

As we discussed in earlier chapters, LLMs are generally used for generative tasks where task is to predict the next token. Other NLP tasks such as text classification, named entity recognition might not be represented easily with the default objective. Here we will see an example of using LLMs for text classification and then further finetuning to improve the metrics. 

### Identify a dataset

Let's pick publicly available dataset to demonstrate the technique. Here we'll use AG news dataset, a well known non-commercial dataset used for benchmarking text classification models and researching data mining, information retrieval and data streaming.

Here, we will explore the dataset to know about the text and labels. The dataset provides 120,000 training examples, more than enough data to fine-tune a model with 4 classification labels. Fine-tuning requires very little data compared to pre-training a model and just using few thousand examples should enough to get a good baseline model.


```python
from datasets import load_dataset
import evaluate

accuracy_metric = evaluate.load("accuracy")
raw_datasets = load_dataset("ag_news")
```

Let's print the first sample from the training dataset. Output shows each sample is a dictionary with two keys: `text` and `label` . The `text` key contains the actual text of the news article, while the `label` key contains an integer representing the category of the article. In this particular example, article is labeled with integer `2`, which corresponds to `business` category according to the dataset's label encoding scheme


```python
raw_datasets['train'][0]
```




    {'text': "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.",
     'label': 2}




```python
labels = raw_datasets['train'].features['label'].names
labels
```




    ['World', 'Sports', 'Business', 'Sci/Tech']



Before the era of LLMs, we used RNNs or BERT-style models to capture the meaning of a sentence and then fine-tuned them for downstream tasks. Now, let's explore how to achieve similar results using LLMs.

But supervised learning isn't the only option for text classification with LLMs. Unsupervised learning through prompt engineering has emerged as a viable alternative. How well do LLMs perform text classification when guided only by a natural language prompt? Can this approach compete with the results from supervised learning? We'll explore these questions and more in the next section.

## Prompt Engineering

Let's look at zero-shot capability of large language models (LLMs) where they can perform a task without explicit training data for that specific task.

We first need to create a dictionary `id_to_label` that maps the lowercase label names to their corresponding integer labels.

Finally, we modify the dictionary `id_to_label` to expand the `sci/tech` label. This is useful since LLM might be trained with full words rather than abbrevations. We should try to be close to the initial vocabulary.


```python
id_to_label = {l.lower():i for i,l in enumerate(labels)}
id_to_label['science/technology']=3 #expanding one of the label
```


```python
id_to_label
```




    {'world': 0,
     'sports': 1,
     'business': 2,
     'sci/tech': 3,
     'science/technology': 3}



We now need to set the prompt engineering pipeline for text classification using an LLM. Here we'll use langchain to add prompts to the text. 

We define a `ChatPromptTemplate` and `tagging_prompt` tht will be used to construct the prompt for the LLM. The prompt instructs the LLM to extract the news label from the given article based on the `Classification` . We'll try to batch multiple sentences together to save up the computation cost.




```python
from datasets import Dataset, DatasetDict

dpdf = raw_datasets['train'].to_pandas()
samples_df = dpdf.groupby('label').apply(lambda x: x.sample(10)).reset_index(drop=True)

def get_label(example):
    example['str_label']=labels[example['label']]
    return example

samples = (Dataset.from_pandas(samples_df)
            .map(get_label))
```

    /tmp/ipykernel_226512/1917432805.py:4: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
      samples_df = dpdf.groupby('label').apply(lambda x: x.sample(10)).reset_index(drop=True)



    Map:   0%|          | 0/40 [00:00<?, ? examples/s]


Here's our first example


```python
example = pd.Series(samples[0])
example
```




    text         India Test Fires Nuclear-Capable Missile NEW D...
    label                                                        0
    str_label                                                World
    dtype: object



Next, we define a `classification` class that inherits from Pydantic `BaseModel`. This class represents the structured output format that the LLM will generate. It has a single field `label` of type `str`, which is an enumeration of the four label categories: 

- World
- sports
- business
- science/technology


```python
#tagging_prompt.format(inputs=enumerate(inputs), examples=zip(samples['text'],samples['str_label']), count=len(inputs))
```


```python
from typing import List
from langchain.output_parsers import PydanticOutputParser

class Classification(BaseModel):
    label:str = Field(enum=['world','sports','business','science/technology'], description='Label for the article') # Note: Using the expanded label of sci/tech

```

Our approach involves utilizing public APIs to obtain labels for articles. To optimize the process, we'll bundle multiple articles into a single request. This way, instead of receiving one result per request, we'll get a list of results in a single API call. To streamline the conversion process, we'll define an additional class that accommodates the list of classification labels returned by the Large Language Model (LLM).


```python
class Results(BaseModel):
    results:List[Classification]=[]
```

We initialize the Large Language Model (LLM) using the ChatTogether class from the `langchain_together` library. In this example, we utilize the LLAMA 3 model from the Together API. If you have any doubts about setting up the request, please refer to the first chapter. We set the temperature parameter to 0 to ensure that the LLM generates deterministic outputs. Furthermore, we employ the with_structured_output method to instruct the LLM to generate outputs in the format specified by the Classification class


```python

parser = PydanticOutputParser(pydantic_object=Results)
```


```python
# LLM
#llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")

tagging_prompt = ChatPromptTemplate.from_template(
"""
##Instruction:
Extract the News Label for below {{count}} articles. 

Make sure you give output to all the articles in the same order and use the labels from the following list:
1. world
2. sports
3. business
4. science/technology

Use article number as the reference and provide article number in the responses along with their labels. 

Make sure to only use above labels and do not add anything extra.

## Output format
{{formatting_instructions}}

Please note output should only be returned in JSON format. Reject all the other formats. 
##Input:
            {%for i,c  in inputs %}
            Article: {{i}}
            Text: {{c}}
            
            {% endfor %}


""",template_format='jinja2'
)

llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    max_tokens=1024,
    temperature=0.1
)

tagging_chain = tagging_prompt  | llm | parser
```

This creates a langchain object which parses the input string in above template, pass it to LLM , result is the passed to Parser which knows how to extract json from the text and convert it into a `Pydantic` class. 

## Invoking the LLM for text classification

Here, we invoke the LLM for text classification using `tagging_chain` we created earlier. First, we prepare 5 random articles for which we want to get the labels. It is the passed to LLM with formatting instructions. Remember, we need output in json format so that it can be automatically parsed by `Pydantic`. 


```python
inputs = samples['text'][:5]
results = tagging_chain.invoke({"inputs": enumerate(inputs), 
                      'examples':zip(samples['text'],samples['str_label']), 
                      'count':len(inputs),
                     'formatting_instructions': parser.get_format_instructions()})
```

Here we obtain the outcomes from the LLM call. Upon examination, it's evident that the length of results matches that of inputs in this instance. It's important to remember that LLMs are generative engines, so there's no guarantee that labels will be provided for all inputs. We are prepared to accept this risk in our example. In a production environment, we would prefer to process these examples individually to ensure that each article receives a label. During post-processing, we will filter out any articles that have empty labels.


```python
results
```




    Results(results=[Classification(label='world'), Classification(label='world'), Classification(label='world'), Classification(label='world'), Classification(label='world')])




```python
print('length of results ', len(results.results))
```

    length of results  5


In the example above, we correctly found the labels for 5 random articles without training! Isn't it amazing. We get the output right out of the box.

We can then repeat this process to all the test samples. Below, we call `tagging_chain.invoke` for all the examples and save output as one of the feature in huggingfaace dataset. Feature is stored as `response` 


```python
def process_text(examples):
    inputs = examples['text']
    out = tagging_chain.invoke({'inputs':enumerate(inputs),'examples': zip(samples['text'], samples['str_label']), 
                                'count':len(inputs), 'formatting_instructions': parser.get_format_instructions()})

    
    if len(out.results) != len(inputs):
        examples['response']=['empty']*len(inputs)
    else:
        examples['response'] = [r.label for r in out.results[:len(inputs)]]
    return examples

sample = raw_datasets['test'].shuffle().select(range(300))
dataset_processed = sample.map(process_text,num_proc=1,batched=True, batch_size=10)
```

    Parameter 'function'=<function process_text at 0x7f388b5b3b50> of the transform datasets.arrow_dataset.Dataset._map_single couldn't be hashed properly, a random hash was used instead. Make sure your transforms and parameters are serializable with pickle or dill for the dataset fingerprinting and caching to work. If you reuse this transform, the caching mechanism will consider it to be different from the previous calls and recompute everything. This warning is only showed once. Subsequent hashing failures won't be showed.



    Map:   0%|          | 0/300 [00:00<?, ? examples/s]



```python
dataset_processed
```




    Dataset({
        features: ['text', 'label', 'response'],
        num_rows: 300
    })




```python
dataset_processed.save_to_disk('data/processed.hf')
```


    Saving the dataset (0/1 shards):   0%|          | 0/300 [00:00<?, ? examples/s]


### Evaluation

Let's evaluate the model with accuracy metrics. This will show how many of the responses are correct.


```python
references = [r['label'] for r in dataset_processed if r['response'] != 'empty']
predictions = [id_to_label[r['response']] for r in dataset_processed if r['response'] != 'empty']
accuracy = accuracy_metric.compute(references=references,predictions=predictions)
print('accuracy {}'.format(accuracy['accuracy']))
```

    accuracy 0.85


Total accuracy found is 85%. 

Let's try to improve the accuracy through further prompt engineering

## Few shot prompt engineering

Above we worked on text classification without providing any help to the model. Few shot engineering refers to the step where we provide certain examples to the model to help in the response. In the prompt below, we provde placeholder for examples. These are extracted from `train` split of dataset to make sure we don't cheat with our test dataset. We group records by labels and extract 10 examples from each label ( world, sci/tech, sports, business ]. Let's oberve what effect it can have on the accuracy


```python
# LLM
#llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")

tagging_prompt = ChatPromptTemplate.from_template(
"""
##Instruction:
Extract the News Label for below {{count}} articles. 

Make sure you give output to all the articles in the same order and use the labels from the following list:
1. world
2. sports
3. business
4. science/technology

Use article number as the reference and provide article number in the responses along with their labels. 

Make sure to only use above labels and do not add anything extra.
## Here are some examples
{% for  example, label in examples %}
Text: {{example}}
label: {{label}}

{% endfor %}
## Output format
{{formatting_instructions}}

Please note output should only be returned in JSON format. Reject all the other formats. 
##Input:
            {%for i,c  in inputs %}
            Article: {{i}}
            Text: {{c}}
            
            {% endfor %}


""",template_format='jinja2'
)

llm = ChatTogether(
    model="meta-llama/Llama-3-70b-chat-hf",
    max_tokens=1024,
    temperature=0.1
)

tagging_chain = tagging_prompt  | llm | parser
```


```python
# inputs = samples['text'][:10]
# results = tagging_chain.invoke({"inputs": enumerate(inputs), 
#                       'examples':zip(samples['text'],samples['str_label']), 
#                       'count':len(inputs),
#                      'formatting_instructions': parser.get_format_instructions()})
```


```python
def process_text(examples):
    inputs = examples['text']
    out = tagging_chain.invoke({'inputs':enumerate(inputs),'examples': zip(samples['text'], samples['str_label']), 
                                'count':len(inputs), 'formatting_instructions': parser.get_format_instructions()})
    
    if len(out.results) != len(inputs):
        examples['response']=['empty']*len(inputs)
    else:
        examples['response'] = [r.label for r in out.results[:len(inputs)]]
    return examples

sample = raw_datasets['test'].shuffle().select(range(100))
dataset_processed = sample.map(process_text,num_proc=1,batched=True, batch_size=10)
```


    Map:   0%|          | 0/100 [00:00<?, ? examples/s]


### Evaluation

Let's evaluate the model again with accuracy metrics.


```python
references = [r['label'] for r in dataset_processed if r['response'] != 'empty']
predictions = [id_to_label[r['response']] for r in dataset_processed if r['response'] != 'empty']
accuracy = accuracy_metric.compute(references=references,predictions=predictions)

```

Accuracy jumped to 87.5% . It shows that additional information helps LLM in text classification. We can further finetune the prompt by adding more examples as context window allows it, negative examples , hard negative examples and so on. Iterating on prompts is an art itself and LLM allows it to modify and see the impact instantly. 

### Tasks
1. Try improving the accuracy by adding negative examples. Use `##Negative examples` as the header and obtain examples from a validation set( not shown here ).
2. Checkout out `dspy` package which allows to iteratively optimize the prompt based on an objective. 

# Finetuning the model

After going through few prompt engineering models and understanding the impact, We will now move towards finetuning the model. This will ensure improvement in accuracy with the cost of computation. 

There are two main approaches to fine-tuning large language models (LLMs) for text classification tasks.

1. **Building an Entire Domain-specific Model from scratch**

- This approach involves training a foundational model entirely on industry-specific knowledge and data, using self-supervised learning techniques like next-token prediction and masking

- It requires a massive amount of domain specific data and significant computational resources

- An example of this approach is **BloombergGPT** which was trained on decades of financial data, requiring $2.7 million and 53 days of training

- The advantage of this approach is that the resulting model is highly specialized and tailored to the specific domain, potentially leading to better performance on domain specific tasks

2. **Finetuning a pretrained LLM**

- This approach involves taking a pre-trained LLM such as GPT OR BERT, and fine tuning it on a smaller, domain specific dataset.
  
- It requires less data computation, and time compared to training from scratch, making it more efficient and cost-effective option

  
- Various techniques can be employed to enhance the fine-tuning process, such as transfer learning, retrieval-augmented generation (RAG), and a multi-task learning

  
- RAG combines the strengths of pre-trained models and information retrieval systems, enabling the model to retrieve and incorporate domain-specific knowledge during inference

  
- Multi-task learning involves training a single model on multiple related tasks simultaneously, allowing the model to learn shared representations and benefit from task synergies.

### Transfer Learning

In above methods, we used online API to call the LLM , generate the output and parse the results into the desired format. Let's see how we can fine-tune LLAMA 3 model for a text classification task

There are again two ways to finetune the LLAMA 3 model.

1. Using transfer learning on the embeddings extracted from LLAMA 3
2. Finetuning large language model on the new dataset

Approach 1 is more suited when we have less amount of training data. 

We need to first download and prepare model for training. Most of the time, we don't have enough resources to work on 70b model ( estimated GPU size required > 48GB ), hence we use quantized model

### Background

Imagine teaching a dog a new trick after it's already mastered basic commands. You're not starting from zero; you're building on existing knowledge. Similarly, transfer learning involves taking a pre-trained language model, often trained on massive datasets like Wikipedia or the entirety of the internet, and fine-tuning it on a smaller, task-specific dataset. This way, we can inherit the general language understanding of the pre-trained model and adapt it to excel at a particular task like sentiment analysis or text summarization.

The process of **fine-tuning** a large language model involves adapting its weights and biases using your specific dataset. This is typically done by continuing the training process on your new data, but with a lower learning rate. Think of it as a gentle nudge in the right direction, refining the model's existing knowledge to align with your task.

### Quantization: Shrinking the giants

While transfer learning saves us from training a model from scratch, large language models are still resource-intensive. That's where quantization comes in. Quantization is a technique to reduce the model's size and memory requirements by representing its parameters (weights and biases) using lower precision data types.

Here's how the code demonstrates quantization:


```python
from transformers import BitsAndBytesConfig, AutoModelForSequenceClassification
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit = True, 
    bnb_4bit_quant_type = 'nf4',
    bnb_4bit_use_double_quant = True, 
    bnb_4bit_compute_dtype = torch.bfloat16 
)

model_name = "meta-llama/Meta-Llama-3-8B"

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    num_labels=4,
    device_map='auto'
)
```

In this snippet:

- load_in_4bit = True: Loads the model parameters using 4-bit precision.
- bnb_4bit_quant_type = 'nf4': Specifies a specific quantization method ('nf4').
- bnb_4bit_use_double_quant = True: Uses a more advanced quantization scheme for better performance.
- bnb_4bit_compute_dtype = torch.bfloat16: Computations are done in a 16-bit floating-point format.

By applying quantization, we can significantly reduce the model's size and memory footprint, making it more accessible for deployment on resource-constrained devices.

### Lora Adapters

LoRA (Low-Rank Adaptation) is a technique designed to make fine-tuning large language models even more efficient. Instead of updating all the model's parameters, LoRA focuses on a small set of new parameters introduced into specific layers of the model.

The provided code showcases the LoRA configuration:


```python
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

lora_config = LoraConfig(
    r = 16, 
    lora_alpha = 8,
    target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    lora_dropout = 0.05, 
    bias = 'none',
    task_type = 'SEQ_CLS'
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
```

Here's a breakdown of the parameters:

- r: The rank of the low-rank matrices used in LoRA, controlling the number of trainable parameters.
- lora_alpha: A scaling factor for the LoRA update.
- target_modules: Specifies the specific layers (query, key, value, output projections) in which to introduce the new parameters.
- lora_dropout: A dropout rate to prevent overfitting during training.

The `prepare_model_for_kbit_training` function optimizes the model for efficient training, and `get_peft_model` applies the LoRA configuration.

### Training

Below code finetunes the model for news classification task. Please note, it takes more than 24 hours to train LLAMA3 7B model even after quantization and rquires atleast 24GB of VRAM. It shows how big the models are.


```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir = 'agnews_classification',
    learning_rate = 1e-4,
    per_device_train_batch_size = 8,
    per_device_eval_batch_size = 64,
    num_train_epochs = 3,
    logging_steps=1,
    weight_decay = 0.01,
    evaluation_strategy = 'epoch',
    save_strategy = 'epoch',
    load_best_model_at_end = True,
    report_to="none"
)
```

    /opt/anaconda3/envs/torch-nightly/lib/python3.8/site-packages/tqdm/auto.py:21: TqdmWarning: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html
      from .autonotebook import tqdm as notebook_tqdm
    The cache for model files in Transformers v4.22.0 has been updated. Migrating your old cache. This is a one-time only operation. You can interrupt this and resume the migration later on by calling `transformers.utils.move_cache()`.
    0it [00:00, ?it/s]
    /opt/anaconda3/envs/torch-nightly/lib/python3.8/site-packages/transformers/training_args.py:1474: FutureWarning: `evaluation_strategy` is deprecated and will be removed in version 4.46 of 🤗 Transformers. Use `eval_strategy` instead
      warnings.warn(



```python
collate_fn = DataCollatorWithPadding(tokenizer=tokenizer)

```


```python
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets['train'],
    eval_dataset = tokenized_datasets['test'],
    tokenizer = tokenizer,
    data_collator = collate_fn,
    compute_metrics = compute_metrics,
)

train_result = trainer.train()
```

### Comparison



The choice of fine-tuning approach depends on the delicate balance between data availability, computational resources, and desired performance. Zero-shot learning offers a rapid and inexpensive option, but its performance is often suboptimal. Few-shot learning strikes a middle ground, providing a boost in performance with minimal data. Full fine-tuning, while resource-intensive, generally leads to the highest performance but requires substantial data.


| Approach | Data Requirements | Computational Cost | Performance | Adaptability | When to Use |
|----------|-------------------|---------------------|-------------|--------------|-------------|
| Zero-Shot | None | Low | Variable | Low | When data is scarce or for quick experimentation |
| Few-Shot | Minimal | Low | Moderate | Moderate | When some examples are available but data is limited |
| Full Fine-Tuning | Large | High | High | High | When ample task-specific data is available and resources permit |



## Reference

- Text classification using large language models
  
https://aclanthology.org/2023.findings-emnlp.603.pdf
