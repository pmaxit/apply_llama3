# Building LLAMA 3 from scratch


```python
from torch import nn

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from collections import Counter

from tokenizers import Tokenizer, normalizers, pre_tokenizers
from tokenizers.models import WordLevel
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Digits, Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from gensim.parsing.preprocessing import preprocess_string
import datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PreTrainedModel, PretrainedConfig
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast, CausalLMOutputWithCrossAttentions

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, Trainer, TrainingArguments, AutoModelForCausalLM, AutoConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
```

This chapter guides you through the process of building a LLAMA 3 model from scratch using PyTorch and Hugging Face Transformers. While we won't replicate the full scale of LLAMA 3 due to computational constraints, you'll gain a solid understanding of the core concepts and implementation steps.

## Chapter Outline

1. Building block for a language model
2. N-gram language model
3. RNN neural network using attention
4. Transformer Architecture: The Foundation
6. The Decoder-Only GPT-2 Model
7. Training with Huggingface API
8. (Optional) converting it into a Chat like format

Imagine a young apprentice learning at the feet of Shakespeare, soaking in his vast knowledge word by word. Over time, the apprentice learns to anticipate the next word, the next phrase, the next line of verse. This is the essence of a language model - a system that predicts the next word, given the ones that come before.

In today's digital world, language models are powered by algorithms and trained on massive datasets of text, much like that apprentice studying Shakespeare's plays. They've become essential tools for a variety of tasks:

- Writing Assistance: They help us write emails, craft essays, and even generate creative content.
- Translation: They bridge language barriers by translating text from one language to another.
- Conversation: They power chatbots and voice assistants, engaging in conversations with us.

## Building block

At its core, a langauge model is a statistical model. It analyzes the patterns and probabilities of words occuring together in a vast corpus of text. The more data it's exposed to, the better it becomes at predicting the next word in a sequence.

Think of it like this:

- **Tokenization**: The model breaks down text into smaller units called tokens (words, punctuation, etc.).
- **Pattern Recognition**: It learns the relationships between these tokens, understanding which words are likely to follow others.
- **Prediction**: Given a sequence of words, it calculates the probability of different words coming next and chooses the most likely one.

### Different Flavors of Language Models:

1. **N-gram Models**: These simpler models look at a fixed number of previous words (bigrams consider two, trigrams consider three) to predict the next.
2. **Neural Network Models**: These more sophisticated models use artificial neural networks to capture complex patterns in language.
3. **Transformer Models**: The latest breakthrough, these models use attention mechanisms to weigh the importance of different words in a sequence, leading to remarkable performance.

## N-gram model

An n-gram is a sequence of n words. For instance, "please turn" and "turn your" are bigrams (2-grams), while "please turn your" is a trigram (3-gram). N-gram models estimate the probability of a word given the preceding n-1 words.

To calculate the probability of a word w given a history h, we can use relative frequency counts from a large corpus:

$P(w|h) = C(hw) / C(h)$

where:

- P(w|h) is the probability of word w given history h.
- C(hw) is the count of the sequence hw in the corpus.
- C(h) is the count of the history h in the corpus.

However, this approach is limited due to the vastness and creativity of language. Many possible word sequences might not exist in even the largest corpus.

### Bi-gram model using pytorch

Here, we will implement bi-gram model using pytorch. Although simple but bigram model can surprise the readers with its surprising predictive power and ability to capture meaningful patterns in text data.

### Bigram model

A bigram model operates on fundamental premise: the probability of a word appearing in a text sequence is heavily influences by the word that preceeds it. By analyzing the large corpora of text, we can calculate the additional probabilities of a word pairs. For instance, the probability of encountering the word 'morning' given the preceeding word 'good' is relatively high

Let's illustrate this concept with an example using the following text corpus

"The cat sat on the mat. The dog chased the cat"

**1. Tokenization**
- Split the corpus into individual words: ["the", "cat", "sat", "on", "the", "mat", "the", "dog", "chased", "the", "cat"]

**2. Create bi-gram pairs**
- Pair consecutive words: [("the", "cat"), ("cat", "sat"), ("sat", "on"), ("on", "the"), ("the", "mat"), ("mat", "the"), ("the", "dog"), ("dog", "chased"), ("chased", "the"), ("the", "cat")]

**3. Calculate probabilities**
- Count the occurence of each bi-gram pair
- Calculate the probability of second word given the first word
  e.g. $$P( cat | the ) = 2/4$$

### Dataset

To train our language model, we will work on wikipedia dataset. In the following section, we will download the raw wikitext, tokenize it and prepare for training


```python
from datasets import load_dataset
import torch

# Load the Wikitext dataset (this might take a while for the first download)
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Preprocessing function to tokenize the text
def tokenize_function(examples):
    return {"text": [text.lower().split() for text in examples["text"]]}

# Apply the preprocessing function
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
```

This code loads the Wikitext dataset, a large collection of text from Wikipedia articles. It then tokenizes the text, breaking it down into individual words (or tokens), and converts the words to lowercase. The original text is removed, leaving a dataset where each entry is a list of lowercase word tokens. This is a standard preprocessing step in natural language processing (NLP) tasks.

### Pytorch implementation


```python
# Combine all tokens for vocabulary and matrix construction
all_tokens = [token for tokens in tokenized_dataset["train"]["text"] for token in tokens]

vocab = list(set(all_tokens))
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

bigram_counts = torch.ones((len(vocab), len(vocab)))

for tokens in tokenized_dataset["train"]["text"]:
    for i in range(len(tokens) - 1):
        bigram_counts[word_to_idx[tokens[i]], word_to_idx[tokens[i + 1]]] += 1

bigram_probs = bigram_counts / bigram_counts.sum(dim=1, keepdim=True)
```

This code creates a vocabulary of all unique words from the training data and assigns each word a numerical index. It then builds a matrix (bigram_counts) to track the frequency of each word pair (bigram) that appears in the training text. Finally, it calculates the probability of each bigram occurring given the preceding word (bigram_probs). This is a key step in creating a bigram language model.


```python
class BigramModel:
    def __init__(self, vocab, bigram_probs):
        self.vocab = vocab
        self.bigram_probs = bigram_probs

    def calculate_probability(self, sentence):
        prob = 1.0  # Initial probability
        words = sentence.lower().split()  # Ensure case-insensitivity
        for i in range(len(words) - 1):
            if words[i] in self.vocab and words[i + 1] in self.vocab:
                prob *= self.bigram_probs[
                    self.vocab.index(words[i]), self.vocab.index(words[i + 1])
                ]
            else:
                # Handle unknown words (you can use smoothing techniques like add-1)
                prob *= 1e-8  # A small probability for unseen bigrams
        return prob

    def generate(self, start_word="the", max_length=20):
        generated_text = [start_word]
        current_word = start_word

        for _ in range(max_length - 1):
            if current_word not in self.vocab:
                # If the word isn't in the vocabulary, end the generation
                break 
            next_word_probs = self.bigram_probs[self.vocab.index(current_word)]

            # Sample the next word based on probabilities
            next_word_idx = torch.multinomial(next_word_probs, 1).item()
            next_word = self.vocab[next_word_idx]
            generated_text.append(next_word)
            current_word = next_word  # Update the current word for the next step

        return " ".join(generated_text)

# Create an instance of the BigramModel
model = BigramModel(vocab, bigram_probs)
```

This code defines a BigramModel class.  Here's what it does:

- **init**: Initializes the model with the vocabulary and bigram probabilities.
- **calculate_probability**: Takes a sentence and calculates the probability of it occurring under the bigram model. It handles unseen bigrams (words that don't appear together in the training data) by assigning them a very low probability.
- **generate**: Generates new text based on the model. Starting from a given word (or the word "the" by default), it samples the next word based on the bigram probabilities, and continues until it reaches the maximum length or encounters a word not in the vocabulary.

Here is how we can generate the output:

model.generate()

We can also use this model to generate the probability of a sentence.


```python
sentence1 = "the cat sat on the mat"
sentence2 = "the cat jumped over the moon"

prob1 = model.calculate_probability(sentence1)
prob2 = model.calculate_probability(sentence2)

print(f"Probability of '{sentence1}': {prob1}")
print(f"Probability of '{sentence2}': {prob2}")
```

    Probability of 'the cat sat on the mat': 9.656807616385803e-20
    Probability of 'the cat jumped over the moon': 5.83346638030119e-20



```python
def calculate_perplexity(model, dataset):
    total_log_prob = 0
    num_tokens = 0
    for tokens in dataset["text"]:
        sentence_log_prob = 0
        for i in range(len(tokens) - 1):
            if tokens[i] in model.vocab and tokens[i + 1] in model.vocab:
                sentence_log_prob += torch.log(
                    model.bigram_probs[
                        model.vocab.index(tokens[i]), model.vocab.index(tokens[i + 1])
                    ]
                )
            else:
                # Handle unknown words (using add-1 smoothing)
                sentence_log_prob += torch.log(torch.tensor(1e-8))
        total_log_prob += sentence_log_prob
        num_tokens += len(tokens) - 1  # Subtract 1 for the last token in each sentence
    perplexity = torch.exp(-total_log_prob / num_tokens)
    return perplexity
```


```python
perplexity = calculate_perplexity(model, tokenized_dataset["validation"])
print(f"Perplexity on the validation set: {perplexity:.2f}")
```

    Perplexity on the validation set: 9393.72


Perplexity of the model is quite high. Let's see if we can improve using RNN

### Limitations & Enhancements
While our bigram model demonstrates the concept, it has limitations due to its simplicity. Real-word text generation often requires more sophisticated models like Recurrent Neural Networks (RNNs) or Transformers. However the bi-gram model serves as a foundational stepping stone for understanding the underlying principles of text generation

In the next section, we will delve into more advanced techniques and explore how to build upon this basic model to create more sophisticated text generation systems

## Recurrent Neural network

Imagine reading a book. You don't start from scratch with each word; you carry the context of previous sentences in your mind. RNNs emulate this behavior by maintaining a hidden state that evolves as it processes each word in a sequence. This hidden state acts as a memory, encoding information from previous time steps, allowing the model to make predictions based on both the current input and accumulated context.


### A simple RNN structure

At its core, an RNN consists of a repeating unit (cell) that takes two inputs: the current current word and the previous hidden state. It produces two outputs: an updated hidden state and a prediction for the next word. This structure allows the RNN to process sequences of aribtrary length, making it suitable for text generation


```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output, hidden = self.rnn(embedded, hidden)
        output = self.linear(output)
        return output, hidden
```

### Training and text generation

Training an RNN involves feeding it sequences of text and adjusting its parameters to minimize the difference between its predictions and the actual next words. Once trained, we can generate text by providing a starting word and iteratively sampling from the model's output distribution.

### Attention Mechanism: Focus where it matters

A crucial enhancement to RNNs is the attention mechanism. In text generation, not all parts of the input sequence are equally important for predicting the next word. Attention allows the model to focus on relevant parts of the input while making predictions. It's like shining a spotlight on specific words or phrases that are most informative for the current context.

Huggingface models need a config object to instantiate the parameters of the model


```python
class AttentionConfig(PretrainedConfig):
    model_type = "custom_attention"
    def __init__(
        self,
        vocab_size=50257,
        hidden_size=124,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

```

Here we define our RNN model with attention. Attentions works by allowing a model to focus on different parts of its input based on the relevance of each part to the task at hand. In essence, it dynamically weights the input elements to emphasize the most important ones for the current context.


```python
class Attention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim):
        super().__init__()
        self.scale = 1./math.sqrt(query_dim)

    def forward(self, query, keys, values):
        #query = query.unsqueeze(1)
        keys = keys.transpose(1,2)
        attention = torch.bmm(query, keys)
        attention = F.softmax(attention.mul_(self.scale), dim=2)
        weighted_values = torch.bmm(attention, values).squeeze(1)
        return attention, weighted_values
```

**Forward Pass**

This defines the core functionality of the attention module, how it processes input during model's forward pass. It takes three input tensors:

- `query` : Query vector ( what model is looking for )
- `keys` : A set of key vectors ( what model can attend to )
- `values` : The values associated with each key. ( what model will retrieve )

Here is how the forward method works:

- **Transpose keys**: The keys matrix is transposed to prepare it for matrix multiplication.
- **Calculate attention scores**: The query and transposed keys are multiplied using batch matrix multiplication (torch.bmm). This produces a matrix where each element represents the similarity (or "attention") between a query and a key.
- Scale and normalize: The attention scores are multiplied by the scaling factor and then normalized using the softmax function. This ensures that the attention scores for each query sum to 1.
- **Weighted values**: The normalized attention scores are used to compute a weighted sum of the value vectors. This produces a new representation of the input sequence, where each element is a weighted combination of the original values, with weights determined by the attention mechanism.

**<TODO: Add diagram>**

Here is the model definition for RNN with self attention mechanism.


```python

class RNNWithAttention(PreTrainedModel):
    config_class = AttentionConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rnn = nn.RNN(config.hidden_size, config.hidden_size, batch_first=True)
        self.attention = Attention(config.hidden_size, config.hidden_size, config.hidden_size)
        self.linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):

        batch_size, seq_length = input_ids.shape
        embedded = self.embedding(input_ids)

        rnn_output, _ = self.rnn(embedded)
        outputs= []

        attention=None
        for t in range(seq_length):
            # get current hidden state
            hidden = rnn_output[:, t,:].unsqueeze(1) # [Bx1xH]

            # apply attention
            attention, weighted_value = self.attention(
                    hidden,
                    rnn_output[:,:t+1,:],
                    rnn_output[:,:t+1,:]
            )

            # generate output
            output = self.linear(weighted_value)
            outputs.append(output)

        logits = torch.stack(outputs,dim=1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[...,:-1,:].contiguous()
            shift_labels = labels[...,1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            attentions=attention
        )

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        return {
            "input_ids": input_ids
        }

    @staticmethod
    def _reorder_cache(past, beam_idx):
        return past
        
```

 RNN-based language models generate text by first encoding an input sequence into a series of hidden states using RNN encoder. Attention mechanism then calculates the attention scores for each hidden state, indicating their importance for predicting the next word


These attention scores are then used to create a weighted context vector, representing a summary of the most relevant information from the input sequence. This context vector, along with the previous predicted word, is fed into an RNN decoder, which produces a probability dsistribution over the vocabulary. The model then selects the next word either by sampling from this distribution or choosing the most likely word, repeating this process to generate the entire text sequence.

### Training

To train our model on the intricacies of language, we'll leverage the powerful Hugging Face Trainer API. We'll use a publicly available dataset containing wikipedia articles. This is usually a dump of all the articles made on a specific date. Our goal is to learn the language structure with RNN


```python
# Load and preprocess the dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
```

To tokenize, we will use `Huggingface Tokenizers`. This knows how to parse the raw text and convert it into tokens.


```python
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)


tokenized_datasets = (dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
                     )
```

This is huggingface specific. We need to create config for each model to train. This config contains model parameters to be used for initialization.


```python
# Create the model and configure training
config = AttentionConfig()
model = RNNWithAttention(config)
```


```python
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=10,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=100,
    logging_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    #gradient_checkpointing=True,
    fp16=True,
    learning_rate=1e-2,
    optim="adafactor"
    
)
```


```python
# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

```


```python
# Train the model
trainer.train()
```



    <div>

      <progress value='2870' max='2870' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [2870/2870 07:41, Epoch 10/10]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>100</td>
      <td>5.046100</td>
    </tr>
    <tr>
      <td>200</td>
      <td>5.247000</td>
    </tr>
    <tr>
      <td>300</td>
      <td>5.169800</td>
    </tr>
    <tr>
      <td>400</td>
      <td>4.918200</td>
    </tr>
    <tr>
      <td>500</td>
      <td>5.003900</td>
    </tr>
    <tr>
      <td>600</td>
      <td>4.905400</td>
    </tr>
    <tr>
      <td>700</td>
      <td>4.645400</td>
    </tr>
    <tr>
      <td>800</td>
      <td>4.758800</td>
    </tr>
    <tr>
      <td>900</td>
      <td>4.631000</td>
    </tr>
    <tr>
      <td>1000</td>
      <td>4.449500</td>
    </tr>
    <tr>
      <td>1100</td>
      <td>4.537300</td>
    </tr>
    <tr>
      <td>1200</td>
      <td>4.368400</td>
    </tr>
    <tr>
      <td>1300</td>
      <td>4.279100</td>
    </tr>
    <tr>
      <td>1400</td>
      <td>4.356300</td>
    </tr>
    <tr>
      <td>1500</td>
      <td>4.164800</td>
    </tr>
    <tr>
      <td>1600</td>
      <td>4.120800</td>
    </tr>
    <tr>
      <td>1700</td>
      <td>4.198600</td>
    </tr>
    <tr>
      <td>1800</td>
      <td>3.992700</td>
    </tr>
    <tr>
      <td>1900</td>
      <td>3.997100</td>
    </tr>
    <tr>
      <td>2000</td>
      <td>4.038100</td>
    </tr>
    <tr>
      <td>2100</td>
      <td>3.833600</td>
    </tr>
    <tr>
      <td>2200</td>
      <td>3.869300</td>
    </tr>
    <tr>
      <td>2300</td>
      <td>3.901300</td>
    </tr>
    <tr>
      <td>2400</td>
      <td>3.720400</td>
    </tr>
    <tr>
      <td>2500</td>
      <td>3.758200</td>
    </tr>
    <tr>
      <td>2600</td>
      <td>3.758200</td>
    </tr>
    <tr>
      <td>2700</td>
      <td>3.648400</td>
    </tr>
    <tr>
      <td>2800</td>
      <td>3.653100</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=2870, training_loss=4.304410098404834, metrics={'train_runtime': 462.0655, 'train_samples_per_second': 794.649, 'train_steps_per_second': 6.211, 'total_flos': 1780264886400000.0, 'train_loss': 4.304410098404834, 'epoch': 10.0})




```python
trainer.save_model('bin/model_128_006')
```


```python
model = RNNWithAttention.from_pretrained('bin/model_128_006')
```

### Generation

We will now use `huggingface pipeline` api to generate the text.


```python
from transformers import AutoTokenizer, pipeline

```


```python
generated_texts = model.manual_generate(
    "The quick brown fox", tokenizer, max_length=100
)
print(generated_texts)
```

    The quick brown fox, the long @-@ brick terrace, Ders conceded varsity coach. He did this lives, as well asigned and roll, microorganisms, specifically females leave the term F @-@ for death in a Comedy Series, in the United States's appointment at Artistsers, John photography Blues Lane – a daughter and Co @-@ nation tropes often referred to as the mainchandised Princess aviation, and had to purchase a heavier solution will to rarity with



```python
generate_text = pipeline("text-generation", model=model, tokenizer=tokenizer)
```

    The model 'RNNWithAttention' is not supported for text-generation. Supported models are ['BartForCausalLM', 'BertLMHeadModel', 'BertGenerationDecoder', 'BigBirdForCausalLM', 'BigBirdPegasusForCausalLM', 'BioGptForCausalLM', 'BlenderbotForCausalLM', 'BlenderbotSmallForCausalLM', 'BloomForCausalLM', 'CamembertForCausalLM', 'LlamaForCausalLM', 'CodeGenForCausalLM', 'CohereForCausalLM', 'CpmAntForCausalLM', 'CTRLLMHeadModel', 'Data2VecTextForCausalLM', 'DbrxForCausalLM', 'ElectraForCausalLM', 'ErnieForCausalLM', 'FalconForCausalLM', 'FuyuForCausalLM', 'GemmaForCausalLM', 'GitForCausalLM', 'GPT2LMHeadModel', 'GPT2LMHeadModel', 'GPTBigCodeForCausalLM', 'GPTNeoForCausalLM', 'GPTNeoXForCausalLM', 'GPTNeoXJapaneseForCausalLM', 'GPTJForCausalLM', 'JambaForCausalLM', 'JetMoeForCausalLM', 'LlamaForCausalLM', 'MambaForCausalLM', 'MarianForCausalLM', 'MBartForCausalLM', 'MegaForCausalLM', 'MegatronBertForCausalLM', 'MistralForCausalLM', 'MixtralForCausalLM', 'MptForCausalLM', 'MusicgenForCausalLM', 'MusicgenMelodyForCausalLM', 'MvpForCausalLM', 'OlmoForCausalLM', 'OpenLlamaForCausalLM', 'OpenAIGPTLMHeadModel', 'OPTForCausalLM', 'PegasusForCausalLM', 'PersimmonForCausalLM', 'PhiForCausalLM', 'Phi3ForCausalLM', 'PLBartForCausalLM', 'ProphetNetForCausalLM', 'QDQBertLMHeadModel', 'Qwen2ForCausalLM', 'Qwen2MoeForCausalLM', 'RecurrentGemmaForCausalLM', 'ReformerModelWithLMHead', 'RemBertForCausalLM', 'RobertaForCausalLM', 'RobertaPreLayerNormForCausalLM', 'RoCBertForCausalLM', 'RoFormerForCausalLM', 'RwkvForCausalLM', 'Speech2Text2ForCausalLM', 'StableLmForCausalLM', 'Starcoder2ForCausalLM', 'TransfoXLLMHeadModel', 'TrOCRForCausalLM', 'WhisperForCausalLM', 'XGLMForCausalLM', 'XLMWithLMHeadModel', 'XLMProphetNetForCausalLM', 'XLMRobertaForCausalLM', 'XLMRobertaXLForCausalLM', 'XLNetLMHeadModel', 'XmodForCausalLM'].



```python
# generate text
result = generate_text("The quick brown fox", max_length =50, do_sample=True, top_k =50, temperature=0.7)
```


```python
print(result[0]['generated_text'])
```

    The quick brown foxes, although the wild of the prerogative is usually harmful to the consumption of " Like the song " the " best @-@ pop song ", and " Ode to Psyche ", which ", she has



```python
eval_f = trainer.evaluate()
```


```python
eval_f
```




    {'eval_loss': 6.330234527587891,
     'eval_runtime': 1.4481,
     'eval_samples_per_second': 2596.464,
     'eval_steps_per_second': 20.716,
     'epoch': 10.0}




```python
import math
```


```python
perplexity_f = math.exp(eval_f['eval_loss'])
```


```python
perplexity_f
```




    561.2882159892837



We significantly improved the perplexity of the model. Let's see if we can improve further with transformer architecture

## Transformer: Architecture

In earlier section, we saw simple RNN model struggling to learn the language but attention model gave a high boost to the language model. There are few drawbacks of RNN with self attention as mentioned below:

- **Computation time**
- **Only one self attention head**
- **Vanishing gradient**

Transformer architecture is built on RNN structures but without above flaws. It does this very cleverly by following a unique mechanism to find the recurrence relation. 

![](https://www.it-jim.com/wp-content/uploads/2023/06/attention_research_1-727x1024-1.webp)

Big Large Language models ( LLAMA 3, GPT , Mistral ) etc. are transformer decoder models. They are based on 'classical transformer' which had two blocks: **encoder** on the left and **decoder** on the right. This encoder-decoder architecture is rather arbitrary, and that is not how must transformers model work today. Typically, a moderl transforemr is entier an encoder (Bert Family) or a decoder (GPT family). So, a GPT architecture looks like this:

![](https://www.it-jim.com/wp-content/uploads/2023/06/Architecture-of-the-GPT-2-Transformer-model-768x633.webp)

The only difference between **encoder** and **decoder** is that latter is causal. i.e. it cannot go back in time. By 'time' here, we mean the position t=1..T of the token(word) in the sequence. Only decoders can be used for text generation. GPT models are pretty much your garden variety of transformer decoders, and different GPT versions differ pretty much only in size, minor details, and the dataset+training regime. If you understand how GPT works then you undertand all big large language model works. For our purposes, we drew our own simplified GPT-2 digaram with explit tensor dimensions. DOn't worry if it confuses you. We'll explain it step by step in a moment.

![](https://www.it-jim.com/wp-content/uploads/2023/06/GPT2-768x452.webp)

Let's diive into how transformers work, step by step, using a diagram above

**The components of a Transformer**

1. **Input Tokens (BxT):**

These are the words or pieces of text that we feed into the transformer. In our example, let's say we have a sentence with three words: "I love reading." Each word is represented by a number (called a token) for the model to understand.
In the diagram, we see tokens: 464, 23878, and 16599.

2. **Embedder:**

The embedder transforms these tokens into a numerical format called embeddings. Think of it as translating each word into a unique code that captures its meaning in a way the model can work with.
The embeddings are shown as EMB_IN1, EMB_IN2, and EMB_IN3.

3. **Transformer Blocks:**

These are the heart of the transformer model. They process the embeddings, allowing the model to understand the context of each word in relation to others. This is crucial because the meaning of a word can change depending on its context.
The transformer blocks modify the embeddings, which are now called EMB_OUT1, EMB_OUT2, and EMB_OUT3.

4. **Generation Head:**

After processing through the transformer blocks, the embeddings are ready to be transformed back into tokens. The generation head takes these embeddings and predicts the most likely next word or sequence of words.
The predictions are called logits, represented as LOGITS1, LOGITS2, and LOGITS3. These logits are then converted back to readable text.

5. **Logits (BxTxV):**

Logits are the raw predictions made by the model for each word in the vocabulary (V = 50257, the total number of possible tokens). The model uses these logits to determine which words are most likely to come next in the sequence.

### The magic of context

One of the most impressive aspects of transformers is their ability to understand context. For example, the word "reading" can mean different things in different sentences. In "I love reading books," it refers to the act of reading. In "Reading is a city in England," it refers to a place. Transformers can differentiate these meanings based on the surrounding words, making them incredibly powerful for natural language processing.

### GPT-2 Model definition

Let's create a GPT-2 decoder only model from scratch. We will then train on wikipedia articles to learn about the language structure. As stated above, we're not changing the definition here but just the number of layers

![](https://miro.medium.com/v2/resize:fit:720/format:webp/0*77memcl1VYIdpE8f.png)


```python
# Create a custom GPT-2 configuration
custom_config = GPT2Config(
    vocab_size=50257,
    n_positions=128,
    n_ctx=256,
    n_embd=768,
    n_layer=4,      # Number of transformer layers
    n_head=4,
)

# Initialize the custom GPT-2 model with the custom configuration
model = AutoModelForCausalLM.from_config(custom_config)
```

### Training

Initialize the trainer for desired number of epochs and batch size. We can also monitor the training using wandb ( not shown here but look at the appendex to know more about callbacks ).


```python
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)
```


```python
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(mlm=False, tokenizer=tokenizer)
)
```


```python
trainer.train()
```



    <div>

      <progress value='861' max='861' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [861/861 10:16, Epoch 3/3]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>7.790100</td>
    </tr>
  </tbody>
</table><p>





    TrainOutput(global_step=861, training_loss=7.151387036331301, metrics={'train_runtime': 617.1376, 'train_samples_per_second': 178.492, 'train_steps_per_second': 1.395, 'total_flos': 2398616836374528.0, 'train_loss': 7.151387036331301, 'epoch': 3.0})




```python
inputs = tokenizer("the cat", return_tensors="pt")
outputs = model.generate(**inputs, num_beams=4, do_sample=True)
print(tokenizer.batch_decode(outputs,skip_special_tokens=True))
```

    Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.


    ["the cat's series's first time, he was released in the first time. He was"]



```python
eval_f = trainer.evaluate()
```



<div>

  <progress value='30' max='30' style='width:300px; height:20px; vertical-align: middle;'></progress>
  [30/30 00:04]
</div>




```python
perplexity_f = math.exp(eval_f['eval_loss'])
```

perplexity_f

# Conclusion

Write conclusion here..


```python

```
