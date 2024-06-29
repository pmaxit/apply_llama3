import json
from langchain_core.runnables import RunnableGenerator
from typing import Iterable
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain.prompts.chat import ChatPromptValue

def extract_json_objects(text):
    """
    Extracts JSON objects from a string.
    
    Args:
    text (str): The string containing JSON objects.
    
    Returns:
    list: A list of extracted JSON objects.
    """
    json_objects = []
    start_positions = []
    brackets_count = 0
    
    # Iterate over each character in the string
    for i, char in enumerate(text):
        if char == '{':
            if brackets_count == 0:
                start_positions.append(i)
            brackets_count += 1
        elif char == '}':
            brackets_count -= 1
            if brackets_count == 0 and start_positions:
                start_pos = start_positions.pop()
                try:
                    # Extract the substring and parse it as JSON
                    print('text ' ,text[start_pos:i+1])
                    json_obj = json.loads(text[start_pos:i+1])
                    print(json_obj)
                    json_objects.append(json_obj)
                except json.JSONDecodeError:
                    pass  # Ignore parsing errors and continue
    
    return json.dumps(json_objects[0])




def streaming_parse(chunks: Iterable[AIMessageChunk]) -> Iterable[str]:

    for chunk in chunks:
        yield chunk.content

streaming_parse = RunnableGenerator(streaming_parse)

def extract_json(ai_message: AIMessage) -> str:
    """Parse the AI message."""
    print(ai_message.content)
    return extract_json_objects(ai_message.content)

def debug_llm(chat_message: HumanMessage):
    print('..................'*2)
    content = chat_message.messages[0].content
    print(content)
    print('..................'*2)
    return ChatPromptValue(messages=[HumanMessage(content=content)])


def collate_inputs_labels(examples):
    # create RNN like input
    #max_len = max(len(ex) for ex in examples['input_ids'])

    all_input_ids = []
    all_labels = []

    for input_ids in examples['input_ids']:
        for i in range(1, len(input_ids)):
            all_input_ids.append(input_ids[:i])
            all_labels.append(input_ids[i])
    
        if len(input_ids) > 0:
            all_input_ids.append(input_ids)
            all_labels.append(tokenizer.eos_token_id)

    
    return {
        'input_ids_new': all_input_ids,
        'labels_new': all_labels
    }
    