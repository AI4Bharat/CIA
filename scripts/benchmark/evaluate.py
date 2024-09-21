import os
import json
import argparse
import pandas as pd
import re

import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from transformers import AutoTokenizer

lang_map = {
    "bn": "bengali",
    "fr": "french",
    "de": "german",
    "hi": "hindi",
    "te": "telugu",
    "ur": "urdu"
}


def get_chat_template(type_):
    if type_ == "llama":
        return "{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n"
    elif type_ == "gemma":
        return "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"
    elif type_ == "sarvam":
        return "{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content'] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}\n        {{- raise_exception('After the optional system message, conversation roles must alternate user/assistant/user/assistant/...') }}\n    {%- endif %}\n    {%- if message['role'] == 'user' %}\n        {%- if loop.first and system_message is defined %}\n            {{- ' [INST] ' + system_message + '\\n\\n' + message['content'] + ' [/INST]' }}\n        {%- else %}\n            {{- ' [INST] ' + message['content'] + ' [/INST]' }}\n        {%- endif %}\n    {%- elif message['role'] == 'assistant' %}\n        {{- ' ' + message['content'] + eos_token}}\n    {%- else %}\n        {{- raise_exception('Only user and assistant roles are supported, with the exception of an initial optional system message!') }}\n    {%- endif %}\n{%- endfor %}\n"
        

def get_predictions(n_gpus, model, type_, lang):
    tokenizer = AutoTokenizer.from_pretrained(model)
    tokenizer.chat_template = get_chat_template(type_)
    
    
    print(f"Loading dataset for {lang}")
    dataset = load_dataset("ai4bharat/cia-bench", lang)
    prompts = [tokenizer.apply_chat_template([row['messages'][0]], tokenize=False, add_generation_prompt=True) for row in dataset['test']]
    ground_truth_scores = [row['orig_score'] for row in dataset['test']]
    ground_truth_feedbacks = [row['orig_feedback'] for row in dataset['test']]


    print(f"Loading model from: {model}")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        tensor_parallel_size=n_gpus
    )

    sampling_params = SamplingParams(
        temperature=0.7, 
        top_p=0.95, 
        max_tokens=256,
        repetition_penalty=1.03
    )

    outputs = llm.generate(prompts, sampling_params)

    results = []
    for prompt, output, ground_truth_score, ground_truth_feedback in zip(prompts, outputs, ground_truth_scores, ground_truth_feedbacks):
        generated_text = output.outputs[0].text
        results.append({
            'model_path': model,
            'prompt': prompt,
            'generated_text': generated_text,
            'true_score': ground_truth_score,
            'feedback': ground_truth_feedback
        })

    predictions = pd.DataFrame(results)
    
    return predictions

def stop_after_first_result(text):
    # Regular expression to match "[RESULT] x" where x is any number
    pattern = r"\[RESULT\] \d+"
    # Search for the pattern in the text
    match = re.search(pattern, text)
    try:
        if match:
            # Get the position of the end of the first match
            end_position = match.end()
            # Return the text up to the end of the first match
            return int(text[:end_position][-1])
        else:
            # If no match is found, return the original text
            return int(text[-1])
    except:
        return -1
    
def accuracy(predictions):
    ground_truth = list(predictions['true_score'])
    generated_label = list(predictions['generated_text'].apply(stop_after_first_result))

    if len(ground_truth) != len(generated_label):
        raise ValueError("The lengths of generated_label and ground_truth must be the same.")
    
    correct_predictions = sum(1 for gen, true in zip(generated_label, ground_truth) if gen == true)
    total_predictions = len(ground_truth)
    
    return correct_predictions / total_predictions

def calculate_approx_accuracy(predictions):
    ground_truth = list(predictions['true_score'])
    generated_label = list(predictions['generated_text'].apply(stop_after_first_result))

    if len(ground_truth) != len(generated_label):
        raise ValueError("The lengths of generated_label and ground_truth must be the same.")
    
    correct_predictions = sum(1 for gen, true in zip(generated_label, ground_truth) if (gen == true) or (gen == true - 1 or gen == true + 1))
    total_predictions = len(ground_truth)
    
    accuracy = correct_predictions / total_predictions
    return accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using VLLM on multiple GPUs and save results to CSV")
    parser.add_argument("--model", type=str, required=True, help="name of the model")
    parser.add_argument("--lang", type=str, required=True, help="language to evaluate")
    parser.add_argument("--type", type=str, required=True, help="type of model", choices=["llama", "gemma", "sarvam"])
    parser.add_argument("--output_path", type=str, default="/projects/data/llmteam/CIA/artifacts")
    return parser.parse_args()

def main(args):
    os.makedirs(args.output_path, exist_ok=True)
    model_file_name = args.model.split("/")[-1]
    
    available_gpus = torch.cuda.device_count()
    predictions = get_predictions(available_gpus, args.model, args.type, args.lang)
    
    acc_ = accuracy(predictions)
    approx_acc = calculate_approx_accuracy(predictions)
    
    os.makedirs(f"{args.output_path}/results/", exist_ok=True)
    with open(f"{args.output_path}/results/{args.lang}-{model_file_name}-result.json", 'w') as f:
    # with open("results.json", 'w') as f:
        results = {
            "model": args.model,
            "accuracy": acc_,
            "approx_accuracy": approx_acc
        }
        print(results)
        json.dump(results, f)
    
    os.makedirs(f"{args.output_path}/predictions/", exist_ok=True)
    predictions.to_csv(f"{args.output_path}/predictions/{args.lang}-{model_file_name}.tsv", sep='\t', index=False)
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)