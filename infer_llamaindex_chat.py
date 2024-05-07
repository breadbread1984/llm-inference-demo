#!/usr/bin/python3

import json
from absl import flags, app
from huggingface_hub import login
from transformers import AutoTokenizer
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.llms.huggingface import HuggingFaceLLM

FLAGS = flags.FLAGS

def add_options():
  pass

def main(unused_argv):
  login('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')
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', trust_remote_code = True)
  def messages_to_prompt(messages):
    messages = [{'role': message.role, 'content': message.content} for message in messages]
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    return prompt
  def completion_to_prompt(completion):
    messages = [{'role': 'user', 'content': completion}]
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    return prompt
  llm = HuggingFaceLLM(
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct',
    tokenizer_name = 'meta-llama/Meta-Llama-3-8B-Instruct',
    generate_kwargs = {'temperature': 0.6, 'top_p': 0.9, 'do_sample': True, 'use_cache': True},
    messages_to_prompt = messages_to_prompt,
    completion_to_prompt = completion_to_prompt,
    device_map = "auto")
  history = list()
  while True:
    query = input('>')
    history.append(ChatMessage.from_str(content = query, role = MessageRole.USER))
    response = llm.chat(history)
    print(response.message.content)
    history.append(response.message)
    if len(history) > 5: history.pop(0)

if __name__ == "__main__":
  add_options()
  app.run(main)

