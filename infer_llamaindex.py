#!/usr/bin/python3

from absl import flags, app
from transformers import AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceInferenceAPI

FLAGS = flags.FLAGS

def add_options():
  pass

def main(unused_argv):
  tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct', trust_remote_code = True)
  def messages_to_prompt(message):
    messages = [{'role': message.role, 'content': message.content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    return prompt
  def completion_to_prompt(completion):
    messages = [{'role': 'user', 'content': completion}]
    prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    return prompt
  llm = HuggingFaceLLM(
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct',
    tokenizer_name = 'meta-llama/Meta-Llama-3-8B-Instruct',
    generate_kwargs = {'temperature': 0.6, 'top_p': 0.9, 'do_sample': True},
    messages_to_prompt = messages_to_prompt,
    completion_to_prompt = completion_to_prompt,
    token = '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',
    device_map = "auto")
  response = llm.chat('how is the day?')
  print(response)

if __name__ == "__main__":
  add_options()
  app.run(main)

