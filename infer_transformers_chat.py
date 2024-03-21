#!/usr/bin/python3

import json
from absl import flags, app
from copy import deepcopy
from huggingface_hub import login
from torch import device
from transformers import AutoTokenizer, AutoModelForCausalLM, \
        LogitsProcessorList, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def process_response(output, history):
  content = ""
  history = deepcopy(history)
  for response in output.split("<|assistant|>"):
    if "\n" in response:
      metadata, content = response.split("\n", maxsplit=1)
    else:
      metadata, content = "", response
    if not metadata.strip():
      content = content.strip()
      history.append({"role": "assistant", "metadata": metadata, "content": content})
      content = content.replace("[[训练时间]]", "2023年")
    else:
      history.append({"role": "assistant", "metadata": metadata, "content": content})
      if history[0]["role"] == "system" and "tools" in history[0]:
        content = "\n".join(content.split("\n")[1:-1])
        def tool_call(**kwargs):
          return kwargs
        parameters = eval(content)
        content = {"name": metadata.strip(), "parameters": parameters}
      else:
        content = {"name": metadata.strip(), "content": content}
  return content, history

def main(unused_argv):
  login(token = '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')
  tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
  llm = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
  llm = llm.to(device(FLAGS.device))
  history = list()
  logits_processor = LogitsProcessorList()
  logits_processor.append(TemperatureLogitsWarper(0.8))
  logits_processor.append(TopPLogitsWarper(0.8))
  print('Ctrl+C to exit...\n')
  while True:
    query = input('>')
    inputs = tokenizer.build_chat_input(query, history = history, role = 'user')
    inputs = inputs.to(device(FLAGS.device))
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"),
                    tokenizer.get_command("<|observation|>")]
    outputs = llm.generate(**inputs, logits_processor = logits_processor, do_sample = True, max_length = 8192, eos_token_id=eos_token_id)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
    response = tokenizer.decode(outputs)
    history.append({'role': 'user', 'content': query})
    response, history = process_response(response, history)
    if len(history) > 10: history.pop(0)
    print(response)

if __name__ == "__main__":
  add_options()
  app.run(main)

