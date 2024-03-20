#!/usr/bin/python3

import json
from absl import app, flags
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList, \
        TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('model', default = 'meta-llama/Llama-2-7b-hf', help = 'model name')
  flags.DEFINE_string('prompt', default = None, help = 'path to file containing prompts, line break is replaced by \\n')
  flags.DEFINE_float('top_p', default = 1, help = 'top-p')
  flags.DEFINE_float('top_k', default = -1, help = 'top-k')
  flags.DEFINE_float('temperature', default = 1, help = 'temperature')
  flags.DEFINE_boolean('sample', default = False, help = 'whether to sample output')
  flags.DEFINE_string('output', default = 'outputs.json', help = 'path to output file')

def main(unused_argv):
  login(token = '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')
  prompts = list()
  with open(FLAGS.prompt, 'r') as f:
    for line in f.readlines():
      prompts.append(line.replace('\\n','\n'))
  tokenizer = AutoTokenizer.from_pretrained(FLAGS.model)
  tokenizer.padding_side = 'left'
  tokenizer.pad_token = tokenizer.eos_token
  llm = AutoModelForCausalLM.from_pretrained(FLAGS.model)
  logits_processor = LogitsProcessorList()
  logits_processor.append(TemperatureLogitsWarper(FLAGS.temperature))
  if FLAGS.top_p != 1:
    logits_processor.append(TopPLogitsWarper(FLAGS.top_p))
  elif FLAGS.top_k != -1:
    logits_processor.append(TopKLogitsWarper(FLAGS.top_k))
  inputs = tokenizer(prompts, return_tensors = 'pt', padding = True)
  kvcache = None
  outputs = llm.generate(**inputs, logits_processor = logits_processor, do_sample = FLAGS.sample, use_cache = True, past_key_values = kvcache, return_dict_in_generate = True) # set return_dict_in_generate to get latest kvcache
  kvcache = outputs.past_key_values
  input_ids = outputs.sequences
  outputs = tokenizer.batch_decode(input_ids, skip_special_tokens = True)
  with open(FLAGS.input, 'w') as f:
    f.write(json.dumps(outputs))

if __name__ == "__main__":
  add_options()
  app.run(main)

