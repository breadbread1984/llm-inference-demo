#!/usr/bin/python3

import json
from absl import app, flags
from vllm import LLM, SamplingParams

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('model', default = 'meta-llama/Llama-2-7b-hf', help = 'model name')
  flags.DEFINE_string('prompt', default = None, help = 'path to file containing prompts. line break is replaced by \\n')
  flags.DFEINE_float('top_p', default = 1, help = 'top-p')
  flags.DEFINE_float('top_k', default = -1, help = 'top-k')
  flags.DEFINE_float('temperature', default = 0, help = 'temperature or greedy sampling if set 0')
  flags.DEFINE_string('output', default = 'outputs.json', help = 'path to output file')

def main(unused_argv):
  prompts = list()
  with open(FLAGS.prompt, 'r') as f:
    for line in f.readlines():
      prompts.append(line.replace('\\n','\n'))
  llm = LLM(model)
  sampling_params = SamplingParams(temperature = FLAGS.temperature, top_p = FLAGS.top_p, top_k = FLAGS.top_k)
  outputs = llm.generate(prompts, sampling_params)
  with open(FLAGS.output, 'w') as f:
    f.write(json.dumps(outputs))

if __name__ == "__main__":
  add_options()
  app.run(main)

