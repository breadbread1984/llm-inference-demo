#!/usr/bin/python3

import json
from absl import app, flags
from langchain.llms import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('model', default = 'google/gemma-7b', help = 'model name')
  flags.DEFINE_string('prompt', default = None, help = 'path to file containing prompts, line break is replaced by \\n')
  flags.DEFINE_string('output', default = 'output.json', help = 'path to output file')

def main(unused_argv):
  llm = HuggingFaceEndpoint(endpoint_url = "https://api-inference.huggingface.co/models/" + FLAGS.model,
                            huggingfacehub_api_token = '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',
                            task = 'text-generation')
  prompt = PromptTemplate.from_template("{prompt}")
  output_parser = StrOutputParser()
  chain = prompt | llm | output_parser
  outputs = list()
  with open(FLAGS.prompt, 'r') as f:
    for line in f.readlines():
      output = chain.invoke({"prompt":line})
      outputs.append(output)
  with open(FLAGS.output, 'w') as f:
    f.write(json.dumps(outputs))

if __name__ == "__main__":
  add_options()
  app.run(main)

