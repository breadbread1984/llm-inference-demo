#!/usr/bin/python3

import json
from absl import flags, app
from torch import device
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate

class ChatGLM3(LLM):
  tokenizer: AutoTokenizer = None
  model: AutoModelForCausalLM = None
  def __init__(self, dev = 'cuda', use_history = True):
    assert dev in {'cpu', 'cuda'}
    super().__init__()
    self.tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = AutoModelForCausalLM.from_pretrained('THUDM/chatglm3-6b', trust_remote_code = True)
    self.model = self.model.to(device(dev))
    self.model.eval()
    self.use_history = use_history
    self.history = list()
  def _call(self, prompt, stop = None, run_manager = None, **kwargs):
    if not self.use_history:
      self.history = list()
    response, self.history = self.model.chat(self.tokenizer, prompt, history = self.history)
    if len(self.history) > 10: self.history.pop(0)
    return response
  @property
  def _llm_type(self):
    return "ChatGLM3-6B"

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  llm = ChatGLM3(FLAGS.device)
  prompt = PromptTemplate.from_template("{prompt}")
  chain = prompt | llm
  while True:
    query = input('>')
    output = chain.invoke({"prompt": query})
    print(outputs)

if __name__ == "__main__":
  add_options()
  app.run(main)

