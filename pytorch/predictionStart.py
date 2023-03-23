import re
import os
import argparse
import random
import time
import distutils.util
from pprint import pprint
from functools import partial
import numpy as np
from itertools import chain
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

# https://huggingface.co/docs/transformers/main/en/model_doc/codegen
# modelStr = "microsoft/CodeGPT-small-java-adaptedGPT2";
os.environ['TORCH_HOME']='D:\\ProgramData\\torchHome';
modelStr ="microsoft/CodeGPT-small-java-adaptedGPT2";
# 预测输入内容
with open("predictionData\P4.txt", "r") as f:  # 打开测试文件
    text = f.read()
text = "define helloWorld funciton"
print("输入:")
print("text:" + text + "\nmodel:" + modelStr)

# 执行预测验证
# Init tokenizer model
tokenizer = AutoTokenizer.from_pretrained(modelStr)
model = AutoModelForCausalLM.from_pretrained(modelStr)

completion = model.generate(**tokenizer(text, return_tensors="pt"))
print("输出:")
print(tokenizer.decode(completion[0]))