import re
import paddle
import os
import argparse
import random
import time
import distutils.util
from pprint import pprint
from functools import partial
import numpy as np
from itertools import chain
from datasets import load_dataset
import math
from visualdl import LogWriter
import paddle
import paddle.nn as nn
from paddle.io import BatchSampler, DistributedBatchSampler, DataLoader
from paddlenlp.transformers import CodeGenForCausalLM
from paddlenlp.transformers import CodeGenTokenizer
from paddlenlp.transformers import LinearDecayWithWarmup
from paddlenlp.utils.log import logger
from paddlenlp.data import DataCollatorWithPadding
from paddle.metric import Accuracy

#使用模型地址
model_dir = "D:\ProgramData\.paddlenlp\models\miaomiao\output"
#预测输入内容
with open("predictionData\P4.txt", "r") as f:  # 打开测试文件
    text = f.read()
print("输入:")        
print("text:" + text + "\nmodel_dir:" + model_dir)

#执行预测验证
# Init tokenizer
tokenizer = CodeGenTokenizer.from_pretrained(model_dir)
# Init model
model = CodeGenForCausalLM.from_pretrained(model_dir)
# Generate
tokenized = tokenizer(text,
                      truncation=True,
                      max_length=1024,
                      return_tensors='pd')
preds, _ = model.generate(input_ids=tokenized['input_ids'],
                          max_length=128)
print("输出:")
print(tokenizer.decode(preds[0], skip_special_tokens=True, clean_up_tokenization_spaces=False))


# prompt = "# this function prints hello world"
# inputs = tokenizer([prompt])
# inputs = {k: paddle.to_tensor(v) for (k, v) in inputs.items()}
# # Generate
# output, score = model.generate(inputs['input_ids'],
#                                max_length=128,
#                                decode_strategy='greedy_search')
# #Decode the result
# print(
#     tokenizer.decode(output[0],
#                      truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"],
#                      skip_special_tokens=True,
#                      spaces_between_special_tokens=False))