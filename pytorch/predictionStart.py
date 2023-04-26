from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface

tokenizerPath = "D:/ProgramData/torchHome/mymodel/codegen-350M-multi/";
modelPath = "./miaomiaoGpt/"
input_path = "./predictionData/P4.txt";


def load_input(inputPath):
    with open(inputPath, "r", encoding='GBK') as f:  # 打开测试文件
        text = f.read()
        return text.split("\n");


print("开始加载模型:" + modelPath)
chef = pipeline('text-generation', model=modelPath, tokenizer=tokenizerPath)

print("开始加载预测文件:" + input_path)
inputStrList = load_input(input_path);
# inputStrList = ["@MySet"]
print("pipeline开始预测:")
for inputStr in inputStrList:
    print("开始预测:\n" + inputStr)
    results = chef(inputStr)
    result = results[0]['generated_text'];
    print("预测结果:\n" + result)

# https://huggingface.co/docs/transformers/main/en/model_doc/codegen

# 加载预测模型
# Init tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizerPath)
# Init model
model = AutoModelForCausalLM.from_pretrained(modelPath)
# 执行预测验证
print("model.generate开始预测:\n")
for inputStr in inputStrList:
    print("开始预测:\n" + inputStr)
    completion = model.generate(**tokenizer(inputStr, return_tensors="pt"))
    result = tokenizer.decode(completion[0]);
    print("预测结果:\n" + result)
