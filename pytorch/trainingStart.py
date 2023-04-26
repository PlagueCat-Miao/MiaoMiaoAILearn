from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset
from transformers import TextDataset
from transformers import LineByLineTextDataset

#modelStr = "microsoft/CodeGPT-small-java-adaptedGPT2";
modelPath = "D:/ProgramData/torchHome/mymodel/codegen-350M-multi/";
output_model = "./miaomiaoGpt";
train_path = './trainData/train5.txt'
test_path = './trainData/test.txt'

print("加载模型：" + modelPath)
tokenizer = AutoTokenizer.from_pretrained(modelPath)
model = AutoModelForCausalLM.from_pretrained(modelPath)
# 报错：Asking to pad but the tokenizer does not have a padding token
# 解决方案：https://stackoverflow.com/questions/70544129/transformers-asking-to-pad-but-the-tokenizer-does-not-have-a-padding-token
# 解决方案：https://blog.csdn.net/qq_51750957/article/details/128856264
if tokenizer.pad_token is None:
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #model.resize_token_embeddings(len(tokenizer))
    tokenizer.pad_token = tokenizer.eos_token

print("加载数据集："+train_path)
train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=train_path,
    block_size=400)
print("加载数据集："+test_path)
test_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path=test_path,
    block_size=128)
print("加载DataCollator")
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False # is Masked Language Modeling(mlm) or Causal language modeling(clm)
)


training_args = TrainingArguments(
    output_dir=output_model, #The output directory
    overwrite_output_dir=True, #overwrite the content of the output directory
    num_train_epochs=3, # number of training epochs
    per_device_train_batch_size=32, # batch size for training
    per_device_eval_batch_size=64,  # batch size for evaluation
    eval_steps = 400, # Number of update steps between two evaluations.
    save_steps=800, # after # steps model is saved
    warmup_steps=500,# number of warmup steps for learning rate scheduler
)
    
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
print("开始训练")
trainer.train()
trainer.save_model()
print("开始结束，保存于："+output_model)