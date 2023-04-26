from transformers import AutoTokenizer
from transformers import TextDataset,DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM


#modelStr = "microsoft/CodeGPT-small-java-adaptedGPT2";
modelStr = "D:/ProgramData/torchHome/mymodel/CodeGPT-small-java-adaptedGPT2/";

train_path = 'train_dataset.txt'
test_path = 'test_dataset.txt'

print("加载模型开始" + modelStr)
tokenizer = AutoTokenizer.from_pretrained(modelStr)
model = AutoModelForCausalLM.from_pretrained(modelStr)

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=train_path,
          block_size=128)

    test_dataset = TextDataset(
          tokenizer=tokenizer,
          file_path=test_path,
          block_size=128)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset,test_dataset,data_collator

print("加载数据集开始")
train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)



training_args = TrainingArguments(
    output_dir="./gpt2-gerchef", #The output directory
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
