# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

inputStr="for(int i; 1==1 ;i++)print(\"sss\")";

tokenizer = AutoTokenizer.from_pretrained("D:\\ProgramData\\torchHome\\mymodel\\codereviewer")
model = AutoModelForSeq2SeqLM.from_pretrained("D:\\ProgramData\\torchHome\\mymodel\\codereviewer")

completion = model.generate(**tokenizer(inputStr, return_tensors="pt"))
result = tokenizer.decode(completion[0])
print("让我听听你高见："+result)