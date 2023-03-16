from paddlenlp import Taskflow

prompt = "def lengthOfLongestSubstring(self, s: str) -> int:";
codegenModel = "Salesforce/codegen-350M-multi";

useModel = codegenModel;
print("输入："+ prompt)
print("模型："+ useModel)
codegen = Taskflow("code_generation", model=useModel,decode_strategy="greedy_search", repetition_penalty=1.0)
print("输出：\n")
print(codegen(prompt))


