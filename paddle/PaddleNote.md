# PaddleNode学习手册

### 快速启动

#### 1. 进入paddlenlp环境

```powershell
conda activate my_paddlenlp
```
```powershell
conda activate paddle_env
```

#### 2.GPU启动训练

- 

- 【注意1】GPU加速及多GPU运行问题：

  1. 使用命令前先用 `unset CUDA_VISIBLE_DEVICES` 命令清空变量；

  2. 再使用 export CUDA_VISIBLE_DEVICES 命令设置使用哪张或者哪几张显卡；

     ```
     export CUDA_VISIBLE_DEVICES= 使用0号显卡（pmemd.cuda）
     
     export CUDA_VISIBLE_DEVICES=, 使用0，1两张显卡（mpirun -np 2 pmemd.cuda.MPI）
     ```

-  



## 附录

### 下载

- Paddle教程：[github官网](https://github.com/PaddlePaddle/PaddleNLP)、 [gitte](https://gitee.com/paddlepaddle/PaddleNLP?_from=gitee_search)、[官网](https://aistudio.baidu.com/aistudio/projectdetail/4903719)
- [Hugging Face](https://huggingface.co/)训练模型开源下载站，我们关注：
  - [CodeGPT-small-java-adaptedGPT2](https://huggingface.co/microsoft/CodeGPT-small-java-adaptedGPT2/tree/main)
  - [codegen-350M-multi](https://huggingface.co/Salesforce/codegen-350M-multi)
- Paddle使用的Salesforce/codegen-xxxx-multi的下载方式
  - 模型参数：`https://bj.bcebos.com/paddlenlp/models/community/Salesforce/codegen-xxxx-multi/model_state.pdparams`
  - 模型配置：`https://bj.bcebos.com/paddlenlp/models/community/Salesforce/codegen-xxxxx-multi/model_config.json`
  - 比如350M的[model_state.pdparams](https://bj.bcebos.com/paddlenlp/models/community/Salesforce/codegen-350M-multi/model_state.pdparams)、[model_config.json](https://bj.bcebos.com/paddlenlp/models/community/Salesforce/codegen-350M-multi/model_config.json)