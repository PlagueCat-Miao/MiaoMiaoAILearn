unset CUDA_VISIBLE_DEVICES

python -m paddle.distributed.launch --gpus 0 paddleCodegen\run_clm.py ^
   --model_name_or_path Salesforce/codegen-350M-multi ^
   --block_size 128 ^
   --output_dir  %PPNLP_HOME%\models\miaomiao\output ^
   --train_file trainData\train3.json ^
   --validation_file trainData\test.json ^
   --num_train_epochs 2 ^
   --logging_steps 1 ^
   --save_steps 10 ^
   --train_batch_size 1 ^
   --eval_batch_size 1 ^
   --learning_rate 1e-4 ^
   --warmup_proportion 0.1 ^
   --device gpu
