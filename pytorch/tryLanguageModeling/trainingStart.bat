python tryLanguageModeling/run_clm.py  ^
    --model_name_or_path Salesforce/codegen-350M-multi  ^
    --train_file trainData ^train3.json  ^
    --validation_file trainData ^test.json  ^
    --per_device_train_batch_size 4  ^
    --per_device_eval_batch_size 4  ^
    --do_train  ^
    --do_eval  ^
    --output_dir /tmp/test-clm
