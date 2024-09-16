ACCELERATE_LOG_LEVEL=info 

MODEL_NAME=llama # options: llama, gemma
MODEL_SIZE=8b # options: 8b, 7b, 2b

for lang in bn de fr hi te ur
do
    # run training
    accelerate launch \
        --config_file configs/accelerate/deepspeed_zero3.yaml \
        run_sft.py \
        configs/sft/$MODEL_NAME/$MODEL_NAME-$MODEL_SIZE-it-$lang.yaml

    # generate predictions
    python predict.py --model_name $lang-$MODEL_NAME-$MODEL_SIZE-it-v0.5 --lang $lang

    # run evaluation
    python eval.py --model_name $lang-$MODEL_NAME-$MODEL_SIZE-it-v0.5 --lang $lang
done