export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8000/v1
export GRADER_OPENAI_BASE_URL=https://api.deepseek.com
export GRADER_OPENAI_API_KEY=sk-b69df43cd25f41138ce4cb13f7bf2f7a
export GRADER_MODEL=deepseek-v4-pro


for MODEL_NAME in Qwen/Qwen3-8B
    do
        for DATASET in CEB-Conversation-S CEB-Conversation-T
            do
                # REFERENCE_MODELS="Qwen/Qwen3-4B google/gemma-3-4b-it Qwen/Qwen3-8B"
                # python eval_ceb.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                # python eval_ceb.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                # python eval_ceb.py --dataset $DATASET --model $MODEL

                # OUTPUT_PATH=$DATASET/alg-roundjudge_model-<short>_rounds-<R>_nrefs-<N>_addrole-<bool>_modend-<bool>.jsonl
                python eval_ceb.py --dataset $DATASET --model $MODEL_NAME \
                    --output_path $OUTPUT_PATH

            done
    done


