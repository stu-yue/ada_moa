export VLLM_MODEL_ENDPOINTS='{
  "Qwen/Qwen3-4B": "http://localhost:8001/v1",
  "Qwen/Qwen3-8B": "http://localhost:8002/v1",
  "google/gemma-3-4b-it": "http://localhost:8003/v1"
}'

echo $VLLM_MODEL_ENDPOINTS

for MODEL in Qwen/Qwen3-8B
    do
        # for DATASET in code_contests_regular tool_use_plan_triple_with_reasoning math_understand
        #     do
        #         python run_mmau.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
        #         python run_mmau.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
        #         python run_mmau.py --dataset $DATASET --model $MODEL
        #     done

        for DATASET in CEB-Conversation-S CEB-Conversation-T 
            do
                # python run_ceb.py --model $MODEL --add_role --moderate_end --moderate_select --reference_models $REFERENCE_MODELS --dataset $DATASET
                # python run_ceb.py --model $MODEL --reference_models $REFERENCE_MODELS --dataset $DATASET
                # python run_ceb.py --dataset $DATASET --model $MODEL

                # New round-structured SC+judge algorithm. Reference models / rounds /
                # gen_num are NOT taken from CLI - they are read from ROUND_SETTING
                # at the top of run_ceb_test.py. Tweak that dict to change the topology.
                python run_ceb_test.py --model $MODEL --add_role --moderate_end --dataset $DATASET
            done
    done
