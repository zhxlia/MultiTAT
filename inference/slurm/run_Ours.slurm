for REASONING in Ours; do
    for SCALE in 70b; do
        DUMP_PATH=./inference/result/Llama3.1-$SCALE/$REASONING/en
        [ ! -d "$DUMP_PATH" ] && mkdir -p "$DUMP_PATH"
        python3 ./inference/inference_$REASONING.py \
            --data_file ./dataset/dev.{language}.json \
            --demo_file ./inference/prompt/demonstration/{language}.json \
            --dump_file $DUMP_PATH/dev.{language}.json \
            --prompt_file ./inference/prompt/instruction/{language}.json \
            --model_name_or_path ./model/Llama3.1/$SCALE \
            --config_file ./config/Llama3.1.json \
            --instruction_language origin \
            --language bn de es fr ja ru sw te th zh \
            --demo_evidence_language en \
            --demo_qa_language en
    done
done

rm ./*.csv
