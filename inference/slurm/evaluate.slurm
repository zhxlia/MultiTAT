#!/bin/bash
#SBATCH -J evaluate
#SBATCH -o ./inference/slurm/evaluate.out
#SBATCH -p gpu02
#SBATCH -N 1
#SBATCH -t 3:00:00
#SBATCH --cpus-per-task=4

source ~/.bashrc
conda activate multi_hybrid

# for REASONING in PoT; do
#     for SCALE in 70B; do
#         for LANGUAGE in bn de es fr ja ru sw te th zh; do
#         # for LANGUAGE in fr; do
#             if [ ! -f "./inference/result/Llama3.1-$SCALE/$REASONING/google-trans/back/dev.$LANGUAGE.json" ]; then
#                 echo "$SCALE-$REASONING-$LANGUAGE not found"
#                 continue
#             fi

#             echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
#             python3 ./inference/evaluate.py \
#                 --data_file ./inference/result/Llama3.1-$SCALE/$REASONING/google-trans/back/dev.$LANGUAGE.json \
#                 --gold_file ./inference/dataset/dev.$LANGUAGE.json \
#                 --dump_file ./inference/result/Llama3.1-$SCALE/$REASONING/google-trans/back/dev.$LANGUAGE.eval.json \
#                 --language $LANGUAGE
#         done
#     done
# done

for REASONING in analyst; do
    for SCALE in 8b 70b; do
        for LANGUAGE in en; do
        # for LANGUAGE in fr; do
            if [ ! -f "./inference/result/Llama3.1-$SCALE/$REASONING/dev.$LANGUAGE.json" ]; then
                echo "$SCALE-$REASONING-$LANGUAGE not found"
                continue
            fi

            echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
            python3 ./inference/evaluate.py \
                --data_file ./inference/result/Llama3.1-$SCALE/$REASONING/dev.$LANGUAGE.json \
                --gold_file ./inference/dataset/dev.$LANGUAGE.json \
                --dump_file ./inference/result/Llama3.1-$SCALE/$REASONING/dev.$LANGUAGE.eval.json \
                --language $LANGUAGE
        done
    done
done

# for REASONING in PoT; do
#     for SCALE in 70B; do
#         # for LANGUAGE in bn de en es fr ja ru sw te th zh; do
#         for LANGUAGE in fr bn; do
#             if [ ! -f "./inference/result/Llama3.1-$SCALE/$REASONING/en/dev.$LANGUAGE.json" ]; then
#                 echo "$SCALE-$REASONING-$LANGUAGE not found"
#                 continue
#             fi

#             echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
#             python3 ./inference/evaluate.py \
#                 --data_file ./inference/result/Llama3.1-$SCALE/$REASONING/en/dev.$LANGUAGE.json \
#                 --gold_file ./inference/dataset/dev.$LANGUAGE.json \
#                 --dump_file ./inference/result/Llama3.1-$SCALE/$REASONING/en/dev.$LANGUAGE.eval.json \
#                 --language $LANGUAGE

#             # echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
#             # python3 ./inference/evaluate.py \
#             #     --data_file ./inference/result/Llama3.1-$SCALE/$REASONING/dev.$LANGUAGE.try.json \
#             #     --gold_file ./inference/dataset/dev.$LANGUAGE.json \
#             #     --dump_file ./inference/result/Llama3.1-$SCALE/$REASONING/dev.$LANGUAGE.try.eval.json \
#             #     --language $LANGUAGE

#             # echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
#             # python3 ./inference/evaluate.py \
#             #     --data_file ./inference/result/Llama3.1-$SCALE/$REASONING/dev.$LANGUAGE.try1.json \
#             #     --gold_file ./inference/dataset/dev.$LANGUAGE.json \
#             #     --dump_file ./inference/result/Llama3.1-$SCALE/$REASONING/dev.$LANGUAGE.try1.eval.json \
#             #     --language $LANGUAGE
#         done
#     done
# done


# for REASONING in PoT; do
#     for LANGUAGE in ru sw te th zh; do
#         if [ ! -f "./inference/result/GPT-4o/$REASONING/native/responses/dev.$LANGUAGE.json" ]; then
#             echo "$SCALE-$REASONING-$LANGUAGE not found"
#             continue
#         fi

#         # echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
#         # python3 ./inference/evaluate.py \
#         #     --data_file ./inference/result/GPT-4o/$REASONING/en/responses/dev.$LANGUAGE.json \
#         #     --gold_file ./inference/dataset/dev.$LANGUAGE.json \
#         #     --dump_file ./inference/result/GPT-4o/$REASONING/en/responses/dev.$LANGUAGE.eval.json \
#         #     --process \
#         #     --language $LANGUAGE

#         echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
#         python3 ./inference/evaluate.py \
#             --data_file ./inference/result/GPT-4o/$REASONING/native/responses/dev.$LANGUAGE.json \
#             --gold_file ./inference/dataset/dev.$LANGUAGE.json \
#             --dump_file ./inference/result/GPT-4o/$REASONING/native/responses/dev.$LANGUAGE.eval.json \
#             --process \
#             --language $LANGUAGE
#     done
# done

# for REASONING in Decompose; do
#     for LANGUAGE in en; do
#         if [ ! -f "./inference/result/GPT-4o/$REASONING/responses/dev.$LANGUAGE.json" ]; then
#             echo "$SCALE-$REASONING-$LANGUAGE not found"
#             continue
#         fi

#         echo "Evaluating $SCALE-$REASONING-$LANGUAGE"
#         python3 ./inference/evaluate.py \
#             --data_file ./inference/result/GPT-4o/$REASONING/responses/dev.$LANGUAGE.json \
#             --gold_file ./inference/dataset/dev.$LANGUAGE.json \
#             --dump_file ./inference/result/GPT-4o/$REASONING/responses/dev.$LANGUAGE.eval.json \
#             --process \
#             --language $LANGUAGE
#     done
# done

rm ./*.csv