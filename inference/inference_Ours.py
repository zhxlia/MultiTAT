import os
import sys
import json
import random
import argparse

from typing import List
from copy import deepcopy

random.seed(42)
sys.path.append('.')


if __name__ == '__main__':
    from utils.generate import LlamaChatGenerator
    from utils.evaluate import fix_answer, TaTQAEmAndF1
    from inference.inference_CoT import evaluate, pack_path, pack_table, pack_text, extract_answer
    from inference.inference_PoT import extract_program, safe_execute

    args = argparse.ArgumentParser()
    args.add_argument('--data_file', type=str)
    args.add_argument('--demo_file', type=str)
    args.add_argument('--dump_file', type=str)
    args.add_argument('--prompt_file', type=str)
    args.add_argument('--config_file', type=str)
    args.add_argument('--model_name_or_path', type=str)
    args.add_argument('--sample_scale', type=int)
    args.add_argument('--language', type=str, nargs='+')
    args.add_argument('--q_language', type=str, default='origin')
    args.add_argument('--dump_prompt', action='store_true')
    args.add_argument('--instruction_language', type=str, choices=['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh', 'origin'], default='origin')
    args.add_argument('--demo_evidence_language', type=str, choices=['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh', 'origin', 'multi'], default='origin')
    args.add_argument('--demo_qa_language', type=str, choices=['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh', 'origin', 'multi'], default='origin')
    args = args.parse_args()

    if not args.dump_prompt:
        generator = LlamaChatGenerator(args.model_name_or_path)
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

    for language in args.language:
        load_file = pack_path(args.data_file, language)
        if not os.path.exists(load_file):
            print(f"File not found: {load_file}")
            continue

        with open(load_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if args.sample_scale:
                data = random.sample(data, args.sample_scale)
            if args.q_language != 'origin':
                q_load_file = pack_path(args.data_file, args.q_language)
                with open(q_load_file, 'r', encoding='utf-8') as qf:
                    q_data = json.load(qf)
            else:
                q_data = data
                    
        instruction_language = args.instruction_language
        demo_evidence_language = args.demo_evidence_language
        demo_qa_language = args.demo_qa_language
        q_language = args.q_language
        if args.instruction_language == 'origin':
            instruction_language = language
        if args.demo_evidence_language == 'origin':
            demo_evidence_language = language
        if args.demo_qa_language == 'origin':
            demo_qa_language = language
        if args.q_language == 'origin':
            q_language = language

        with open(pack_path(args.prompt_file, instruction_language), 'r', encoding='utf-8') as f:
            instruction = json.load(f)
        if demo_evidence_language == 'multi':
            with open(pack_path(args.demo_file, 'en'), 'r', encoding='utf-8') as f:
                en_demonstrations = json.load(f)
            with open(pack_path(args.demo_file, 'es'), 'r', encoding='utf-8') as f:
                es_demonstrations = json.load(f)
            with open(pack_path(args.demo_file, 'zh'), 'r', encoding='utf-8') as f:
                zh_demonstrations = json.load(f)
            demonstrations = [en_demonstrations[0], es_demonstrations[1], zh_demonstrations[2]]
        else:
            with open(pack_path(args.demo_file, demo_evidence_language), 'r', encoding='utf-8') as f:
                evidence_demonstrations = json.load(f)
            with open(pack_path(args.demo_file, demo_qa_language), 'r', encoding='utf-8') as f:
                qa_demonstrations = json.load(f)
            demonstrations = []
            for ed, qd in zip(evidence_demonstrations, qa_demonstrations):
                de = deepcopy(qd)
                de["table"] = ed["table"]
                de["text"] = ed["text"]
                demonstrations.append(de)

        prompts: List[List[str]] = [[] for _ in range(len(data))]
        demonstrations_selected = [demonstrations for _ in range(len(data))]
        for ti, prompt in enumerate(instruction['Decompose']['prompt']):
            for di, (d, demo) in enumerate(zip(data, demonstrations_selected)):
                if ti == 0:
                    prompt_demo = [instruction['Decompose']['example'][ti].format(
                        table=pack_table(x['table']['content']),
                        text=pack_text(x['text']['paragraph']),
                        question=x['question'],
                        answer=x['link']
                    ) for x in demo]
                    prompt_user = instruction['Decompose']['user'][ti].format(
                        table=pack_table(d['table']['content']),
                        text=pack_text(d['text']['paragraph']),
                        # question=d['question']
                        question=q_data[di]['question']
                    )
                    prompts[di].append(instruction['Decompose']['prompt'][ti].format(
                        examples="\n\n---\n\n".join(prompt_demo),
                        example_user=prompt_user
                    ))
                else:
                    prompt_demo = [instruction['Decompose']['example'][ti].format(
                        table=pack_table(x['table']['content']),
                        text=pack_text(x['text']['paragraph']),
                        question=x['question'],
                        answer=x['program']
                    ) for x in demo]
                    prompt_user = instruction['Decompose']['user'][ti].format(
                        table=pack_table(d['table']['content']),
                        text=pack_text(d['text']['paragraph']),
                        question=q_data[di]['question']
                    )
                    prompts[di].append(instruction['Decompose']['prompt'][ti].format(
                        examples="\n\n---\n\n".join(prompt_demo),
                        language=language,
                        example_user=prompt_user
                    ))
        for p in prompts[-1]:
            print(p)
        print("\n\n\n\n\n")
        if args.dump_prompt:
            results = [{
                "idx": d['source']['qid'],
                "prompt": [{
                    "role": "user",
                    "content": p[0]
                }, {
                    "role": "assistant",
                    "content": ""
                },{
                    "role": "user",
                    "content": p[1]
                },{
                    "role": "assistant",
                    "content": ""
                }]
            } for d, p in zip(data, prompts)]
            # with open(pack_path(args.dump_file, language), 'w', encoding='utf-8') as f:
            #     json.dump(results, f, ensure_ascii=False, indent=4)
            continue

        responses = generator.generate_mt(prompts, config)
        for d, r in zip(data, responses):
            program = extract_program(r[1][0][0])
            answer = safe_execute(program, ['ans'])
            answer = fix_answer(answer, args.language)
            d['prediction'] = {
                "link": r[0][0][0].split('\n'),
                "response": r[1][0][0].split('\n'),
                "answer": answer
            }

        evaluator: TaTQAEmAndF1 = evaluate(data, language)
        print(f"{pack_path(args.dump_file, language)} : {str(evaluator)}")
        with open(pack_path(args.dump_file, language), 'w', encoding='utf-8') as f:
            json.dump(evaluator.get_raw(), f, ensure_ascii=False, indent=4)

