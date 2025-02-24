import os
import sys
import json
import random
import argparse
import func_timeout

from typing import List

random.seed(42)
sys.path.append('.')


def extract_program(response: str) -> List[str]:
    try:
        lines = response.split('\n')
        result = []
        flag = False
        for line in lines:
            if line.startswith('```') and flag:
                return '\n'.join(result)
            if flag:
                result.append(line)
            if line.startswith('```'):
                flag = not flag
    except:
        pass
    return response


def safe_execute(code_string: str, keys=None):
    def execute(x):
        try:
            exec(x)
            locals_ = locals()
            if keys is None:
                return locals_.get('ans', None)
            else:
                return [locals_.get(k, None) for k in keys]
        except Exception:
            return None
    try:
        ans = func_timeout.func_timeout(5, execute, args=(code_string,))
    except func_timeout.FunctionTimedOut:
        ans = None

    return ans


if __name__ == '__main__':
    from utils.generate import LlamaChatGenerator
    from utils.evaluate import fix_answer, TaTQAEmAndF1
    from inference.inference_CoT import evaluate, pack_path, pack_table, pack_text

    args = argparse.ArgumentParser()
    args.add_argument('--data_file', type=str)
    args.add_argument('--demo_file', type=str)
    args.add_argument('--dump_file', type=str)
    args.add_argument('--prompt_file', type=str)
    args.add_argument('--config_file', type=str)
    args.add_argument('--model_name_or_path', type=str)
    args.add_argument('--sample_scale', type=int)
    args.add_argument('--language', type=str, nargs='+')
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
        dump_file = pack_path(args.dump_file, language)
        if not os.path.exists(load_file):
            print(f"File not found: {load_file}")
            continue
        # if os.path.exists(dump_file):
        #     print(f"File exists: {dump_file}")
        #     continue

        with open(load_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if args.sample_scale:
                data = random.sample(data, args.sample_scale)

        if args.instruction_language == 'origin':
            args.instruction_language = language
        if args.demo_evidence_language == 'origin':
            args.demo_evidence_language = language
        if args.demo_qa_language == 'origin':
            args.demo_qa_language = language
        with open(pack_path(args.prompt_file, args.instruction_language), 'r', encoding='utf-8') as f:
            instruction = json.load(f)
        if args.demo_evidence_language == 'multi':
            with open(pack_path(args.demo_file, 'en'), 'r', encoding='utf-8') as f:
                en_demonstrations = json.load(f)
            with open(pack_path(args.demo_file, 'es'), 'r', encoding='utf-8') as f:
                es_demonstrations = json.load(f)
            with open(pack_path(args.demo_file, 'zh'), 'r', encoding='utf-8') as f:
                zh_demonstrations = json.load(f)
            demonstrations = [en_demonstrations[0], es_demonstrations[1], zh_demonstrations[2]]
        else:
            with open(pack_path(args.demo_file, args.demo_evidence_language), 'r', encoding='utf-8') as f:
                demonstrations = json.load(f)
        demonstrations = demonstrations[:3]

        prompts: List[str] = []
        demonstrations_selected = [demonstrations for _ in range(len(data))]
        for d, demo in zip(data, demonstrations_selected):
            prompt_demo = [instruction['PoT']['example'].format(
                table=pack_table(x['table']['content']),
                text=pack_text(x['text']['paragraph']),
                question=x['question'],
                answer=x['program']
            ) for x in demo]
            prompt_user = instruction['PoT']['user'].format(
                table=pack_table(d['table']['content']),
                text=pack_text(d['text']['paragraph']),
                question=d['question']
            )
            prompts.append(instruction['PoT']['prompt'].format(
                examples="\n\n---\n\n".join(prompt_demo),
                example_user=prompt_user
            ))
        print(prompts[-1])
        print("\n\n\n")
        if args.dump_prompt:
            results = [{
                "idx": d['source']['qid'],
                "prompt": p
            } for d, p in zip(data, prompts)]
            with open(pack_path(args.dump_file, language), 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            continue

        responses = generator.generate(prompts, config)
        for d, r in zip(data, responses):
            program = extract_program(r[0][0])
            answer = safe_execute(program, ['ans'])
            answer = fix_answer(answer, args.language)
            d['prediction'] = {
                "response": r[0][0].split('\n'),
                "program": program.split('\n'),
                "answer": answer
            }

        evaluator: TaTQAEmAndF1 = evaluate(data, language)
        print(f"{pack_path(args.dump_file, language)} : {str(evaluator)}")
        with open(pack_path(args.dump_file, language), 'w', encoding='utf-8') as f:
            json.dump(evaluator.get_raw(), f, ensure_ascii=False, indent=4)
