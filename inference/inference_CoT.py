import os
import sys
import json
import random
import argparse

from typing import List

random.seed(42)
sys.path.append('.')


from utils.evaluate import fix_answer, TaTQAEmAndF1


def pack_table(table: List[List[str]]) -> str:
    if len(table) == 0:
        return ""

    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*table)]

    def format_row(row):
        return '| ' + ' | '.join(f'{cell:{col_widths[i]}}' for i, cell in enumerate(row)) + ' |'

    header_separator = '| ' + \
        ' | '.join('-' * width for width in col_widths) + ' |'
    markdown_table = format_row(table[0]) + '\n' + header_separator + '\n'
    markdown_table += '\n'.join(format_row(row) for row in table[1:])
    return markdown_table


def pack_text(text: List[str]) -> str:
    return "\n".join(text)


def extract_answer(response: str, language: str) -> List[str]:
    try:
        for tag in [': ', '：', ':\n']:
            if tag in response:
                response = response.rsplit(tag)[-1]
                break
    except:
        pass
    # print(response)
    return fix_answer(response, language)


def evaluate(data: list, language: str):
    from utils.evaluate import TaTQAEmAndF1

    evaluator = TaTQAEmAndF1()
    for d in data:
        d['answer_type'] = d['source']['answer_type'],
        d['scale'] = ""
        if d['answer_type'] in ['span', 'multi-span'] and not isinstance(d['answer'], list):
            d['answer'] = [d['answer']]
        evaluator(language, d, d['prediction']['answer'], "")
    return evaluator


def pack_path(path: str, language: str) -> str:
    return path.format(language=language)


if __name__ == '__main__':
    from utils.generate import LlamaChatGenerator

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
        if not os.path.exists(load_file):
            print(f"File not found: {load_file}")
            continue

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

            

        prompts: List[str] = []
        demonstrations_selected = [demonstrations for _ in range(len(data))]
        for d, demo in zip(data, demonstrations_selected):
            prompt_demo = [instruction['CoT']['example'].format(
                table=pack_table(x['table']['content']),
                text=pack_text(x['text']['paragraph']),
                question=x['question'],
                explanation=x['explanation'],
                answer=fix_answer(x['answer'], language)
            ) for x in demo]
            prompt_user = instruction['CoT']['user'].format(
                table=pack_table(d['table']['content']),
                text=pack_text(d['text']['paragraph']),
                question=d['question']
            )
            prompts.append(instruction['CoT']['prompt'].format(
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
            d['prediction'] = {
                "response": r[0][0].split('\n'),
                "answer": extract_answer(r[0][0], language)
            }

        evaluator: TaTQAEmAndF1 = evaluate(data, language)
        print(f"{pack_path(args.dump_file, language)} : {str(evaluator)}")
        with open(pack_path(args.dump_file, language), 'w', encoding='utf-8') as f:
            json.dump(evaluator.get_raw(), f, ensure_ascii=False, indent=4)
