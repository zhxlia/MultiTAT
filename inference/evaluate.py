import sys
import json
import argparse

sys.path.append('.')

from copy import deepcopy

if __name__ == '__main__':
    from inference.inference_CoT import extract_answer 
    from utils.evaluate import TaTQAEmAndF1, fix_answer
    from inference.inference_PoT import safe_execute, extract_program

    args = argparse.ArgumentParser()
    args.add_argument('--data_file', type=str)
    args.add_argument('--gold_file', type=str)
    args.add_argument('--dump_file', type=str)
    args.add_argument('--process', action='store_true')
    args.add_argument('--language', type=str)
    args = args.parse_args()

    with open(args.data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    with open(args.gold_file, 'r', encoding='utf-8') as f:
        gdata = json.load(f)
    gdata_list = deepcopy(gdata)
    gdata = {d["source"]["qid"]: d for d in gdata} 

    evaluator = TaTQAEmAndF1()
    for di, d in enumerate(data):
        if args.process:
            if "GPT-4o" in args.data_file:
                org_d = deepcopy(d)
                d = gdata_list[di]
                d["prediction"] = {"response": org_d[-1]['content'].split("\n")}
                if "PoT" in args.data_file or "Decompose" in args.data_file:
                    program = extract_program("\n".join(d["prediction"]["response"]))
                    answer = safe_execute(program, ['ans'])
                    d["prediction"]["answer"] = fix_answer(answer, args.language)
                else:
                    d["prediction"]["answer"] = extract_answer("\n".join(d["prediction"]["response"]), args.language)
            else:
                if "PoT" or "Decompose" in args.data_file:
                    program = extract_program("\n".join(d["prediction"]["response"]))
                    answer = safe_execute(program, ['ans'])
                    d["prediction"]["answer"] = fix_answer(answer, args.language)
                else:
                    d["prediction"]["answer"] = extract_answer("\n".join(d["prediction"]["response"]), args.language)
            gold_answer = gdata[d["source"]["qid"]]['answer']
            if args.language == 'en' and isinstance(d['prediction']['answer'], str) and len(d['prediction']['answer'].split(" ")) < 5 and d['prediction']['answer'].endswith('.'):
                d['prediction']['answer'] = d['prediction']['answer'][:-1]
            if isinstance(gold_answer, float) and isinstance(d['prediction']['answer'], str) and d['prediction']['answer'].endswith('%'):
                d['prediction']['answer'] = d['prediction']['answer'][:-1]
            elif isinstance(gold_answer, str) and isinstance(d['prediction']['answer'], float) and gold_answer.endswith('%'):
                d['prediction']['answer'] = d['prediction']['answer'] * 0.01
                
        gold = {
            "answer": gdata[d["source"]["qid"]]['answer'],
            "response": d["prediction"]["response"],
            "answer_type": gdata[d["source"]["qid"]]['source']['answer_type'],
            "scale": ""
        }
        if gold['answer_type'] in ['span', 'multi-span'] and not isinstance(gold['answer'], list):
            gold['answer'] = [gold['answer']]
        evaluator(args.language, gold, d['prediction']['answer'], "")
    print(f"{args.data_file}: {str(evaluator)}")

    with open(args.dump_file, 'w', encoding='utf-8') as f:
        json.dump(evaluator.get_raw(), f, ensure_ascii=False, indent=4)
