import os
import sys
import json


sys.path.append('.')


if __name__ == '__main__':

    for l in ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']:
        load_path = f'./dataset/dev.{l}.json'
        if not os.path.exists(load_path):
            continue

        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for d in data:
            if d["source"]["answer_type"] == "multi-span":
                d["source"]["answer_type"] = "span"

        with open(load_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f'Fixed {load_path}')
