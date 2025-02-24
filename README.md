# MULTITAT: Benchmarking Multilingual Table-and-Text Question Answering

## Introduction
This repository contains the code and data for the paper MULTITAT: Benchmarking Multilingual Table-and-Text Question Answering


## Dataset

MULTITAT contains parallel data, including 250 questions from 233 hybrid context, across 11 diverse languages. 
We sample English data from [HybridQA](https://hybridqa.github.io), [TAT-QA](https://nextplusplus.github.io/TAT-QA/), and [SciTAT](https://github.com/zhxlia/SciTaT), and translate them into Bengali (bn), Chinese (zh), French (fr), German (de), Japanese (ja), Russian (ru), Spanish (es), Swahili (sw), Telugu (te), and Thai (th).

You can download our dataset via [MULTITAT](./dataset).

Each instance in our dataset contains the following keys:
```python
{
        "source": {
            "dataset": The source dataset of the corresponding English instance,
            "qid": The unique id of the corresponding English instance,
            "answer_from": the answer source, including table, text, and hybrid,
            "answer_type": the answer type, including arithmetic, span, and count,
        }, 
        "text": {               
            # The paragraphs related to the instance
            "paragraph": The unique id of the paragraph,
        },
        "table": {                                                                                              
            # The table related to the instance
            "content": List[List[str]], the content of the table,
        },
        "question": The question of the instance,
        "explanation": The reasoning rationale of the instance,
        "answer": The answer of the question
    }

```

## Baselines

You can run [run_baseline.slurm](./inference/slurm/run_baseline.slurm) to conduct the baselines.

You can run [run_Ours.slurm](./inference/slurm/run_Ours.slurm) to perform our proposed baseline.

## Evaluation

You can run [evaluate.slurm](./inference/slurm/evaluate.slurm) to evaluate your predicted result.

