import sys
import random

from copy import deepcopy
from functools import partial
from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map

from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

random.seed(42)
sys.path.append('.')


class BaseSelector():
    def __init__(self, demonstrations: List[Dict[str, Any]]):
        self.demonstrations = deepcopy(demonstrations)
        random.shuffle(self.demonstrations)

    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        return [(d, 0) for d in self.demonstrations[:number]]


class BM25Selector(BaseSelector):
    def __init__(self, demonstrations: List[Dict[str, Any]]):
        super().__init__(demonstrations)
        self.bm25 = BM25Okapi([word_tokenize(d['question'])
                              for d in demonstrations])

    def select(self, number: int, example: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        if 'question_tokenized' in example:
            question_tokenized = example['question_tokenized'].split()
        else:
            question_tokenized = word_tokenize(example['question'])
            example['question_tokenized'] = ' '.join(question_tokenized)
        scores = self.bm25.get_scores(question_tokenized)
        sorted_demos = [(demo, score) for score, demo in sorted(
            zip(scores, self.demonstrations), key=lambda x: x[0], reverse=True)[:number]]
        return sorted_demos


SELECTOR_MAP: Dict[str, BaseSelector] = {
    "base": BaseSelector,
    "bm25": BM25Selector
}


def select_single_pair(example: Dict[str, Any], number: int, selector: BaseSelector) -> Dict[str, Any]:
    return selector.select(number, example)


def select_multiple(examples: List[Dict[str, Any]], demonstrations: List[Dict[str, str]], selector_type: str = 'bm25', demonstration_number: int = 3) -> List[List[Dict[str, Any]]]:
    if not demonstrations:
        return [[] for _ in examples]

    selector = SELECTOR_MAP[selector_type](
        demonstrations)
    if hasattr(selector, "select_multiple"):
        return selector.select_multiple(demonstration_number, examples)
    partial_func = partial(
        select_single_pair, number=demonstration_number, selector=selector)
    results = process_map(partial_func, examples, chunksize=1,
                          max_workers=None, total=len(examples))
    return [[x[0] for x in r] for r in results]

