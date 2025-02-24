import ast
import json
import numpy as np
import pandas as pd

from utils.tatqa_utils import *
from typing import Set, Tuple, Union
from scipy.optimize import linear_sum_assignment


def fix_answer(answer, language: str):
    # print("#" * 50)
    # print(answer)
    # if language == 'en' and len(answer.split(" ")) < 5 and answer.endswith('.'):
    #     answer = answer[:-1]
    try:
        if language == 'de' or language == 'fr':
            answer = re.sub(r',(?!\s)', '.', answer)
        if answer.startswith('[') and answer.endswith(']'):
            # result = answer[1:-1].split(', ')
            # result = [r.strip('\"\'') for r in result]
            try:
                # 使用 ast.literal_eval 安全地解析字符串
                result_list = ast.literal_eval(answer)
                # 检查解析后的结果是否为列表
                if isinstance(result, list):
                    result = result_list
            except (ValueError, SyntaxError):
                result = answer
        else:
            result = answer
    except:
        result = answer
    if not isinstance(result, list):
        result = [result]
    for i, x in enumerate(result):
        try:
            result[i] = float(x)
        except:
            pass
    if len(result) == 1:
        if not isinstance(result[0], float):
            # print(str(result[0]))
            return str(result[0])
        # print(result[0])
        return result[0]
    else:
        result = [str(x) for x in result]
    # print(result)
    return result


def _answer_to_bags(
    answer: Union[str, List[str], Tuple[str, ...]],
    language: str
) -> Tuple[List[str], List[Set[str]]]:
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
    normalized_spans: List[str] = []
    token_bags = []
    for raw_span in raw_spans:
        normalized_span = normalize_answer(raw_span, language)
        normalized_spans.append(normalized_span)
        tokens = tokenize(normalized_span, language)
        token_bags.append(set(tokens))
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> List[float]:
    """
    Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
    between them and gets maximum metric values over all the answers.
    """
    scores = np.zeros([len(gold), len(predicted)])
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # if _match_numbers_if_present(gold_item, pred_item): no need to match number in tatqa
            scores[gold_index, pred_index] = _compute_f1(pred_item, gold_item)
    row_ind, col_ind = linear_sum_assignment(-scores)

    max_scores = np.zeros([max(len(gold), len(predicted))])
    for row, column in zip(row_ind, col_ind):
        max_scores[row] = max(max_scores[row], scores[row, column])
    return max_scores


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> float:
    intersection = len(gold_bag.intersection(predicted_bag))
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
    f1 = (2 * precision * recall) / (precision +
                                     recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    return f1


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    gold_numbers = set()
    predicted_numbers = set()
    for word in gold_bag:
        if is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if is_number(word):
            predicted_numbers.add(word)
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(predicted: Union[str, List[str], Tuple[str, ...]],
                gold: Union[str, List[str], Tuple[str, ...]],
                language: str) -> Tuple[float, float]:
    """
    Takes a predicted answer and a gold answer (that are both either a string or a list of
    strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
    writing a script for evaluating objects in memory (say, the output of predictions during
    validation, or while training), this is the function you want to call, after using
    :func:`answer_json_to_strings` when reading the gold answer from the released data file.
    """
    predicted_bags = _answer_to_bags(predicted, language)
    gold_bags = _answer_to_bags(gold, language)

    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    f1_per_bag = _align_bags(predicted_bags[1], gold_bags[1])
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)
    return exact_match, f1


def extract_gold_answers(qa_annotation):
    '''
    span
    multi-span
    arithmetic (+ - * /)
    count
    date
    other
    gold answers is a list of list, each item in gold answers is a valid answer
    '''
    answer_type, scale = qa_annotation["answer_type"], qa_annotation['scale']
    answer_content = qa_annotation['answer']
    gold_answers = []
    if answer_type in ['multi-span', 'span']:  # list
        assert isinstance(answer_content, list), answer_content
        gold_answers = answer_content  # multi-span
    elif answer_type in ["arithmetic"]:
        gold_answers.append(str(answer_content))
    elif answer_type in ['count']:
        gold_answers.append(str(int(answer_content)))
    else:
        gold_answers.append(str(answer_content))
    return answer_type, gold_answers, scale


def metric_max_over_ground_truths(metric_fn, predictions, ground_truths, language):
    scores_for_ground_truths = []
    for pred in predictions:
        for ground_truth in ground_truths:
            score = metric_fn(pred, ground_truth, language)
            scores_for_ground_truths.append(score)
    if len(scores_for_ground_truths) == 0:
        return 0, 0
    return max(scores_for_ground_truths)


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_answer_str(answers: list, scale: str):
    """
    :param ans_type:  span, multi-span, arithmetic, count
    :param ans_list:
    :param scale: "", thousand, million, billion, percent
    :param mode:
    :return:

    """
    sorted_ans = sorted(answers)
    ans_temp = []
    for ans in sorted_ans:
        ans_str = str(ans)
        if is_number(ans_str):
            ans_num = to_number(ans_str)
            if ans_num is None:
                if scale:
                    ans_str = ans_str + " " + str(scale)
            else:
                if '%' in ans_str:  # has been handled the answer itself is a percentage
                    ans_str = '%.4f' % ans_num
                else:
                    ans_str = '%.4f' % (
                        round(ans_num, 2) * scale_to_num(scale))
        else:
            if scale:
                ans_str = ans_str + " " + str(scale)
        ans_temp.append(ans_str)
    return [" ".join(ans_temp)]


# handle percentage
def add_percent_pred(prediction_strings, pred_scale, pred):
    """
    to solve [pred = 0.2342] <>   [ans = 23.42 and scale == 'percent']

    :param prediction_strings:
    :param gold_ans_type:
    :param gold_scale:
    :param pred:
    :return:
    """
    if len(pred) > 1:
        return prediction_strings
    pred_str = str(pred[0])
    if pred_str is None:
        return prediction_strings
    # mode only or no pred_scale num only
    if not pred_scale and '%' not in pred_str and is_number(pred_str):
        pred_str = to_number(pred_str)
        if pred_str is None:
            return prediction_strings
        prediction_strings.append('%.4f' % pred_str)
    return prediction_strings


class TaTQAEmAndF1(object):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    """

    def __init__(self) -> None:
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._details = []

    def __call__(self,
                 language: str,
                 ground_truth: dict,
                 prediction: Union[str, List],
                 pred_scale="",
                 pred_span=None,
                 gold_span=None,
                 pred_op=None,
                 gold_op=None):  # type: ignore
        """
        :param ground_truth:
        :param prediction:
        :param pred_scale:
        :param pred_span:
        :param gold_span:
        :param pred_op:
        :param gold_op:
        :return:
        """
        if not prediction:
            exact_match = 0
            f1_score = 0
            span_exact_match = 0
            span_f1_score = 0
        else:
            gold_type, gold_answer, gold_scale = extract_gold_answers(
                ground_truth)
            if not gold_answer:
                exact_match = 0
                f1_score = 0
                span_exact_match = 0
                span_f1_score = 0
            else:
                ground_truth_answer_strings = get_answer_str(
                    gold_answer, gold_scale)

                prediction = prediction if isinstance(
                    prediction, list) else [prediction]
                prediction_strings = get_answer_str(prediction, pred_scale)
                prediction_strings = add_percent_pred(
                    prediction_strings, pred_scale, prediction)
                exact_match, f1_score = metric_max_over_ground_truths(
                    get_metrics,
                    prediction_strings,
                    ground_truth_answer_strings,
                    language
                )
                if gold_type in ['arithmetic', 'count']:
                    """if gold type equals with arithmetic and count, set the f1_score == exact_match"""
                    f1_score = exact_match
                if not pred_span:
                    span_exact_match = 0
                    span_f1_score = 0
                else:
                    pred_span_strings = get_answer_str(pred_span, "")
                    gold_span_strings = get_answer_str(gold_span, "")
                    span_exact_match, span_f1_score = metric_max_over_ground_truths(
                        get_metrics,
                        pred_span_strings,
                        gold_span_strings,
                        language
                    )

        self._total_em += exact_match
        self._total_f1 += f1_score
        self._count += 1
        it = {**ground_truth,
              **{"pred": prediction,
                 "pred_scale": pred_scale,
                 "em": exact_match,
                 "f1": f1_score}}
        self._details.append(it)
    
    def get_raw(self):
        return self._details

    def reset(self):
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
        self._details = []

    def __str__(self):
        return f"TaTQAEmAndF1(em={round(self._total_em / self._count * 100, 2)}, f1={round(self._total_f1 / self._count * 100, 2)}, count={self._count})"
