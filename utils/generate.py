import os
import time
import openai
import torch

from tqdm import tqdm
from copy import deepcopy
from vllm import LLM, SamplingParams
from typing import List, Dict, Any, Tuple
from tqdm.contrib.concurrent import process_map
from concurrent.futures import ProcessPoolExecutor


class GPTAzureChatGenerator(object):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.error_types = {
            "continue_error": [
                "timed out",
                "Connection error",
                "Connection reset by peer",
                "Remote end closed connection without response",
                "occurred in violation of protocol",
                "Failed to resolve",
                "TLSV1_ALERT_INTERNAL_ERROR",
                "Error communicating",
                "The server is overloaded or not ready yet",
                "upstream_error",
                "new_api_error",
                "当前分组上游负载已饱和",
                "Lock wait timeout exceeded"
            ],
            "sleep_error": [
                "call rate limit",
                "token rate limit"
            ],
            "ignore_error": [
                "content",
                "reduce the length"
            ]
        }

    def generate_single(self, packed_data: List[tuple]) -> List[Tuple[str, float]]:
        from openai import AzureOpenAI
        from openai.types.chat import ChatCompletion

        sentence, engine, config = packed_data
        client = AzureOpenAI(
            api_version="2023-07-y-preview",
            azure_endpoint="https://yfllm01.openai.azure.com/",
            api_key="5c870eb35151406180f137ab8e94c703"
        )

        while True:
            try:
                completion: ChatCompletion = client.chat.completions.create(
                    model=engine,
                    messages=[{"role": "user", "content": sentence}],
                    **config)
                return [(x.message.content, 1.0) for x in completion.choices]
            except Exception as e:
                continue_flag = False
                sleep_flag = False
                ignore_flag = False
                for x in self.error_types['continue_error']:
                    if x in str(e):
                        continue_flag = True
                for x in self.error_types['sleep_error']:
                    if x in str(e):
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types['ignore_error']:
                    if x in str(e):
                        ignore_flag = True
                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)
                return [""]

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        config = deepcopy(config)
        if config['parallel']:
            config.pop('parallel')
            if 'batch_size' in config:
                config.pop('batch_size')
            packed_data = [(x, self.model_name, config) for x in source]
            with ProcessPoolExecutor(max_workers=os.cpu_count() // 2) as _:
                result: List[List[str]] = list(process_map(
                    self.generate_single, packed_data, max_workers=os.cpu_count() // 2, chunksize=1))
        else:
            config.pop('parallel')
            result: List[List[str]] = [self.generate_single(
                (x, self.model_name, config)) for x in tqdm(source)]
        return result


class GPTOpenAIChatGenerator(GPTAzureChatGenerator):
    def generate_single(self, packed_data: List[tuple]) -> List[Tuple[str, float]]:
        openai.api_key = "sk-xMvkJLDPJKJHwrTtCa2f975d4bAf47C694Fb11770c57Cc96"
        openai.api_base = "https://api.xiaoai.plus/v1"

        sentence, engine, config = packed_data
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo-1106",
                    messages=[{"role": "user", "content": sentence}],
                    **config
                )
                return [(c.message['content'].strip(), 1.0) for c in response.choices]
            except Exception as e:
                continue_flag = False
                sleep_flag = False
                ignore_flag = False
                for x in self.error_types['continue_error']:
                    if x in str(e):
                        continue_flag = True
                for x in self.error_types['sleep_error']:
                    if x in str(e):
                        sleep_flag = True
                        continue_flag = True
                for x in self.error_types['ignore_error']:
                    if x in str(e):
                        ignore_flag = True
                if sleep_flag:
                    time.sleep(5)
                if continue_flag:
                    continue
                if not ignore_flag:
                    print(e)
                return [("", 0.0)]


class LlamaGenerator(object):
    def __init__(self, model_name_or_path: str):
        def check_cuda_gt_8() -> bool:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_properties = torch.cuda.get_device_properties(i)
                compute_capability = float(
                    f"{device_properties.major}.{device_properties.minor}")
                if compute_capability < 8.0:
                    return False
            return True

        self.llm = LLM(model=model_name_or_path,
                       tensor_parallel_size=torch.cuda.device_count(),
                       dtype="auto" if check_cuda_gt_8() else "float",
                       trust_remote_code=True)
        self.tokenizer = self.llm.get_tokenizer()

    def _filter_too_long_input(self, source: List[str], config: Dict[str, Any]) -> List[str]:
        too_long_data_count = 0
        source_filtered = []
        for i, x in tqdm(enumerate(source), total=len(source), desc="filtering too long input"):
            if len(self.tokenizer(x)['input_ids']) > self.llm.llm_engine.model_config.max_model_len:
                source[i] = "TL;NR"
                too_long_data_count += 1
            else:
                source_filtered.append(x)
        print(f"too long input count: {too_long_data_count}")
        if config['ignore_too_long']:
            return source_filtered
        return source

    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        source = self._filter_too_long_input(source, config)
        sampling_params = SamplingParams(
            temperature=config['temperature'],
            top_p=config['top_p'],
            max_tokens=config['max_tokens'],
            n=config['n'],
            logprobs=1,
            stop=config['stop']
        )

        res_completions = []
        batch_size = config['batch_size']
        batch_instances = batch_data(
            source, batch_size=batch_size)
        for _, prompt in tqdm(enumerate(batch_instances), total=len(batch_instances), desc="generating"):
            if not isinstance(prompt, list):
                prompt = [prompt]
            completions = self.llm.generate(
                prompt, sampling_params, use_tqdm=False)
            for output in completions:
                generated_text = []
                for x in output.outputs:
                    total_logprob = 0.0
                    for t in x.logprobs:
                        max_logprob_token = max(
                            t.items(), key=lambda x: x[1].logprob)
                        if max_logprob_token[0] == 13:
                            break
                        total_logprob += max_logprob_token[1].logprob
                    generated_text.append(
                        (x.text.lstrip('\n'), total_logprob))
                res_completions.append(generated_text)

        return res_completions


class LlamaChatGenerator(LlamaGenerator):
    def generate(self, source: List[str], config: Dict[str, Any]) -> List[List[Tuple[str, float]]]:
        messages_list = [[{"role": "user", "content": x}] for x in source]
        source = [self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False) for messages in messages_list]
        return super().generate(source, config)
    
    def generate_mt(self, source: List[List[str]], config: Dict[str, Any]) -> List[List[List[Tuple[str, float]]]]:
        """
        生成多轮对话，支持每轮多个响应。
        
        :param source: 对话列表，每个对话是一个用户消息列表。
        :param config: 生成参数配置字典。
        :return: 生成的响应列表，结构为 List[List[List[Tuple[str, float]]]]。
        """
        all_responses = [[] for _ in range(len(source))]  # 初始化每个对话的响应
        histories = [[] for _ in range(len(source))]      # 初始化每个对话的历史记录
        
        max_turns = max(len(conv) for conv in source) if source else 0  # 确定最大的轮次数
        
        for turn in range(max_turns):
            prompts = []
            active_indices = []
            
            for idx, conv in enumerate(source):
                if turn < len(conv):
                    user_message = conv[turn]
                    histories[idx].append({"role": "user", "content": user_message})
                    
                    # 应用聊天模板
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        histories[idx], add_generation_prompt=True, tokenize=False)
                    
                    # 检查长度，必要时截断
                    token_length = len(self.tokenizer(formatted_prompt)['input_ids'])
                    if token_length > self.llm.llm_engine.model_config.max_model_len:
                        # 需要截断历史记录
                        while token_length > self.llm.llm_engine.model_config.max_model_len and histories[idx]:
                            histories[idx].pop(0)  # 移除最早的消息
                            formatted_prompt = self.tokenizer.apply_chat_template(
                                histories[idx], add_generation_prompt=True, tokenize=False)
                            token_length = len(self.tokenizer(formatted_prompt)['input_ids'])
                        
                        if token_length > self.llm.llm_engine.model_config.max_model_len:
                            # 即使截断后仍然过长，标记为 "TL;NR"
                            all_responses[idx].append([("TL;NR", 0.0)])
                            continue  # 跳过此轮生成
                    
                    prompts.append(formatted_prompt)
                    active_indices.append(idx)
            
            if not prompts:
                continue  # 当前轮次没有需要生成的提示
            
            # 生成响应
            generated = super().generate(prompts, config)
            
            for i, idx in enumerate(active_indices):
                responses_for_turn = generated[i]  # 这是一个 List[Tuple[str, float]]
                all_responses[idx].append(responses_for_turn)
                # 将第一个响应添加到历史记录中作为上下文
                histories[idx].append({"role": "assistant", "content": responses_for_turn[0][0]})
        
        return all_responses



MODEL_MAP: Dict[str, object] = {
    "llama": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "deepseek": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "glm": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "qwen": {
        'text': LlamaGenerator,
        'chat': LlamaChatGenerator
    },
    "gpt": {
        'text': GPTOpenAIChatGenerator,
        'chat': GPTOpenAIChatGenerator
    }
}


def batch_data(data_list: List[str], batch_size: int) -> List[List[str]]:
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1) * batch_size
        batch_data.append(data_list[start:end])
    last_start = (n-1) * batch_size
    batch_data.append(data_list[last_start:])
    return batch_data


def detect_generator(model_name_or_path: str, mode: str = 'text') -> object:
    for token in MODEL_MAP:
        if token in model_name_or_path.lower():
            return MODEL_MAP[token][mode](model_name_or_path)


def generate_with_llm(model_name_or_path: str, source: List[str], config: Dict[str, Any], mode: str = 'text') -> List[List[Tuple[str, float]]]:
    generator = detect_generator(model_name_or_path, mode)
    results = generator.generate(source, config)
    del generator
    return results


def consistency(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}
    for a, b, c in answers:
        x = str(b)
        if "error" in x.lower():
            continue
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c
    if not count:
        return "", ""
    return record[max(count, key=lambda x: count[x])]


def consistency_with_error(answers: List[Tuple[str, Any, float]]) -> Tuple[str, Any]:
    count: Dict[str, float] = {}
    record: Dict[str, Tuple[str, str]] = {}
    for a, b, c in answers:
        x = str(b)
        # if "error" in x.lower():
        #     continue
        if x not in count:
            count[x] = 0
            record[x] = (a, b)
        count[x] += c
    if not count:
        return "", ""
    return record[max(count, key=lambda x: count[x])]


def pack_answer(answer: str, rationale: str = None) -> str:
    if not rationale:
        return f"So the answer is: {answer}"
    return f"{rationale}\nSo the answer is: {answer}"
