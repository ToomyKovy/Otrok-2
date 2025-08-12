import hashlib
import json
import re
import shelve
import time
import traceback
import warnings
from datetime import datetime
from functools import partial
from io import BytesIO
from itertools import chain
from pathlib import Path
from typing import Union

from openai import OpenAI, InternalServerError
from tqdm import tqdm

from utils.config import OPENAI_API_KEY, PERPLEXITY_API_KEY


LOG_LEVELS = {
    'ERROR': 5,
    'ISSUE': 4,
    'IMPORTANT': 3,
    'INFO': 2,
    'DEBUG': 1,
}
LOG_NAMES = {val: key for key, val in LOG_LEVELS.items()}


def log(threshold: Union[int, str], message: str, log_level: Union[int, str], id_=None) -> None:
    threshold_int = LOG_LEVELS.get(threshold, threshold)
    if threshold_int >= LOG_LEVELS.get(log_level, log_level):
        if id_ is not None:
            id_ = '-' + id_ + '-'
        else:
            id_ = '-'
        print(f'{datetime.now()} {id_} {LOG_NAMES[threshold_int]:9s} - {message}')


class NoCache:
    def __getitem__(self, key: str) -> dict | None:
        return None

    def __setitem__(self, key: str, value) -> None:
        pass

    def get_meta(self, key: str) -> dict:
        return {}


class FileCache(NoCache):
    def __init__(self, cache_path: str):
        self.cache_path = cache_path
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)

    def __getitem__(self, key: str) -> dict | None:
        with shelve.open(self.cache_path) as db:
            if key not in db:
                return None
            return db[key]

    def __setitem__(self, key: str, value) -> None:
        with shelve.open(self.cache_path) as db:
            db[key] = value


LLM_MODELS = {
    'gpt-4o-2024-08-06': {
        'cost_input': 2.50 / 1e6,
        'cost_output': 10 / 1e6,
        'llm_provider': 'openai',
    },
    'gpt-4o-2024-11-20': {
        'cost_input': 2.50 / 1e6,
        'cost_output': 10 / 1e6,
        'llm_provider': 'openai',
    },
    'gpt-4o-mini-2024-07-18': {
        'cost_input': 0.15 / 1e6,
        'cost_output': 0.6 / 1e6,
        'llm_provider': 'openai',
    },
    'gpt-4.1-2025-04-14': {
        'cost_input': 2 / 1e6,
        'cost_output': 8 / 1e6,
        'llm_provider': 'openai',
    },
    'sonar-reasoning': {
        'cost_input': 1 / 1e6,
        'cost_output': 5 / 1e6,
        'cost_search': 5 / 1000,
        'min_searches': 1,
        'llm_provider': 'perplexity',
    },
    'sonar-reasoning-pro': {
        'cost_input': 2 / 1e6,
        'cost_citation': 2 / 1e6,
        'cost_output': 8 / 1e6,
        'cost_search': 5 / 1000,
        'min_searches': 1,
        'llm_provider': 'perplexity',
    },
    'sonar-pro': {
        'cost_input': 3 / 1e6,
        'cost_citation': 3 / 1e6,
        'cost_output': 15 / 1e6,
        'cost_search': 5 / 1000,
        'min_searches': 1,
        'llm_provider': 'perplexity',
    },
    'sonar': {
        'cost_input': 1 / 1e6,
        'cost_output': 1 / 1e6,
        'cost_search': 5 / 1000,
        'min_searches': 1,
        'llm_provider': 'perplexity',
    },
}


LLM_API_CONFIG = {
    'openai': {
        'api_key': OPENAI_API_KEY,
        'base_url': 'https://api.openai.com/v1'
    },
    'perplexity': {
        'api_key': PERPLEXITY_API_KEY,
        'base_url': 'https://api.perplexity.ai'
    },
}


def get_total_cost(completion, llm_provider, from_batch_api=False):
    total_factor = 1
    if from_batch_api:
        assert llm_provider == 'openai', "Only OpenAI batch API is supported."
        total_factor = 0.5

    model = completion['model']
    input_tokens = completion['usage']['prompt_tokens']
    output_tokens = completion['usage']['completion_tokens']
    citation_tokens = completion['usage'].get('citation_tokens', 0)
    num_searches = max(completion['usage'].get('num_search_queries', 0), LLM_MODELS.get(model, {}).get('min_searches', 0))

    in_cost = LLM_MODELS.get(model, {}).get('cost_input', 0) * input_tokens
    citation_cost = LLM_MODELS.get(model, {}).get('cost_citation', 0) * citation_tokens
    out_cost = LLM_MODELS.get(model, {}).get('cost_output', 0) * output_tokens
    search_cost = LLM_MODELS.get(model, {}).get('cost_search', 0) * num_searches

    total_cost = (in_cost + citation_cost + out_cost + search_cost) * total_factor
    return total_cost, {
        'input_cost_usd': in_cost,
        'citation_cost_usd': citation_cost,
        'output_cost_usd': out_cost,
        'search_cost_usd': search_cost,
        'total_cost_usd': total_cost,
    }


def extract_reasoning(completion: str) -> (str | None, str):
    """
    Extracts the <think> text </think> from a completion and return the text and remaining completion.
    """
    reasoning_text = re.match(r'^(.*)\s*<think>(.*)</think>\s*(.*?)$', completion, flags=re.DOTALL)
    if reasoning_text is None:
        return None, completion
    return reasoning_text.group(2).strip(), (reasoning_text.group(1) + '\n' + reasoning_text.group(3)).strip()


def extract_json_data(text: str) -> dict | list:
    """
    Attempts to extract data from JSON codes in a text.
    Returns a dict if successful, else an empty dict.
    """
    # 1) Try standard triple-backtick JSON fences
    matches = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    matches2 = re.findall(r"(\{.*})", text, flags=re.DOTALL)
    for match in chain(matches, matches2):
        match = match.strip()
        try:
            parsed = json.loads(match)
            if isinstance(parsed, dict):
                return parsed
            else:
                pass
        except json.JSONDecodeError:
            pass  # continue to next match if parsing fails

    # 2) Try parsing entire text if all else fails
    text = text.strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, (dict, list)) else {}
    except json.JSONDecodeError:
        return {}


class SimpleWarning:
    def warn(self, affected_system: str, message: str, request_md5: str = None, full_aggregate_id: int = None):
        warnings.warn(f"Warning[{affected_system}]: {message}")


class LLM:
    def __init__(self, model='gpt-4o-mini-2024-07-18', system_prompt='You are a helpful assistant',
                 log_level=5, call_rate_limiter=None, log_id=None):

        self.model = model
        self.default_system_prompt = system_prompt
        self.cache = FileCache('cache.dat')

        self.log = partial(log, log_level=log_level, id_=log_id)
        self.call_rate_limiter = call_rate_limiter

        # Decide provider: known mapping → provider; URL-like → custom base URL; otherwise default to OpenAI
        mapped_provider = LLM_MODELS.get(model, {}).get('llm_provider')
        if mapped_provider:
            self._llm_provider = mapped_provider
        else:
            if isinstance(model, str) and model.startswith(("http://", "https://")):
                self._llm_provider = "$" + model
            else:
                # Unknown model name but not a URL → assume OpenAI
                self._llm_provider = "openai"

        if self._llm_provider.startswith('$'):
            self.client = OpenAI(api_key="lm-studio", base_url=self._llm_provider[1:])  # for llama.cpp api key can be any non-empty string
        else:
            self.client = OpenAI(**LLM_API_CONFIG[self._llm_provider])

        self._api_called = 0
        self._cache_used = 0

        self.n_trials = 10

        self.warning = SimpleWarning()

    def _get_cache_key(self, **kwargs):
        data = json.dumps(kwargs, sort_keys=True)
        return hashlib.md5(data.encode()).hexdigest()

    def _get_messages(self, system_prompt, input_prompt) -> list[dict]:
        messages = []
        if len(system_prompt) > 0:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": input_prompt})
        return messages

    def _process_response(self, full_response: dict, from_batch_api=False) -> dict:
        total_cost, cost_details = get_total_cost(full_response, self._llm_provider, from_batch_api=from_batch_api)
        return {
            'model': full_response["model"],
            'completion': full_response["choices"][0]["message"]["content"],
            'cost_usd': total_cost,
            'cost_details': cost_details,
            'citations': full_response.get('citations'),
            'full_response': full_response,
        }

    def call_api(self, input_prompt, temperature=0., max_tokens=1000, system_prompt=None, force=False, **kwargs):
        assert isinstance(input_prompt, str) and len(input_prompt) > 0, "Input prompt cannot be empty."
        api_inputs = {
            'model': self.model,
            'system_prompt': system_prompt or self.default_system_prompt,
            'input_prompt': input_prompt,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        assert isinstance(api_inputs['system_prompt'], str) and len(api_inputs['system_prompt']) > 0, "System prompt cannot be empty."
        response = None
        cache_md5 = self._get_cache_key(**api_inputs)
        if not force:
            cashed_data = self.cache[cache_md5]
            if cashed_data is not None:
                if api_inputs != cashed_data['api_inputs']:
                    raise ValueError(f"There are inconsistencies between api_inputs {api_inputs} and "
                                     f"cashed api_inputs {cashed_data['api_inputs']}")
                self._cache_used += 1
                self.log("INFO", f"Cache hit: {cache_md5}")
                response = cashed_data['response']

        if response is None:
            self._api_called += 1

            api_kwargs = api_inputs.copy()
            api_kwargs['messages'] = self._get_messages(api_kwargs.pop('system_prompt'), api_kwargs.pop('input_prompt'))
            self.log("INFO", f"Calling API.")

            for trial_id in range(self.n_trials):
                try:
                    if self.call_rate_limiter is not None:
                        self.log("DEBUG", f"Waiting for next call.")
                        self.call_rate_limiter.wait_for_next()
                    response = self.client.chat.completions.create(**api_kwargs).to_dict()
                    self.log("INFO", f"API called successfully.")
                    break

                except InternalServerError:
                    traceback_str = traceback.format_exc()
                    self.warning.warn("API calling", traceback_str)
                    time.sleep(10)
            else:
                raise AssertionError(f"Failed to call API after {self.n_trials} trials.")

            self.log("INFO", f"Saving results to {self.cache.__class__.__name__}.")
            self.cache[cache_md5] = {"api_inputs": api_inputs, "response": response}

        cache_meta = self.cache.get_meta(cache_md5)
        return self._process_response(response, False) | {'request_md5': cache_md5, **cache_meta}

    def _get_gpt_batch_request(self, custom_id: str, model: str, input_prompt: str, temperature=0., system_prompt=None,
                               max_tokens=1000) -> dict:
        return {
            "custom_id": custom_id,
            "method": "POST", "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "temperature": temperature,
                "messages": self._get_messages(system_prompt, input_prompt),
                "max_tokens": max_tokens,
            },
        }

    def _upload_batch_file(self, jsonl):
        assert self._llm_provider == 'openai'
        if isinstance(jsonl, (str, Path)):
            file = open(jsonl, "rb")
        else:
            file = jsonl
        return self.client.files.create(
            file=file,
            purpose="batch",
        )

    def _create_batch_call(self, input_file_id: str, completion_window="24h"):
        assert self._llm_provider == 'openai'
        return self.client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
            metadata={"description": ""},
        )

    def call_batch(self, inputs, use_batch_api=False, use_tqdm=True):
        assert len(inputs) > 0, "No inputs provided."
        if use_batch_api:
            if not self.cache.__class__ == NoCache:
                raise NotImplementedError("Batch API does not support caching.")
            assert self._llm_provider == 'openai'
            # Create BytesIO jsonl file
            jsonl_str = ''
            for input_prompt in inputs:
                jsonl_str += json.dumps(self._get_gpt_batch_request(**input_prompt)) + '\n'

            ub = self._upload_batch_file(BytesIO(jsonl_str.encode()))
            return self._create_batch_call(ub.id)
        else:
            if use_tqdm:
                inputs = tqdm(inputs, desc=f'Calling {self.model} by {self._llm_provider} API')
            return {input_prompt['custom_id']: self.call_api(**input_prompt) for input_prompt in inputs}

    def get_batch_status(self, batch_name: str):
        assert self._llm_provider == 'openai'
        return self.client.batches.retrieve(batch_name)

    def get_batch_results(self, batch_name: str = None, output_file_id: str = None) -> dict[str, dict]:
        if output_file_id is None:
            assert batch_name is not None
            batch = self.get_batch_status(batch_name)
            assert batch.status == 'completed'
            output_file_id = batch.output_file_id

        assert self._llm_provider == 'openai'
        br = self.client.files.content(output_file_id)
        res = [json.loads(line) for line in br.text.strip().splitlines()]
        return {item['custom_id']: self._process_response(item["response"]["body"], True) for item in res}
