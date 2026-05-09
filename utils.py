import os
import json
import time
import requests
import openai
import copy
import re
import ast
from openai.types.chat import ChatCompletion

from loguru import logger


DEBUG = int(os.environ.get("DEBUG", "0"))


def generate_together(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    n=1,
):

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            endpoint = "https://api.together.xyz/v1/chat/completions"

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to `{model}`."
                )

            res = requests.post(
                endpoint,
                json={
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": (temperature if temperature > 1e-4 else 0),
                    "messages": messages,
                    "n": n,
                },
                headers={
                    "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY')}",
                },
            )
            if "error" in res.json():
                logger.error(res.json())
                if res.json()["error"]["type"] == "invalid_request_error":
                    logger.info("Input + output is longer than max_position_id.")
                    return None, None

            if n == 1:
                output = res.json()["choices"][0]["message"]["content"].strip()
            else:
                output = [item["message"]["content"].strip() for item in res.json()["choices"]]

            break

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return None,None

    if DEBUG:
        logger.debug(f"Output: `{output[:20]}...`.")

    return output, messages


def _resolve_vllm_endpoint(model):
    """
    一个 vLLM server 进程只能 serve 一个模型，因此当 `reference_models`
    中包含多个不同模型时，必须按模型路由到不同的 base_url。

    解析优先级:
        1. VLLM_MODEL_ENDPOINTS (JSON, 形如 {"meta-llama/Llama-3-70b": "http://localhost:8001/v1"})
        2. VLLM_BASE_URL (单一 endpoint, 适用于所有模型挂在同一个 router 后的情况)
        3. http://localhost:8000/v1 (默认)
    """
    mapping_str = os.environ.get("VLLM_MODEL_ENDPOINTS")
    if mapping_str:
        try:
            mapping = json.loads(mapping_str)
            if model in mapping:
                return mapping[model]
        except Exception as e:
            logger.warning(f"Failed to parse VLLM_MODEL_ENDPOINTS: {e}")

    return os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")


def generate_vllm(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
    streaming=False,
    n=1,
):
    """
    通过 vLLM 的 OpenAI 兼容接口生成回复，签名与返回值与 `generate_together` 对齐，
    便于在 `process_fn` 中直接替换。

    环境变量:
        VLLM_MODEL_ENDPOINTS: JSON dict，按模型名映射到 base_url（多端口部署时使用）
        VLLM_BASE_URL: 兜底的单一 base_url，默认 "http://localhost:8000/v1"
        VLLM_API_KEY:  鉴权 token，默认 "EMPTY"（vLLM 默认不校验）

    返回:
        (output, messages)
          - n == 1 时 output 为 str
          - n  > 1 时 output 为 List[str]
          - 失败时返回 (None, None)
    """

    base_url = _resolve_vllm_endpoint(model)
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    output = None

    for sleep_time in [1, 2, 4, 8, 16, 32]:

        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}...`) to vLLM `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=(temperature if temperature > 1e-4 else 0),
                max_tokens=max_tokens,
                n=n,
            )

            if n == 1:
                output = completion.choices[0].message.content.strip()
            else:
                output = [c.message.content.strip() for c in completion.choices]

            break

        except openai.BadRequestError as e:
            logger.error(e)
            logger.info("Input + output is longer than max_position_id.")
            return None, None

        except Exception as e:
            logger.error(e)
            if DEBUG:
                logger.debug(f"Msgs: `{messages}`")

            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    if output is None:

        return None, None

    if DEBUG:
        preview = output[:20] if isinstance(output, str) else output[0][:20]
        logger.debug(f"Output: `{preview}...`.")

    return output, messages


def generate_together_stream(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    endpoint = "https://api.together.xyz/v1"
    client = openai.OpenAI(
        api_key=os.environ.get("TOGETHER_API_KEY"), base_url=endpoint
    )
    endpoint = "https://api.together.xyz/v1/chat/completions"
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature if temperature > 1e-4 else 0,
        max_tokens=max_tokens,
        stream=True,  # this time, we set stream=True
    )

    return response


def generate_openai(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):

    client = openai.OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:

            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output


def generate_gitaigc(
    model,
    messages,
    max_tokens=2048,
    temperature=0.7,
):
    api_key = 'sk-jfhonAkNxKzfViMm5dD93d8d8a0844D7B4160bE837A024Fa'
    url = 'https://gitaigc.com/v1/chat/completions'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }

    for sleep_time in [1, 2, 4, 8, 16, 32]:
        try:
            
            if DEBUG:
                logger.debug(
                    f"Sending messages ({len(messages)}) (last message: `{messages[-1]['content'][:20]}`) to `{model}`."
                )

            # completion = client.chat.completions.create(
            #     model=model,
            #     messages=messages,
            #     temperature=temperature,
            #     max_tokens=max_tokens,
            # )
            completion = requests.post(
                url, headers=headers, json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }
            )
            completion = ChatCompletion(**(completion.json()))

            output = completion.choices[0].message.content
            break

        except Exception as e:
            logger.error(e)
            logger.info(f"Retry in {sleep_time}s..")
            time.sleep(sleep_time)

    output = output.strip()

    return output, messages


def generate_with_references(
    model,
    messages,
    system,
    role='',
    references=[],
    max_tokens=2048,
    temperature=0.7,
    generate_fn=generate_together,
):

    if len(references) > 0:
        messages = inject_references_to_messages(messages, references, system)

    if role != '':
        messages = inject_role_to_messages(messages, role)


    return generate_fn(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def extract_indexes_and_indicator_from_output(output):
    # Regular expressions to extract "chosen responses" and "end debate"
    chosen_responses_pattern = re.compile(r'"chosen responses": (\[.*?\])')
    end_debate_pattern = re.compile(r'"end debate": (True|False|true|false)')

    # Extract the chosen responses
    chosen_responses_match = chosen_responses_pattern.search(output)
    chosen_responses = ast.literal_eval(chosen_responses_match.group(1)) if chosen_responses_match else None

    # Extract the end debate value
    end_debate_match = end_debate_pattern.search(output)
    end_debate = True if end_debate_match and end_debate_match.group(1).lower() == 'true' else False

    return chosen_responses, end_debate



def extract_role_from_output(output):
    roles = []
    new_role = ""
    for line in output.strip().split('\n'):
        if "Generated Role Description" in line:
            if new_role != "":
                roles.append(new_role.strip())
            new_role = ""
            continue
        new_role += line
    roles.append(new_role.strip())

    return roles


def inject_role_to_messages(
    messages,
    role,
):

    messages = [{"role": "system", "content": role}] + messages
    return messages


def inject_references_to_messages(
    messages,
    references,
    system,
):

    messages = copy.deepcopy(messages)

    for i, reference in enumerate(references):

        system += f"Response {i}\n. {reference}"

    if messages[0]["role"] == "system":

        messages[0]["content"] += "\n\n" + system

    else:

        messages = [{"role": "system", "content": system}] + messages

    return messages

def get_tokenizer_name(model_name):
    if model_name == "meta-llama/Llama-3-70b-chat-hf":
        return "meta-llama/Meta-Llama-3-70B-Instruct"
    elif model_name == "microsoft/WizardLM-2-8x22B":
        return "alpindale/WizardLM-2-8x22B"
    else:
        return model_name