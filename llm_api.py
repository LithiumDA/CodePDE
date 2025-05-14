from anthropic import Anthropic
from google import genai
from google.genai import types
from openai import OpenAI
import time


def get_client(messages, cfg):
    if 'gpt' in cfg.model.family_name or cfg.model.family_name == 'o':
        client = OpenAI(api_key=cfg.model.api_key)
    elif 'claude' in cfg.model.family_name:
        client = Anthropic(api_key=cfg.model.api_key)
    elif 'deepseek' in cfg.model.family_name:
        client = OpenAI(api_key=cfg.model.api_key, base_url=cfg.model.base_url, timeout=cfg.model.timeout)
    elif 'gemini' in cfg.model.family_name:
        client = genai.Client(api_key=cfg.model.api_key)
    elif cfg.model.family_name == 'qwen':
        client = OpenAI(api_key=cfg.model.api_key, base_url=cfg.model.base_url)
    else:
        raise ValueError(f'Model {cfg.model.family_name} not recognized')
    return client


def generate_response(messages, cfg):
    client = get_client(messages, cfg)
    model_name = cfg.model.name
    if 'o1' in model_name or 'o3' in model_name:
        # Need to follow the restrictions of o1 arguments
        # TODO: add these to the hydra config
        num_tokens = 16384
        temperature = 1.0
        start_time = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=num_tokens,
            temperature=temperature)
        end_time = time.time()
        print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
        return response
    
    if 'claude' in model_name:
        num_tokens = 8192  # Give claude more tokens
        temperature = 0.7

        if len(messages)>0 and messages[0]['role'] == 'system':
            system_prompt = messages[0]['content']
            messages = messages[1:]

        start_time = time.time() 
        if cfg.model.thinking:
            num_thinking_tokens = 12288
            response = client.messages.create(
                model=model_name,
                max_tokens=num_tokens+num_thinking_tokens,
                thinking= {"type": "enabled", "budget_tokens": num_thinking_tokens},
                system=system_prompt,
                messages=messages,
                # temperature has to be set to 1 for thinking
                temperature=1.0,
            )
        else:
            response = client.messages.create(
                model=model_name,
                max_tokens=num_tokens,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
            )
        end_time = time.time()
        print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
        return response

    if 'gemini' in model_name:
        start_time = time.time() 
        if len(messages)>0 and messages[0]['role'] == 'system':
            # If the first message is a system message, we need to prepend it to the user message
            system_prompt = messages[0]['content']
            messages = messages[1:]
            messages[0]['content'] = system_prompt + messages[0]['content']

        for message in messages:
            if message['role'] == 'assistant':
                message['role'] = 'model'
    
        chat = client.chats.create(
            model=model_name,
            history=[
                types.Content(role=message['role'], parts=[types.Part(text=message['content'])])
                for message in messages[:-1]
            ],
        )
        response = chat.send_message(message=messages[-1]['content'])
        end_time = time.time()
        print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
        return response

    num_tokens = 4096
    temperature = 0.7

    start_time = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=num_tokens,
        temperature=temperature,
        stream=('qwq' in model_name),
    )

    if 'qwq' in model_name:
        answer_content = ""
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    # We don't need to print the reasoning content
                    pass
                else:
                    answer_content += delta.content
        response = answer_content

    end_time = time.time()
    print(f'It takes {model_name} {end_time - start_time:.2f}s to generate the response.')
    return response

