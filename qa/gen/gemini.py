"""
Copyright 2024 - Chansung Park

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import ast
import copy
from pathlib import Path
from string import Template

import google.generativeai as genai
import toml
from flatdict import FlatDict

from qa.gen.utils import parse_first_json_snippet


def determine_model_name(given_image=None):
    if given_image is None:
        return 'gemini-pro'
    else:
        return 'gemini-pro-vision'


def construct_image_part(given_image):
    return {'mime_type': 'image/jpeg', 'data': given_image}


def call_gemini(prompt='',
                API_KEY=None,
                given_text=None,
                given_image=None,
                generation_config=None,
                safety_settings=None):
    genai.configure(api_key=API_KEY)

    if generation_config is None:
        generation_config = {
            'temperature': 0.8,
            'top_p': 1,
            'top_k': 32,
            'max_output_tokens': 8192,
        }

    if safety_settings is None:
        safety_settings = [
            {
                'category': 'HARM_CATEGORY_HARASSMENT',
                'threshold': 'BLOCK_ONLY_HIGH'
            },
            {
                'category': 'HARM_CATEGORY_HATE_SPEECH',
                'threshold': 'BLOCK_ONLY_HIGH'
            },
            {
                'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
                'threshold': 'BLOCK_ONLY_HIGH'
            },
            {
                'category': 'HARM_CATEGORY_DANGEROUS_CONTENT',
                'threshold': 'BLOCK_ONLY_HIGH'
            },
        ]

    model_name = determine_model_name(given_image)
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings)

    USER_PROMPT = prompt
    if given_text is not None:
        USER_PROMPT += f"""{prompt}
    ------------------------------------------------
    {given_text}
    """
    prompt_parts = [USER_PROMPT]
    if given_image is not None:
        prompt_parts.append(construct_image_part(given_image))

    response = model.generate_content(prompt_parts)
    return response.text


def try_out(prompt, given_text, gemini_api_key, given_image=None, retry_num=3):
    qna_json = None
    cur_retry = 0

    while qna_json is None and cur_retry < retry_num:
        try:
            qna = call_gemini(
                prompt=prompt,
                given_text=given_text,
                given_image=given_image,
                API_KEY=gemini_api_key)

            qna_json = parse_first_json_snippet(qna)
        except Exception as e:
            cur_retry = cur_retry + 1
            print(f'......failed due to exception {e}')
            print('......retry')

    return qna_json


def get_basic_qa(text, gemini_api_key, truncate=7000):
    prompts = toml.load(Path('.') / 'qa' / 'constants' / 'prompts.toml')
    basic_qa = try_out(
        prompts['basic_qa']['prompt'],
        text[:truncate],
        gemini_api_key=gemini_api_key)
    return basic_qa


def get_deep_qa(text, basic_qa, gemini_api_key, truncate=7000):
    prompts = toml.load(Path('.') / 'qa' / 'constants' / 'prompts.toml')

    title = basic_qa['title']
    qnas = copy.deepcopy(basic_qa['qna'])

    for idx, qna in enumerate(qnas):
        q = qna['question']
        a_expert = qna['answers']['expert']

        depth_search_prompt = Template(
            prompts['deep_qa']['prompt']).substitute(
                title=title,
                previous_question=q,
                previous_answer=a_expert,
                tone='in-depth')
        breath_search_prompt = Template(
            prompts['deep_qa']['prompt']).substitute(
                title=title,
                previous_question=q,
                previous_answer=a_expert,
                tone='broad')

        depth_search_response = {}
        breath_search_response = {}

        while 'follow up question' not in depth_search_response or \
                'answers' not in depth_search_response or \
                'eli5' not in depth_search_response['answers'] or \
                'expert' not in depth_search_response['answers']:
            depth_search_response = try_out(
                depth_search_prompt,
                text[:truncate],
                gemini_api_key=gemini_api_key)

        while 'follow up question' not in breath_search_response or \
                'answers' not in breath_search_response or \
                'eli5' not in breath_search_response['answers'] or \
                'expert' not in breath_search_response['answers']:
            breath_search_response = try_out(
                breath_search_prompt,
                text[:truncate],
                gemini_api_key=gemini_api_key)

        if depth_search_response is not None:
            qna['additional_depth_q'] = depth_search_response
        if breath_search_response is not None:
            qna['additional_breath_q'] = breath_search_response

        qna = FlatDict(qna)
        qna_tmp = copy.deepcopy(qna)
        for k in qna_tmp:
            value = qna.pop(k)
            qna[f'{idx}_{k}'] = value
        basic_qa.update(ast.literal_eval(str(qna)))

    return basic_qa
