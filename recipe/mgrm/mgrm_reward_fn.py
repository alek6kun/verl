# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import re

import aiohttp
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PREPROCESS_INPUT_PROMPT_WITH_GROUND_TRUTH = """# Task Description
You are an expert evaluator assessing the quality of AI-generated responses.
Your task is to compare a response against the ground truth answer and provide a quality score.
You shouldn't try to resolve the question, your role is to verify the answer after giving a short reasoning for the verification.

# Instructions
1. First, provide a brief reasoning (2-3 sentences) explaining your evaluation. You start with a score of 1.0/1.0 Consider factors such as:
   - Correctness: Does the response match the ground truth? If the response doesn't match, you must give a score of 0.0/1.0. If the responses match, keep the score of 1.0/1.0.
   - Completeness: Does it cover all key points, while not giving irrelevant information? Remove 0.05 per key point missed or irrelevant information given.
   - Clarity: Is the explanation clear and well-structured? Remove 0.1 if the explanation is unclear or not well-structured.
2. Then, provide a numerical score between 0.0 and 1.0 on the last line. Do not provide any additional explanation.
3. Use the exact format: "Final score: X.X/1.0"

# Example
Question: What is the capital of France?
Response: The capital of France is Paris, which is located in the north-central part of the country.
Ground Truth: Paris

Reasoning:
The response correctly identifies Paris as the capital of France. It also provides additional context about the location, which adds value without introducing any errors. The answer is accurate and complete.
Final score: 1.0/1.0

# Evaluation Task
[Beginning of question]
Question: {problem}
[End of question]
[Beginning of response]
Response: {solution}
[End of response]
[Beginning of ground truth]
Ground Truth: {ground_truth}
[End of ground truth]

Now provide your evaluation:"""

PREPROCESS_INPUT_PROMPT_WITHOUT_GROUND_TRUTH = """# Task Description
You are an expert evaluator assessing the quality of AI-generated responses.
Your task is to compare a response against the ground truth answer and provide a quality score.
You shouldn't try to resolve the question, your role is to verify the answer after giving a short reasoning for the verification.

# Instructions
1. First, provide a brief reasoning (2-3 sentences) explaining your evaluation. Consider factors such as:
   - Correctness: Is the information accurate?
   - Completeness: Does it cover all key points, while not giving irrelevant information?
   - Clarity: Is the explanation clear and well-structured?
2. Then, provide a numerical score between 0.0 and 1.0 on the last line. Do not provide any additional explanation.
3. Use the exact format: "Final score: X.X/1.0", without any additional character or emphasis.

# Example
Question: What is the capital of France?
Response: The capital of France is Paris, which is located in the north-central part of the country.

Reasoning:
The response correctly identifies Paris as the capital of France. It also provides additional context about the location, which adds value without introducing any errors. The answer is accurate and complete.
Final score: 1.0/1.0

# Evaluation Task
[Beginning of question]
Question: {problem}
[End of question]
[Beginning of response]
Response: {solution}
[End of response]

Now provide your evaluation:"""

_SOLUTION_CLIP_CHARS = 300

def verify(
    solution_str: str,
    gt: str,
) -> tuple[bool, str]:
    # Optimization: Regular expression matching on very long strings can be slow.
    # For math problems, the final answer is usually at the end.
    # We only match on the last 300 characters, which is a safe approximation for 300 tokens.
    if len(solution_str) > _SOLUTION_CLIP_CHARS:
        solution_str = solution_str[-_SOLUTION_CLIP_CHARS:]

    final_answer = solution_str.split('####')[-1].strip()

    for remove_char in [',', '$', '%', 'g']:
        final_answer = final_answer.replace(remove_char, '')

    return (final_answer == gt), final_answer

FLAWED_REWARD_PENALTY = 1.0

async def generate_aiohttp(router_address: str, text: str):
    payload = {
        "text": text,
    }
    url = f"http://{router_address}/generate"
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with session.post(url, json=payload) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                text = output["text"]
                return text
            except Exception as e:
                logger.info(f"Error: {e}. Output: {output}")
                return {}
    finally:
        await session.close()

async def chat_completions_aiohttp(router_address: str, chat_complete_request: dict):
    url = f"http://{router_address}/v1/chat/completions"
    try:
        session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
        async with session.post(url, json=chat_complete_request) as resp:
            output = await resp.text()
            try:
                output = json.loads(output)
                text = output["choices"][0]["message"]["content"]
                return text
            except Exception as e:
                logger.info(f"Error: {e}. Output: {output}")
                return {}
    finally:
        await session.close()


async def compute_score_mgrm(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: str,
    reward_model_tokenizer: PreTrainedTokenizer,
    reward_model_name: str,
):
    """Compute the reward score."""
    question, split = extra_info["question"], extra_info["split"]
    correct, pred = verify(solution_str, ground_truth)
    reward_score = 1.0 if correct else 0
    is_flawed_positive = False

    # for test set, directly return the reward score
    if split == "test":
        return {"score": reward_score, "acc": correct, "pred": pred, "is_flawed_positive": is_flawed_positive}

    grm_prompt = PREPROCESS_INPUT_PROMPT_WITH_GROUND_TRUTH.format(
        problem=question,
        ground_truth=ground_truth,
        solution=solution_str,
    )
    # grm_response = await generate_aiohttp(reward_router_address, text=grm_prompt)
    chat_complete_request = {
        "model": reward_model_name,
        "messages": [{"role": "user", "content": grm_prompt}],
    }
    grm_response = await chat_completions_aiohttp(reward_router_address, chat_complete_request)

    if grm_response is not None:
        try:
            grm_score = postprocess_fn(grm_response)
        except Exception as e:
            logger.info(f"Error: {e}. Output: {grm_response}")
            return {"score": 0, "acc": correct, "pred": pred, "is_flawed_positive": 0}
    return {"score": grm_score, "acc": correct, "pred": pred, "is_flawed_positive": (grm_score != reward_score)}

def postprocess_fn(output_text: str) -> float:
    """
    Postprocess generative reward model output to extract a numerical score.
    
    Args:
        output_text: The text output from the generative reward model
    
    Returns:
        Numerical score (float) between 0.0 and 1.0
    """    
    # Look for the "Final score: X.X/1.0" pattern in the last lines
    lines = output_text.strip().split('\n')
    
    patterns = [
        r'final\s+score[:\s]+([0-9]*\.?[0-9]+)\s*/\s*1\.0',  # Final score: 0.8/1.0
        r'score[:\s]+([0-9]*\.?[0-9]+)\s*/\s*1\.0',          # Score: 0.8/1.0
        r'([0-9]*\.?[0-9]+)\s*/\s*1\.0',                     # 0.8/1.0 (standalone)
        r'final\s+score[:\s]+([0-9]*\.?[0-9]+)',             # Final score: 0.8
        r'score[:\s]+([0-9]*\.?[0-9]+)',                     # Score: 0.8
        r'\b([0-1]\.?[0-9]*)\b',                             # 1.0, 0.5, 0.0 (standalone)
    ]
    
    for pattern in patterns:
        for line in reversed(lines):
            match = re.search(pattern, line.lower())
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))

    
    # Default: return 0.0 if no score found
    print(f"Warning: Could not extract score from output: {output_text[:100]}, ... {output_text[100:]}")
    return 0.0