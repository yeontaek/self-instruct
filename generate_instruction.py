
import time
import json
import os
import random
import re
import string
from functools import partial
from multiprocessing import Pool

import numpy as np
import tqdm
from rouge_score import rouge_scorer
import utils
import argparse


# prompt에 정의하 조건들과 사람이 정의한 instrcution 셋을 기준으로 N개의 task를 추가 생성하는 함수
def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""

    # prompt.txt에는 task 생성 조건이 정의된다. 그리고 생성할 task 숫자도 정의된다.
    prompt = open("./prompt.txt").read() + "\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"

    # 최종적으로는 [생성 조건 + 사람의 생성한 instruction]을 결합해서 최대 20개의 task를 생성하는 프롬프트 작성
    return prompt


def post_process_gpt3_response(num_prompt_instructions, response):
    if response is None:
        return []
    raw_instructions = f"{num_prompt_instructions+1}. Instruction:" + response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue
        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Instruction|Input|Output):", inst)
        if len(splitted_data) != 7:
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            "image",
            "images",
            "graph",
            "graphs",
            "picture",
            "pictures",
            "file",
            "files",
            "map",
            "maps",
            "draw",
            "plot",
            "go to",
            "video",
            "audio",
            "music",
            "flowchart",
            "diagram",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions


def find_word_in_string(w, s):
    return re.compile(r"\b({0})\b".format(w), flags=re.IGNORECASE).search(s)


def generate_instruction_following_data(args):

    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]

    seed_instruction_data = [
        {"instruction": t["instruction"], "input": t["instances"][0]["input"], "output": t["instances"][0]["output"]}
        for t in seed_tasks
    ]

    # 사람이 정의한 instruction 데이터들을 불러온다.
    # 미리 스탠포드에서 정의한 데이터 셋 175개 존재한다
    print(f"Loaded {len(seed_instruction_data)} human-written seed instructions")

    os.makedirs(args.output_dir, exist_ok=True)
    request_idx = 0

    # load the LM-generated instructions
    machine_instruction_data = []
    if os.path.exists(os.path.join(args.output_dir, "regen.json")):
        machine_instruction_data = utils.jload(os.path.join(args.output_dir, "regen.json"))
        print(f"Loaded {len(machine_instruction_data)} machine-generated instructions")

    # similarities = {}
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)

    # now let's generate new instructions!
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    if machine_instruction_data:
        progress_bar.update(len(machine_instruction_data))

    # 사람이 작성한 instruction과 machine이 작성한 instruction만을 모아서 저장
    all_instructions = [d["instruction"] for d in seed_instruction_data] + [
        d["instruction"] for d in machine_instruction_data
    ]

    # 저장된 instrcution data를 tokenizer 한다.
    all_instruction_tokens = [scorer._tokenizer.tokenize(inst) for inst in all_instructions]

    # 사람이 생성한 instrcution보다 기계가 생성한 수가 많을 때까지 while문을 돌린다.
    while len(machine_instruction_data) < args.num_instructions_to_generate:
        request_idx += 1

        batch_inputs = []

        # request_batch_size만큼 숫자의 프롬프트 문구를 생성한다.
        for _ in range(args.request_batch_size):

            # 사람이 생성한 instruction 셋(seed_instruction_data)에서 랜덤하게 num_prompt_instructions 수만큼 뽑는다.
            prompt_instructions = random.sample(seed_instruction_data, args.num_prompt_instructions)

            #  [생성 조건 + 사람의 생성한 instruction]을 결합해서 최대 20개의 task를 생성하는 프롬프트 작성하는 함수
            prompt = encode_prompt(prompt_instructions)
            batch_inputs.append(prompt)

        decoding_args = utils.OpenAIDecodingArguments(
            temperature=args.temperature,
            n=1,
            max_tokens=3072,  # hard-code to maximize the length. the requests will be automatically adjusted
            top_p=args.top_p,
            stop=["\n20", "20.", "20."],
        )
        request_start = time.time()
        results = utils.openai_completion(
            prompts=batch_inputs,
            model_name=args.model_name,
            batch_size=args.request_batch_size,
            decoding_args=decoding_args,
            api_key=args.api_key,
            logit_bias={"50256": -100},  # prevent the <|endoftext|> token from being generated
        )
        request_duration = time.time() - request_start

        process_start = time.time()
        instruction_data = []
        for result in results:
            new_instructions = post_process_gpt3_response(args.num_prompt_instructions, result)
            instruction_data += new_instructions

        total = len(instruction_data)
        keep = 0
        for instruction_data_entry in instruction_data:

            # computing similarity with the pre-tokenzied instructions
            new_instruction_tokens = scorer._tokenizer.tokenize(instruction_data_entry["instruction"])
            with Pool(args.num_cpus) as p:
                rouge_scores = p.map(
                    partial(rouge_scorer._score_lcs, new_instruction_tokens),
                    all_instruction_tokens,
                )
            rouge_scores = [score.fmeasure for score in rouge_scores]
            most_similar_instructions = {
                all_instructions[i]: rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
            }
            if max(rouge_scores) > 0.7:
                continue
            else:
                keep += 1

            instruction_data_entry["most_similar_instructions"] = most_similar_instructions
            instruction_data_entry["avg_similarity_score"] = float(np.mean(rouge_scores))
            machine_instruction_data.append(instruction_data_entry)
            all_instructions.append(instruction_data_entry["instruction"])
            all_instruction_tokens.append(new_instruction_tokens)
            progress_bar.update(1)
        process_duration = time.time() - process_start
        print(f"Request {request_idx} took {request_duration:.2f}s, processing took {process_duration:.2f}s")
        print(f"Generated {total} instructions, kept {keep} instructions")
        utils.jdump(machine_instruction_data, os.path.join(args.output_dir, "regen.json"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='generate self-instruction')

    parser.add_argument('--output_dir', default="./dataset")
    parser.add_argument('--seed_tasks_path', default="./seed_tasks.jsonl")
    parser.add_argument('--num_instructions_to_generate', default=1)
    parser.add_argument('--model_name', default="text-davinci-003")
    parser.add_argument('--num_prompt_instructions', default=3)
    parser.add_argument('--request_batch_size', default=1)
    parser.add_argument('--temperature', default=1.0)
    parser.add_argument('--top_p', default=1.0)
    parser.add_argument('--num_cpus', default=16)
    parser.add_argument('--api_key', default="")

    args = parser.parse_args()

    generate_instruction_following_data(args)