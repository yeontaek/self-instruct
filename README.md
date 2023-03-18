# Self-Instruct by Stanford Alpaca

## Overview 

- Stanford Alpaca 모델에서 Self-Instruct 부분만 따로 가져와서 원하는 만큼의 instruction 데이터만 생성할 수 있도록 함

## Setup

- python 버전은 3.9 이상에서 동작하도록 되어 있어서 3.9 이상을 사용해야 된다. 

```
pip install -r requirements.txt
```

## Data Generation Process

- Prompt.txt 파일에 명시적으로 task 생성 조건을 정의하였다. 
> Here are the requirements:
> 1. Try not to repeat the verb for each instruction to maximize diversity.
> 2. The language used for the instruction also should be diverse. For example, you should combine questions with imperative instrucitons.
> 3. The type of instructions should be diverse. The list should include diverse types of tasks like open-ended generation, classification, editing, etc.

- 기존 Stanford Alpaca 모델에서는 text-davinci-003 모델을 사용하여, task를 대량으로 생산하였다. 
- **text-davinci-003** 보다는 **gpt-3.5-turbo**, **gpt-3.5-turbo-0301** 을 사용하는 것이 비용이나 속도 측면에서 더 좋다.
> We used text-davinci-003 to generate the instruction data instead of davinci.

- 최종적으로 생성된 데이터는 아래와 같이 생성되게 됨
- most_similar_instructions은 기존에 seed_tasks.jsonl에서 정의한 instruction가 얼마나 유사한지를 나타낸다.
- 학습에는 most_similar_instructions,  avg_similarity_score을 제거하고 사용하면 된다. 

```
{
        "instruction": "Describe the architecture of a neural network.",
        "input": "",
        "output": "A neural network is composed of a set of layers, each of which performs a specific type of computation. Generally, the input layer of a neural network takes the input data, which can be images, audio, or text. These inputs are then sent to hidden layers, which each perform a specific type of computation, such as convolutional or pooling layers for images and recurrent or normal layers for audio and text. The output of the hidden layers is then sent to the output layer, which produces the output of the neural network. The weights of the neurons in the various layers determine their contribution to the model's output. The neural network is then trained to adjust its weights in order to better understand the data and produce better results. Training a neural network involves feeding it data, adjusting its weights, and evaluating its performance. The training continues until the performance of the model has reached the desired level.",
        "most_similar_instructions": {
            "Explain the ending of given movie.": 0.30769230769230765,
            "Give me the address of this person.": 0.2857142857142857,
            "Give me the definition of the word.": 0.2857142857142857,
            "Come up with the synonyms of the word.": 0.26666666666666666,
            "Explain the social stereotype of the following profession.": 0.26666666666666666,
            "Using the facts given, write a question-answer pair.": 0.25,
            "Fill in the blank with a word or phrase.": 0.25,
            "Write a list of questions to start a conversation.": 0.25,
            "write a code to get the current value of a Bitcoin in US dollars using the bitcoincharts API.": 0.24,
            "Make a list of salty snacks with a soft texture.": 0.23529411764705882
        },
        "avg_similarity_score": 0.12367607822364603
    },
    {
        "instruction": "Compose a poem about death.",
        "input": "",
        "output": "Death is a strange thing, so fleeting and so fast;\nIt tricks you into loving, but you're soon left in the past.\n\nIt takes those we love, with no second thought;\nLeaving us feeling lost, unfulfilled and distraught.\n\nIt marks the end of one life, and the start of the next;\nBringing with it peace, and wisdom to those who reflect.\n\nLife will carry on, but not how it used to be;\nFor death will continue to take, though it sets us free.",
        "most_similar_instructions": {
            "Write a knock knock joke about bananas.": 0.3333333333333333,
            "Create a survey that asks about eating and cooking habits.": 0.26666666666666666,
            "Given the facts, compose them into a coherent and fascinating story.": 0.25000000000000006,
            "Give me a joke about PhD. Try to make a long joke.": 0.23529411764705882,
            "Create a birthday planning checklist.": 0.20000000000000004,
            "Plan a syllabus for the the class.": 0.16666666666666666,
            "Summarize this email into a single sentence:": 0.16666666666666666,
            "Describe the architecture of a neural network.": 0.16666666666666666,
            "Generate a haiku using the following word:": 0.16666666666666666,
            "Design a chess puzzle in FEN Notation.": 0.16666666666666666
        },
        "avg_similarity_score": 0.06340385974896465
    },
```

## 실행 

```
python generate_instruction.py --api_key your_openai_api_key --model_name gpt-3.5-turbo-0301
```


## Refrence
- https://github.com/tatsu-lab/stanford_alpaca
- https://github.com/tloen/alpaca-lora