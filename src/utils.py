import os
import re
import json
from tqdm import tqdm
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import boto3
from botocore.exceptions import BotoCoreError, ClientError

import os
import json
import boto3
import time
from botocore.exceptions import BotoCoreError, ClientError
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import AzureOpenAI


class BaseLLM(object):
    def __init__(self, llm_name):
        self.llm_name = llm_name
        llm_name_lower = llm_name.lower()

        # ---- Local Hugging Face models ----
        if llm_name_lower in ['llama3.1', 'llama3']:
            self.llm_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                "meta-llama/Meta-Llama-3.1-8B-Instruct", device_map="auto"
            )

        # ---- Azure GPT models ----
        elif llm_name_lower in ['gpt-4-turbo', 'gpt-4o']:
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="2024-05-01-preview",
            )

        # ---- AWS Bedrock models (Converse API) ----
        elif llm_name_lower.startswith("bedrock"):
            self.bedrock_client = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_REGION", "us-east-1")
            )
            self.bedrock_model_id = {
                "bedrock-claude": "anthropic.claude-3-sonnet-20240229-v1:0",
                "bedrock-titan": "amazon.titan-text-premier-v1:0",
                "bedrock-llama3": "meta.llama3-1-70b-instruct-v1:0",
            }.get(llm_name_lower, "anthropic.claude-3-sonnet-20240229-v1:0")

        else:
            raise ValueError(f"Unknown LLM name: {llm_name}")

    # ============================================================
    # Hugging Face local generation
    # ============================================================
    def __generate_LLM__(self, query, num_tokens_num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        input_ids = self.llm_tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.llm_model.device)

        self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        self.llm_model.config.pad_token_id = self.llm_model.config.eos_token_id

        outputs = self.llm_model.generate(
            input_ids,
            max_new_tokens=num_tokens_num,
            pad_token_id=self.llm_tokenizer.eos_token_id,
            eos_token_id=self.llm_tokenizer.eos_token_id,
        )
        response = outputs[0][input_ids.shape[-1]:]
        generated_text = self.llm_tokenizer.decode(response, skip_special_tokens=True)
        return generated_text

    # ============================================================
    # Azure GPT generation
    # ============================================================
    def __generate_GPT__(self, query, num_tokens_num):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ]
        try:
            response = self.client.chat.completions.create(
                model=self.llm_name,
                max_tokens=num_tokens_num,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print("Azure GPT error:", e)
            return "None"

    # ============================================================
    # Bedrock generation (Converse API)
    # ============================================================
    def __generate_Bedrock__(self, query, num_tokens_num):
        messages = [{"role": "user", "content": [{"text": query}]}]

        converse_params = {
            "modelId": self.bedrock_model_id,
            "messages": messages,
            "inferenceConfig": {
                "maxTokens": int(num_tokens_num),
                "temperature": 0.7,
                "topP": 0.9,
            },
        }

        # Retry logic
        for attempt in range(3):
            try:
                resp = self.bedrock_client.converse(**converse_params)
                message = (resp.get("output", {}) or {}).get("message", {})
                blocks = message.get("content", [])
                if blocks:
                    return "".join([b.get("text", "") for b in blocks])
                return ""
            except (BotoCoreError, ClientError, Exception) as e:
                print(f"[Bedrock Converse] attempt {attempt+1} failed:", e)
                time.sleep(2 ** attempt)

        print("Bedrock Converse failed after retries.")
        return "None"

    # ============================================================
    # Unified interface
    # ============================================================
    def generate(self, query, new_tokens_num=256):
        if self.llm_name.lower() in ['llama3.1', 'llama3']:
            return self.__generate_LLM__(query, new_tokens_num)
        elif self.llm_name.lower() in ['gpt-4-turbo', 'gpt-4o']:
            return self.__generate_GPT__(query, new_tokens_num)
        elif self.llm_name.lower().startswith("bedrock"):
            return self.__generate_Bedrock__(query, new_tokens_num)
        else:
            raise ValueError("Unsupported LLM backend.")


class QADataset:
    def __init__(self, data, dir="dataset/"):
        self.data = data.lower().split("_")[0]
        benchmark = json.load(open(os.path.join(dir, "benchmark.json")))
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        
        self.dataset = benchmark[self.data]
        self.index = sorted(self.dataset.keys())

    def __process_data__(self, key):
        data = self.dataset[self.index[key]]
        question = data["question"]
        choices = [v for k, v in data["options"].items()]

        options = [" A: ", " B: ", " C: ", " D: "]

        text = question + "\n"
        for j in range(len(choices)):
            text += "{} {}\n".format(options[j], choices[j])

        answer = data["answer"].strip()
        label_index = ord(answer) - ord('A')
        answer_content = choices[label_index]

        return {"text": text, "answer": answer, "answer_index": label_index, "answer_content": answer_content}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")
    
class CustomBioQALoader:
    """
    Loads your custom biomedical multi-choice dataset.
    Expects JSON or JSONL with fields:
      question, options, answer (optional).
    """
    def __init__(self, path):
        with open(path, 'r') as f:
            data = json.load(f) if path.endswith(".json") else [json.loads(l) for l in f]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for sample in self.data:
            q = sample["question"]
            opts = sample["options"]
            mc_text = q + "\n"
            for k, v in opts.items():
                mc_text += f"{k}. {v}\n"
            yield {
                "text": mc_text.strip(),
                "answer": sample.get("answer", "A")  # optional
            }
           
         
class MedDDxLoader:
    def __init__(self, data, dir="dataset/"):
        benchmark = self.process_dataset(dir)
        self.data = data
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        print(self.dataset)
        self.index = sorted(self.dataset.keys())
    
    def process_dataset(self, dir):       

        benchmark = json.load(open(os.path.join(dir, "MedDDx.json")))
        
        data_dict = {'MedDDx':{}, 'MedDDx-Basic':{}, 'MedDDx-Intermediate':{}, 'MedDDx-Expert': {}}

        for idx, b in enumerate(benchmark):
            if b['sim_level_std'] > 0.04:
                data_dict['MedDDx-Basic'][idx] = b
            elif b['sim_level_std'] < 0.02:
                data_dict['MedDDx-Expert'][idx] = b
            else:
                data_dict['MedDDx-Intermediate'][idx] = b
            data_dict['MedDDx'][idx] = b
        
        return data_dict
        
    def __process_data__(self, key):
        data = self.dataset[self.index[key]]

        answer = data["answer"].strip()
        label_index = ord(answer) - ord('A')
        
        return {"text": data['query'], "answer": answer, "answer_index": label_index}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")
            
class AfrimedLoader:
    def __init__(self, data='mcq_expert', dir="dataset/"):
        print("data is {}".format(data))
        if data == 'AfrimedQA-MCQ':
            self.data = 'mcq_expert'
        elif data == 'AfrimedQA-SAQ':
            self.data = 'saq_expert' 
        
        benchmark = self.process_dataset(dir)
        if self.data not in benchmark:
            raise KeyError("{:s} not supported".format(data))
        self.dataset = benchmark[self.data]
        print("{} has {} queries".format(data, len(self.dataset)))
        print(self.dataset)
        self.index = sorted(self.dataset.keys())
    
    def process_dataset(self, dir):       
        
        dataset_name = self.data
        datafile_name = "AfrimedQA_{}.json".format(dataset_name)

        print(datafile_name)

        if os.path.exists(os.path.join(dir, datafile_name)):
            dataset = json.load(open(os.path.join(dir, datafile_name)))
            return dataset
        else:
            from datasets import load_dataset
            options = [" A: ", " B: ", " C: ", " D: ", " E: ", " F: "]
            ds = load_dataset("intronhealth/afrimedqa_v2")['train']
            dataset = {dataset_name: {}}
            print("dataset is {}".format(dataset_name))
            
            for d in ds:
                print(d['question_type'])
                #if d['split'] == 'train':
                #    continue
                if d['tier'] != 'expert':
                    continue
                if d['question_type'] == 'mcq' and 'mcq' in dataset_name:
                    choices = [v for k, v in json.loads(d["answer_options"]).items()]
                
                    text = d['question_clean'].strip() + "\n"
                    for j in range(len(choices)):
                        text += "{} {}\n".format(options[j], choices[j])

                    label_index = int(d['correct_answer'][6])-1
                    answer = chr(ord('A') + label_index)
                    answer_content = choices[label_index]

                    idx = len(dataset['mcq_expert'])
                    dataset['mcq_expert'][idx] = {"query": text, "answer": answer, "answer_index": label_index, "answer_content": answer_content}
                if d['question_type'] == 'saq' and 'saq' in dataset_name:
                    
                    text = d['question_clean'].strip() + "\n"
                    answer = d['answer_rationale'].strip().replace('\n', ' ').replace('\r', '')

                    #label_index = int(d['correct_answer'][6])-1
                    #answer = chr(ord('A') + label_index)
                    #answer_content = choices[label_index]

                    idx = len(dataset['saq_expert'])
                    dataset['saq_expert'][idx] = {"query": text, "answer": answer, "answer_index": None, "answer_content": None}

            with open(os.path.join(dir, datafile_name), 'w') as f:
                json.dump(dataset, f, indent=2)

        return dataset

    def __process_data__(self, key):
        data = self.dataset[self.index[key]]

        answer = data["answer"].strip()
        if self.data == 'saq_expert':
            label_index = answer
        else:        
            label_index = ord(answer) - ord('A')
        
        return {"text": data['query'], "answer": answer, "answer_index": label_index}

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, key):
        if type(key) == int:
            return self.__process_data__(key)
        elif type(key) == slice:
            return [self.__getitem__(i) for i in range(self.__len__())[key]]
        else:
            raise KeyError("Key type not supported.")
