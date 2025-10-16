import argparse
import os
from transformers import set_seed
import json
from tqdm import tqdm
import logging
from src.utils import QADataset, MedDDxLoader, BaseLLM, AfrimedLoader, CustomBioQALoader
from action.generate import Generate
from action.review import Review
from action.answer import Answer

set_seed(42)


class KGARevion(object):
    def __init__(self,
                 args
                 ):
        super().__init__()
        self.agent_name = "KGARevion"
        self.role = """You can answer questions by choosing Extract_Triplets, KnowledgeGraph_Classifier and Answer_Generator actions. Finish it if you find answer."""
        self.args = args
        self.llm = BaseLLM(args.llm_name)
        self.triplets_generator = Generate(self.llm, args)
        if args.llm_name == 'gpt-4-turbo':
            self.review_llm = BaseLLM('llama3.1')
        else:
            self.review_llm = self.llm
        self.classifier = Review(self.review_llm, args)
        self.answer_generator = Answer(self.llm)

    
    def call(self, query):
        logging.info(query)
        print("query")
        generated_triplets, mt = self.triplets_generator.call(query)
        print(generated_triplets)
        filtered_triplets, score = self.classifier.call(generated_triplets, query)
        answer = self.answer_generator.call(filtered_triplets, query)
        logging.info("filtered_triplets are {}".format(filtered_triplets))
        logging.info(answer)
       
        return answer

def main(args):

    ##load llm
    set_seed(42)

    import gc
    gc.collect()

    logging.basicConfig(filename = args.dataset + "_test_case_study_multi-choice.log", level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    bioKG_agent = KGARevion(args=args)

    ##load data
    if args.dataset == 'custom':
        data = CustomBioQALoader(args.data_path)
    elif args.dataset in ['MedDDx', 'MedDDx-Basic', 'MedDDx-Intermediate', 'MedDDx-Expert']:
        data = MedDDxLoader(args.dataset)
    elif args.dataset in ['AfrimedQA-MCQ']:
        data = AfrimedLoader(args.dataset)
    else:
        data = QADataset(args.dataset)
   
    accurate_sample_idx = []
    response_all = []
    
    for idx, d in tqdm(enumerate(data),  total=len(data), desc=f'Evaluating data'):
        
        if 'text' in d and 'answer' in d:
            query = d['text']
            label = d['answer']
        
        response = bioKG_agent.call(query)
        response = response.strip().replace('\n', '').replace('\"', '')
        
        logging.info(f'Response: {response}, Label: {label}')
           
        predict_answer = 'None'
        if "Answer: " in response:
            answer_index = response.find("Answer: ")
            predict_answer = response[answer_index + len("Answer: "):].strip()[0]
        
        
        if predict_answer not in ['A', 'B', 'C', 'D', 'E'] and label in response:
            predict_answer = label
        predict_answer = predict_answer.strip()

        logging.info("predict_answer: {} and correct answer: {}".format(predict_answer, label))
        
        if predict_answer == label:
            accurate_sample_idx.append(idx)
       
        response_all.append(response[answer_index + len("Answer: "):].strip())

        
    if args.type == 'SAQ':
        from rouge_score import rouge_scorer
        correct_predictions = []
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        for idx, r in enumerate(response_all):
            results = scorer.score(response, data[idx]['answer'])
            correct_predictions.append([results['rouge1'].precision, results['rouge1'].recall, results['rouge1'].fmeasure])    
        import numpy as np
        correct_predictions = np.array(correct_predictions)
        print(correct_predictions)
        print(correct_predictions.shape)
        correct_predictions = np.sum(correct_predictions, axis=0)
        print(correct_predictions.shape)
        metrics_value = correct_predictions / len(response_all)
        print("metrics_value!!")
        print(metrics_value)
        print(f"Rouge: {metrics_value[0]:.2%} {metrics_value[1]:.2%} {metrics_value[2]:.2%}")   
    elif args.type == 'MCQ':
        accuracy = len(accurate_sample_idx)/len(response_all)
        print(accuracy)
        print(len(accurate_sample_idx))
        metrics_value = accuracy
    
    with open("results/" + args.dataset + "_" + args.llm_name + "_is_Revision_" + str(args.is_revise) + "_round_" + str(args.max_round) + "_all.txt", 'w') as f:
        json.dump(args.__dict__, f, indent=2)
        f.write('\n')
        f.write("accuracy: {}".format(metrics_value))
        f.write('\n')
        f.write("correct task id: ")
        for a in accurate_sample_idx:
            f.write(str(a))
            f.write('\t')
        f.write('\n')
        for idx, r in enumerate(response_all):
            f.write(str(idx))
            f.write(': ')
            f.write(str(r))
            f.write('\n')

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='MedDDx', choices=['mmlu', 'medqa', 'pubmedqa', 'bioasq', 'MedDDx', 'MedDDx-Basic', 'MedDDx-Intermediate', 'MedDDx-Expert', 'afrimedqa_v2', 'AfrimedQA-SAQ', "custom"], type=str)
    parser.add_argument("--key", type=str)
    parser.add_argument("--data_path", type = str)
    parser.add_argument("--type", type=str, default='MCQ', choices=['MCQ', 'SAQ'])
    parser.add_argument("--max_round", type=int, default=1)
    parser.add_argument("--is_revise", type=bool, default=True)
    parser.add_argument("--KG_name", default='primeKG', choices=['UMLS', 'primeKG', 'ogb-biokg'], type=str)
    parser.add_argument("--llm_name", default='llama3.1', choices=['llama3.1', 'llama3', 'gpt-4-turbo', 'llama3.1-70', "bedrock-llama3"], type=str)
    parser.add_argument("--weights_path", type=str, default='fine_tuned_model/')
    args = parser.parse_args()
    main(args)

