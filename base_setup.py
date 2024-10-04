from collections import defaultdict
import sys

sys.path.insert(0, "/Users/marina/Source/trn/question-extraction/lifeline_experiments")
from langchain_openai import ChatOpenAI
import pandas as pd
from dotenv import dotenv_values
import os
import logging

from tqdm import tqdm



from QA_system.retriever.qdrant_openai import QdrantOpenAIRetriever
from QA_system.generator.base_generator import BaseGenerator
from prompts import Prompts

class BaseSetup: 
   
   # BaseSetup ensures similar experimental setup for all experiments

    def  __init__(self, ds, llm, uses_ollama=False):
        """
        Initialize the BaseSetup class
        Input: user_questions: pandas dataframe with user questions
        expert_answers: pandas dataframe with expert answers
        llm: language model to use to generate answers
        retriever: retriever if retrieval is used
        """
        ENV = dotenv_values("../../.env")
        os.environ["OPENAI_API_KEY"] = ENV["OPENAI_API_KEY"]
        logging.basicConfig(filename='lifeline_experiments.log', level=logging.DEBUG, encoding='utf-8')

        self.logger = logging.getLogger(__name__)
        self.ds = ds
        self.user_questions = user_questions
        self.expert_answers = expert_answers
        self.prompts = Prompts()
        self.llm = llm
        self.summarizer_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=None)
        self.retriever = QdrantOpenAIRetriever(collection_name='apothekenumschau_1000_200', search_type='similarity_score_threshold', path_to_env="../../.env", local=True) 
        self.rag = BaseGenerator(self.retriever, self.llm)  
        self.uses_ollama = uses_ollama
    
    def get_results(self):
        random = pd.read_json("results/results_gpt-4o.json")
        longest = pd.read_json("results/results_gpt-4o_longest.json")
        shortest = pd.read_json("results/results_gpt-4o_shortest.json")
        return pd.concat([random, longest, shortest], ignore_index=True)
    
    def get_situation(self, text, questions):
        for question in questions:
            text = str(text).replace(question, '')
        return text
    
    def get_questions(self, user_question):
        return user_question['questions'].split(',')
    
    def summarize_situation(self, user_question):
        """
        Summarize the situation of the user question
        Input: user_question: row of pandas dataframe quesions
        Output: summary of the situation
        """
        if(user_question['conversation_id'] in self.result_df['conversation_id'].values):
            return self.result_df[self.result_df['conversation_id'] == user_question['conversation_id']]['summarized_situation'].values[0]

        prompt = self.prompts.get_summary_prompt(user_question['text'])
        return self.summarizer_llm.invoke(prompt).content
    
    def generate_summarized_question(self, user_question):
        """
        Generate a question from the full user question.
        Input: user_question: row of pandas dataframe quesions
        Output: summarized question
        """
        summary = self.summarize_situation(user_question)
        prompt = self.prompts.get_question_from_summary_prompt(summary)

        if(user_question['conversation_id'] in self.result_df['conversation_id'].values):
            return self.result_df[self.result_df['conversation_id'] == user_question['conversation_id']]['summarized_question'].values[0]

        return self.summarizer_llm.invoke(prompt).content
    
    def ask_llm(self, prompt):
        """
        Ask the llm a question
        Input: prompt: question to ask
        Output: llm answer
        """
        return self.llm.invoke(prompt)
    
    def ask_RAG(self, question):
        """
        Ask the RAG model a question
        Input: question: question to ask
        Output: RAG answer
        """
        return self.rag.ask_rag(question)
    
    def run_prompt_safely(self,func, prompt, conv_id):
        try: 
            def inner():
                if(self.uses_ollama):
                    return func(prompt)
                return func(prompt).content
            
            return inner()
        except Exception as e:
            self.logger.error(f"Error getting answer for conversation {conv_id} for prompt:  {prompt}", e)
    
    def run_single_llm_experiment(self, row):
        """
        Run a single experiment based on a user_question
        1. Ask LLM directly 
            a. Full query
            b. Context + Query split
            c. Summary + Query split
            d. reformulated query based on summarized context
        2. Ask RAG model with similar query setups
        Input: user_question: row of pandas dataframe quesions
        Output: List containing all values of results dataframe row: 
                [user_question, expert_answer, user_context, summarized_situation, summarized_question, llm0...llm3, rag0...rag3]
        """
        question = row['user_question']
        expert_answer = row
        user_context = self.get_situation(question, self.get_questions(row))
        summarized_situation = self.summarize_situation(row)
        summarized_question = self.generate_summarized_question(row)
        conv_id = row['conversation_id']

        llm_answer = self.run_prompt_safely(self.ask_llm, self.prompts.get_plain_prompt(question), conv_id )
        llm_answer_split = self.run_prompt_safely(self.ask_llm, self.prompts.get_split_prompt(user_context, question), conv_id)
        llm_answer_summary = self.run_prompt_safely(self.ask_llm, self.prompts.get_split_prompt(summarized_situation, summarized_question), conv_id)
        llm_answer_reformulated = self.run_prompt_safely(self.ask_llm, self.prompts.get_plain_prompt(summarized_question), conv_id)

        return [conv_id, question, expert_answer, user_context, summarized_situation, summarized_question, 
                llm_answer, llm_answer_split, llm_answer_summary, llm_answer_reformulated]

        
    def run_experiment(self, batch_size=10, save_batch="", start_from_batch=0):
        """
        Run all experiments
        Ask LLM directly:
            a. Full query
            b. Context + Query split
            c. Summary + Query split
            d. reformulated query based on summarized context
        Input: batch_size: number of questions to process at once
        Output: Pandas dataframe with results
        """
        results = defaultdict(list)
        keys = ['conversation_id', 'user_question', 'expert_answer', 'user_context', 'summarized_situation', 
                'summarized_question', 'llm0', 'llm1', 'llm2', 'llm3']
        for i in range(start_from_batch*batch_size, len(self.ds), batch_size):
            print(f"Processing batch {i} to {i+batch_size}")
            user_questions = self.ds.iloc[i:i+batch_size]
            for idx, user_question in tqdm(user_questions.iterrows()):
                result = self.run_single_llm_experiment(user_question)
                for key, value in zip(keys, result):
                    results[key].append(value)
            if save_batch != "":
                os.makedirs(f"results/{save_batch}", exist_ok=True)
                pd.DataFrame(results).to_csv(f"results/{save_batch}/results_batch_{i}_{i+batch_size}.csv", sep=";")
        
        if start_from_batch != 0:
            results_to_append = pd.read_csv(f"results/{save_batch}/results_batch_{(start_from_batch-1)*batch_size}_{(start_from_batch)*batch_size}.csv", sep=";")
            return pd.concat([results_to_append, pd.DataFrame(results)], ignore_index=True)
        return pd.DataFrame(results)

        


    

    

