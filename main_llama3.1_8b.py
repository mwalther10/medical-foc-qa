import os
from dotenv import dotenv_values
from langchain_community.llms.ollama import Ollama
from base_setup import BaseSetup
import pandas as pd
from pprint import pprint

# main
def main():
    # Load data
    user_questions = pd.read_csv('../lifeline_embedded_clustered.csv', sep=';')
    user_questions.drop(columns=['hdbscan_clusters_unreduced','hdbscan_clusters_reduced', 'kmeans_clusters_unreduced','kmeans_clusters_reduced', 'questions_embedding' , 'text_embedding' ], inplace=True)
    first_100 = user_questions.head(100)
    shortest_queries = pd.read_csv("./shortest_queries.csv", sep=";")
    print(shortest_queries.head())
    longest_queries = pd.read_csv("./longest_queries.csv", sep=";")
    expert_answers = pd.read_csv('../answers.csv', sep=';')

    # Initialize the BaseSetup class
    ENV = dotenv_values("../../.env")
    llm = Ollama(model='llama3.1:8b', temperature=0)
    base_setup = BaseSetup(longest_queries, expert_answers, llm, uses_ollama=True)

    # start experiments
    results = base_setup.run_experiment(batch_size=10, save_batch="llama3.1_8b/longest", start_from_batch=0)
    results.to_csv("results/results_llama3.1_8b_longest.csv", sep=";")

if __name__ == "__main__":
    main()