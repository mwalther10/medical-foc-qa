import os
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from base_setup import BaseSetup
import pandas as pd
from pprint import pprint

# main
def main():
    # Load data
    user_questions = pd.read_csv('../lifeline_embedded_clustered.csv', sep=';')
    user_questions.drop(columns=['hdbscan_clusters_unreduced','hdbscan_clusters_reduced', 'kmeans_clusters_unreduced','kmeans_clusters_reduced', 'questions_embedding' , 'text_embedding' ], inplace=True)
    
    expert_answers = pd.read_csv('../answers.csv', sep=';')
    shortest_queries = pd.read_csv("./shortest_queries.csv", sep=";")
    longest_queries = pd.read_csv("./longest_queries.csv", sep=";")

    # Initialize the BaseSetup class
    ENV = dotenv_values("../../.env")
    os.environ["OPENAI_API_KEY"] = ENV["OPENAI_API_KEY"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=None)
    base_setup = BaseSetup(shortest_queries, expert_answers, llm)

    # start experiments
    results = base_setup.run_experiment(batch_size=10, save_batch="gpt-4o-mini/shortest")
    results.to_csv("results/results_gpt-4o-mini_shortest.csv", sep=";")

if __name__ == "__main__":
    main()