import os
from dotenv import dotenv_values
from langchain_mistralai import ChatMistralAI
from base_setup import BaseSetup
import pandas as pd
from pprint import pprint


# main
def main():
    # Load data
    user_questions = pd.read_csv("../lifeline_embedded_clustered.csv", sep=";")
    user_questions.drop(
        columns=[
            "hdbscan_clusters_unreduced",
            "hdbscan_clusters_reduced",
            "kmeans_clusters_unreduced",
            "kmeans_clusters_reduced",
            "questions_embedding",
            "text_embedding",
        ],
        inplace=True,
    )
    first_100 = user_questions.head(100)
    shortest_queries = pd.read_csv("./shortest_queries.csv", sep=";")
    expert_answers = pd.read_csv("../answers.csv", sep=";")

    # Initialize the BaseSetup class
    ENV = dotenv_values("../../.env")
    os.environ["MISTRAL_API_KEY"] = ENV["MISTRAL_API_KEY"]
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0,
        max_retries=4,
        # other params...
    )
    base_setup = BaseSetup(shortest_queries, expert_answers, llm)

    # start experiments
    results = base_setup.run_experiment(batch_size=10, save_batch="mistral_largest_latest/shortest")
    results.to_csv("results/results_mistral_largest_latest_shortest.csv", sep=";")

    second_llm = ChatMistralAI(
        model="open-mixtral-8x22b",
        temperature=0,
        max_retries=4,
        # other params...
    )
    base_setup2 = BaseSetup(shortest_queries, expert_answers, second_llm)
    results2 = base_setup2.run_experiment(batch_size=10, save_batch="open_mixtral_8x22b/shortest")  
    results2.to_csv("results/results_open_mixtral_8x22b_shortest.csv", sep=";")

if __name__ == "__main__":
    main()
