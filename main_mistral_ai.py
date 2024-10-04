import os
from dotenv import dotenv_values
from langchain_mistralai import ChatMistralAI
from base_setup import BaseSetup
import pandas as pd
from pprint import pprint


# main
def main():
    # Load data
    shortest_ds = pd.read_json("datasets/shortest_queries.json")

    # Initialize the BaseSetup class
    ENV = dotenv_values("../../.env")
    os.environ["MISTRAL_API_KEY"] = ENV["MISTRAL_API_KEY"]
    llm = ChatMistralAI(
        model="mistral-large-latest",
        temperature=0,
        max_retries=4,
        # other params...
    )
    base_setup = BaseSetup(shortest_ds, llm)

    # start experiments
    results = base_setup.run_experiment(batch_size=10, save_batch="mistral_largest_latest/shortest")
    results.to_csv("results/results_mistral_largest_latest_shortest.csv", sep=";")

    second_llm = ChatMistralAI(
        model="open-mixtral-8x22b",
        temperature=0,
        max_retries=4,
        # other params...
    )
    base_setup2 = BaseSetup(shortest_ds, second_llm)
    results2 = base_setup2.run_experiment(batch_size=10, save_batch="open_mixtral_8x22b/shortest")  
    results2.to_csv("results/results_open_mixtral_8x22b_shortest.csv", sep=";")

if __name__ == "__main__":
    main()
