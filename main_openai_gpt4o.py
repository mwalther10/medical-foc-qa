import os
from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from base_setup import BaseSetup
import pandas as pd

# main
def main():
    # Load data
    ds = pd.read_json("datasets/shortest_queries.json")

    # Initialize the BaseSetup class
    ENV = dotenv_values("../../.env")
    os.environ["OPENAI_API_KEY"] = ENV["OPENAI_API_KEY"]
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=None)
    base_setup = BaseSetup(ds, llm)

    # start experiments
    results = base_setup.run_experiment(batch_size=10, save_batch="gpt-4o-mini/shortest")
    results.to_csv("results/results_gpt-4o-mini_shortest.csv", sep=";")

if __name__ == "__main__":
    main()