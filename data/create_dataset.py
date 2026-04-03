from datasets import load_dataset

dataset = load_dataset("squad")

subset = dataset["validation"].select(range(1000))

import json

with open("/Users/harmansingh/AI Engineering/AI/GenAI/Agent Benchmark/Data/squad_subset.json", "w") as f:
    json.dump(subset.to_list(), f, indent=2)



dataset = load_dataset("hotpot_qa", "distractor")

subset = dataset["validation"].select(range(1000))

import json

with open("/Users/harmansingh/AI Engineering/AI/GenAI/Agent Benchmark/Data/hotpot_subset.json", "w") as f:
    json.dump(subset.to_list(), f, indent=2)
