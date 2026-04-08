from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import json

import pandas as pd

# with open("eval_data.json","r" ) as f:
#   data = json.load(f)

# dataset = Dataset.from_list(data)

def run_evaluate(dataset):

  clean_data = []

  for d in dataset:
    if not isinstance(d,dict):
      continue

    question = str(d.get("question", "")).strip()
    answer = str(d.get("answer", "")).strip()
    contexts = d.get("contexts", [])
    ground_truth = str(d.get("ground_truth", "")).strip()

    # ensure contexts is list of strings
    if not isinstance(contexts, list):
        contexts = [str(contexts)]
    contexts = [str(c) for c in contexts if c]

      # skip bad rows
      if not question or not answer or not contexts:
          continue

      clean_data.append({
          "question": question,
          "answer": answer,
          "contexts": contexts,
          "ground_truth": ground_truth if ground_truth else contexts[0]
        })

  # 🚨 CRITICAL CHECK
  if len(clean_data) == 0:
      raise ValueError("All eval_data rows are invalid. Nothing to evaluate.")

  dataset = Dataset.from_list(data)

  result = evaluate(
          dataset,
          metrics=[
              faithfulness,
              answer_relevancy,
              context_precision,
              context_recall
          ]
)


  df=result.to_pandas()
  return result, df






