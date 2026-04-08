from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import json

import pandas as pd

# with open("eval_data.json","r" ) as f:
#   data = json.load(f)

# dataset = Dataset.from_list(data)

def run_evaluate(dataset):
  dataswt = Dataset.from_list(data)

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






