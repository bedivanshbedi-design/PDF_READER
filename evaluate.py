from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import json

import pandas as pd

with open("eval_data.json","r" ) as f:
  data = json.load(f)

dataset = Dataset.from_list(data)

dataset = evaluate(
          dataset,
          metrics=[
              faithfulness,
              answer_relevancy,
              context_precision,
              context_recall
          ]
)

print("Evaluation results:\n")
print(result)

#Save results

df=result.to_pandas()
df.to_csv("ragas_results.csv",index=false)

print("\n Results saved to ragas_results.csv")



