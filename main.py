from bert_score import score

# Example texts
references = ["The cat sat on the mat and looked outside the window."]
summaries = ["A cat was sitting on a mat watching outside."]

# Compute BERTScore (default: RoBERTa-large model)
P, R, F1 = score(summaries, references, lang="en", verbose=True)

print(f"Precision: {P.mean().item():.4f}")
print(f"Recall:    {R.mean().item():.4f}")
print(f"F1 Score:  {F1.mean().item():.4f}")
