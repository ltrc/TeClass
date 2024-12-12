print("************* new rouge score *************")
ref = 'రవీంద్ర నగర్లో ఉరి వేసుకుని దంపతుల ఆత్మహత్య'
hyp = 'ఉరి'

from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, lang="bengali")
scores = scorer.score(ref, hyp)
print(scores)


print("*********** Old rouge score ****************")
from rouge import Rouge 


rouge = Rouge()
scores = rouge.get_scores(ref, hyp)

print(scores)