from evaluate import load

bertscore = load("bertscore")

def bert_score(preds, targets, lang='tr', batch_size=32, verbose=False):
    results = bertscore.compute(predictions=preds, references=targets, model_type="dbmdz/bert-base-turkish-cased", batch_size=batch_size, verbose=verbose)
    return {
        'precision': 100 * sum(results['precision']) / len(results['precision']),
        'recall': 100 * sum(results['recall']) / len(results['recall']),
        'f1': 100 * sum(results['f1']) / len(results['f1']),
    }