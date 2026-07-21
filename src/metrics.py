import numpy as np


def bert_score(preds, targets, lang='tr', batch_size=32, verbose=False):
    # results = bertscore.compute(predictions=preds, references=targets, model_type="dbmdz/bert-base-turkish-cased", batch_size=batch_size, verbose=verbose)
    results = None
    return {
        'precision': 100 * sum(results['precision']) / len(results['precision']),
        'recall': 100 * sum(results['recall']) / len(results['recall']),
        'f1': 100 * sum(results['f1']) / len(results['f1']),
    }
    
def islr_performance_topk(txt_ref, txt_hyps, k=5):
    true_sample = 0
    for tgt_pres, tgt_ref in zip(txt_hyps, txt_ref):
        true_sample += int(tgt_ref in tgt_pres[:k])
    
    topk_acc_pi = true_sample / len(txt_hyps) * 100

    gt_dict = {}
    pred_dict = {}
    for tgt_pres, tgt_ref in zip(txt_hyps, txt_ref):
        if tgt_ref in gt_dict.keys():
            gt_dict[tgt_ref] += 1
            pred_dict[tgt_ref] += int(tgt_ref in tgt_pres[:k])
        else:
            gt_dict[tgt_ref] = 1
            pred_dict[tgt_ref] = int(tgt_ref in tgt_pres[:k])

    mean_acc_pc = []
    for key in gt_dict.keys():
        mean_acc_pc.append(pred_dict[key] / gt_dict[key])
    topk_acc_pc = np.array(mean_acc_pc).mean() * 100

    print(f"top{k}_acc_pi: {topk_acc_pi:.2f}")
    print(f"top{k}_acc_pc: {topk_acc_pc:.2f}")
    
    return topk_acc_pi, topk_acc_pc