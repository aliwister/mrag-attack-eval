
def compute_metrics_from_predictions(results, labels, rag_list):
    N = len(results)

    # Normalize predictions: take first word, uppercase
    norm_preds = [label.split()[0].upper() if isinstance(label, str) else "NO" for label in results]

    tp, tn, fp, fn = [], [], [], []
    is_rag_gt, is_rag_fp = 0, 0
    
    for i in range(N):
        in_rag = i in rag_list[i]
        pred = norm_preds[i]
        is_gt_yes = labels[i]
        is_pred_yes = pred == "YES"

        is_rag_gt += 1 if in_rag & is_gt_yes else 0
        is_rag_fp += 1 if in_rag ^ is_gt_yes else 0

        if is_gt_yes and is_pred_yes:
            tp.append(i)
        elif not is_gt_yes and is_pred_yes:
            fp.append(i)
        elif is_gt_yes and not is_pred_yes:
            fn.append(i)
        else:
            tn.append(i)

    # Metrics
    accuracy = (len(tp) + len(tn)) / N if N else 0
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 0
    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "retrieval_accuracy": is_rag_gt / (is_rag_gt + is_rag_fp)
        #"false_positives": fp,
        #"false_negatives": fn
    }