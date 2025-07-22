import os
import json
from pathlib import Path
from difflib import SequenceMatcher

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def match_headings(gt_headings, pred_headings, text_threshold=0.7, page_tolerance=1):
    matched_gt = set()
    matched_pred = set()
    level_mismatches = []
    for i, gt in enumerate(gt_headings):
        for j, pred in enumerate(pred_headings):
            # Allow ±1 page tolerance
            if (
                gt['level'] == pred['level'] and
                abs(gt['page'] - pred['page']) <= page_tolerance and
                similar(gt['text'].strip().lower(), pred['text'].strip().lower()) >= text_threshold
            ):
                matched_gt.add(i)
                matched_pred.add(j)
                break
            # Track level mismatches for debugging
            elif (
                abs(gt['page'] - pred['page']) <= page_tolerance and
                similar(gt['text'].strip().lower(), pred['text'].strip().lower()) >= text_threshold
            ):
                level_mismatches.append((gt['level'], pred['level'], gt['text'], pred['text']))
    tp = len(matched_gt)
    fp = len(pred_headings) - len(matched_pred)
    fn = len(gt_headings) - len(matched_gt)
    return tp, fp, fn, level_mismatches, matched_gt, matched_pred

def evaluate_all(ground_truth_dir, prediction_dir):
    gt_files = sorted([f for f in os.listdir(ground_truth_dir) if f.endswith('.json')])
    results = []
    total_tp = total_fp = total_fn = 0
    for gt_file in gt_files:
        gt_path = Path(ground_truth_dir) / gt_file
        pred_path = Path(prediction_dir) / gt_file
        if not pred_path.exists():
            print(f"Prediction for {gt_file} not found. Skipping.")
            continue
        gt_json = load_json(gt_path)
        pred_json = load_json(pred_path)
        gt_headings = gt_json.get('outline', [])
        pred_headings = pred_json.get('outline', [])
        tp, fp, fn, level_mismatches, matched_gt, matched_pred = match_headings(gt_headings, pred_headings)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        results.append({
            'file': gt_file,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        })
        total_tp += tp
        total_fp += fp
        total_fn += fn
        # Print unmatched headings for debugging
        unmatched_gt = [gt_headings[i] for i in range(len(gt_headings)) if i not in matched_gt]
        unmatched_pred = [pred_headings[j] for j in range(len(pred_headings)) if j not in matched_pred]
        print(f"\n{gt_file} - Unmatched ground-truth headings (first 5):")
        for h in unmatched_gt[:5]:
            print(h)
        print(f"{gt_file} - Unmatched predicted headings (first 5):")
        for h in unmatched_pred[:5]:
            print(h)
        if level_mismatches:
            print(f"{gt_file} - Level mismatches (first 5):")
            for l1, l2, t1, t2 in level_mismatches[:5]:
                print(f"GT: {l1} | Pred: {l2} | Text: {t1} <-> {t2}")
    # Overall metrics
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    print("\nPer-file results:")
    for r in results:
        print(f"{r['file']}: Precision={r['precision']:.2f}, Recall={r['recall']:.2f}, F1={r['f1']:.2f} (TP={r['tp']}, FP={r['fp']}, FN={r['fn']})")
    print("\nOverall:")
    print(f"Precision={overall_precision:.2f}, Recall={overall_recall:.2f}, F1={overall_f1:.2f}")

if __name__ == '__main__':
    gt_dir = Path('../sample_dataset/ground_truth')
    pred_dir = Path('../output')

    evaluate_all(gt_dir, pred_dir) 