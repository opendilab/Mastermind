import re


def compute_go_metrics(preds, labels):
    def extract_score(s):
        pattern = r'(?:the final leading score of black is: )([-+]?\d*\.\d+|\d+)'
        return float(re.findall(pattern, s)[0])

    def extract_winrate(s):
        pattern = r'(?:the win rate of black is: )([-+]?\d*\.\d+|\d+)'
        return float(re.findall(pattern, s)[0])

    # Extract from pred results.
    pred_score = []
    pred_winrate = []
    for string in preds:
        try:
            pred_score.append(extract_score(string))
        except:
            pred_score.append(None)
        try:
            pred_winrate.append(extract_winrate(string))
        except:
            pred_winrate.append(None)

    # Extract from label.
    label_score = []
    label_winrate = []
    for string in labels:
        label_score.append(extract_score(string))
        label_winrate.append(extract_winrate(string))

    # Calculate MAE
    score_mae = []
    winrate_mae = []
    for i in range(len(pred_score)):
        score_mae.append(10 if pred_score[i] is None else abs(pred_score[i] - label_score[i]))
        winrate_mae.append(1 if pred_winrate[i] is None else abs(pred_winrate[i] - label_winrate[i]))

    return {
        'ScoreMAE': sum(score_mae) / len(score_mae),
        'WinrateMAE': sum(winrate_mae) / len(winrate_mae)
    }


def compute_doudizhu_metrics(preds, labels):
    pattern = r'(?:Therefore, I will finally play )(.*)'
    # Extract from pred results.
    pred_val = []
    for string in preds:
        out = re.findall(pattern, string)
        pred_val.append(None if len(out) == 0 else out[0])

    # Extract from label.
    label_val = []
    for string in labels:
        out = re.findall(pattern, string)
        label_val.append(None if len(out) == 0 else out[0])

    action_correct = 0
    for i in range(len(pred_val)):
        if not (pred_val[i] is None or label_val[i] is None or pred_val[i] != label_val[i]):
            action_correct += 1
    thought_correct = 0
    for pred, label in zip(preds, labels):
        if pred == label:
            thought_correct += 1
    return {'Action Acc': round(action_correct / len(preds), 4), "Thought Acc": round(thought_correct / len(preds), 4)}
