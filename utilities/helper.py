#!/usr/bin/env python3

def sumSentiment(preds: list) -> float:
    percent_pos = preds.count('positive') / len(preds)
    percent_neg = preds.count('negative') / len(preds)
    percent_neu = preds.count('neutral') / len(preds)
    return percent_pos, percent_neu, percent_neg
