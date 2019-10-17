#!/usr/bin/env python3

def sumSentiment(preds: list) -> float:
    percent_pos = (preds.count('positive') / len(preds)) * 100
    percent_neg = (preds.count('negative') / len(preds)) * 100
    percent_neu = (preds.count('neutral') / len(preds)) * 100
    opinions = [percent_pos, percent_neu, percent_neg]
    return opinions 
