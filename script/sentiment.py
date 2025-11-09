from textblob import TextBlob

def analyze_tone(user_input):
    analysis = TextBlob(user_input).sentiment
    polarity = analysis.polarity  # -1 (negative) → +1 (positive)

    if polarity > 0.3:
        return "The user seems happy. Reply in an upbeat, friendly tone."
    elif polarity < -0.3:
        return "The user seems upset. Reply calmly, politely, and supportively."
    else:
        return "The user seems neutral. Reply normally."
