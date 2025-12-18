from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=-1
)

def summarize(texts, length="medium"):
    max_len = {
        "short": 60,
        "medium": 150,
        "detailed": 300
    }[length]

    summaries = []
    for text in texts:
        if len(text) < 50:
            continue

        s = summarizer(
            text,
            max_length=max_len,
            min_length=30,
            do_sample=False
        )
        summaries.append(s[0]["summary_text"])

    return " ".join(summaries)
