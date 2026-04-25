from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_tokenizer(model_name: str):
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(model_name)


def load_sequence_classifier(model_name: str, num_labels: int):
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            local_files_only=True,
        )
    except OSError:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
