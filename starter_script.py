# Start of metadata ID script

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)

# Choosing the AI model
# google's (i believe) BERT natural language model, trained on masked english language, meaning
# it is tasked to identify missing words in sentences that are fed to it as part of its training
model_name = "bert-base-uncased"

# Load the model and tokenizer for a classification task.
# Set num_labels to the number of metadata fields you want to classify.
num_labels = 5  # e.g., date, creator, description, subject, identifier
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a classification pipeline.
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Example metadata snippet – replace with your actual examples.
example_text = "Acquired on 12/31/2024"
result = classifier(example_text)
print(result)
