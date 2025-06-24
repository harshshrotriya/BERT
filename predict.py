import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys

# Check if text input is provided
if len(sys.argv) < 2:
    print("Usage: python predict.py \"Your text here\"")
    sys.exit(1)

# Get text from command line
text = sys.argv[1]

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the saved model and tokenizer
model_dir = "bert_model"
try:
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
except:
    print(f"Error: Could not load model from {model_dir}. Make sure to run train.py first.")
    sys.exit(1)

# Prepare the text for the model
inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

# Make prediction
model.eval()
with torch.no_grad():
    outputs = model(**inputs)

# Process the results
logits = outputs.logits
prediction = torch.argmax(logits, dim=1).item()
confidence = torch.nn.functional.softmax(logits, dim=1)[0][prediction].item()

# Map prediction to sentiment
sentiment = "Positive" if prediction == 1 else "Negative"

print(f"\nText: \"{text}\"")
print(f"Sentiment: {sentiment}")
print(f"Confidence: {confidence:.2%}")
