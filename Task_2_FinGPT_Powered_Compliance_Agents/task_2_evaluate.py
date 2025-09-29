#!/usr/bin/env python3
"""
Task 2: Financial Sentiment Analysis - Starter Kit
SecureFinAI Contest 2025

Example script that loads the FPB dataset and model, using Llama-3.1-8B on the FPB dataset.
We will evaluate the submitted models using similar scripts based on different datasets settings.
"""

import torch
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from sklearn.metrics import accuracy_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set logging levels
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

def setup_model():
    print("Loading Llama-3.1-8B...")

    # Completely suppress output during model loading
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            trust_remote_code=True,
            padding_side="left"
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B",
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )

        text_generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )

    print("Model loaded!")
    return text_generator

def predict_sentiment(text, text_generator):
    """Predict sentiment"""
    prompt = f"""Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.

Text: {text}

Answer:"""

    try:
        # Temporarily suppress stdout/stderr
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            outputs = text_generator(prompt)

        # Since return_full_text=False, we get only the generated part
        response = outputs[0]['generated_text'].strip().lower()

        if "positive" in response:
            return "positive"
        elif "negative" in response:
            return "negative"
        else:
            return "neutral"
    except:
        return "neutral"

def main():
    print("=== Task 2: Financial Sentiment Analysis ===")
    
    # Load dataset
    print("Loading FPB dataset...")
    dataset = load_dataset("ChanceFocus/en-fpb")
    print(f"Dataset: train={len(dataset['train'])}, test={len(dataset['test'])}")
    
    # Setup model
    text_generator = setup_model()
    
    # Demo on 3 samples
    print("\n--- Demo Samples ---")
    label_names = ['positive', 'neutral', 'negative']
    
    for i in range(3):
        sample = dataset['test'][i]
        text = sample['text']
        true_label = label_names[sample['gold']]
        
        predicted = predict_sentiment(text, text_generator)
        correct = "✓" if predicted == true_label else "✗"
        
        print(f"\nSample {i+1}: {correct}")
        print(f"Text: {text[:80]}...")
        print(f"True: {true_label} | Predicted: {predicted}")
    
    # Evaluate on full test set
    test_size = len(dataset['test'])
    print(f"\n--- Evaluating {test_size} samples ---")
    predictions = []
    true_labels = []

    for i in range(test_size):
        sample = dataset['test'][i]
        text = sample['text']
        true_label = label_names[sample['gold']]

        predicted = predict_sentiment(text, text_generator)
        predictions.append(predicted)
        true_labels.append(true_label)

        if i % 50 == 0:
            print(f"Processed {i+1}/{test_size}...")
    
    # Results
    accuracy = accuracy_score(true_labels, predictions)
    print(f"\nAccuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Count by label
    correct_pos = sum(1 for t, p in zip(true_labels, predictions) if t == p == 'positive')
    correct_neu = sum(1 for t, p in zip(true_labels, predictions) if t == p == 'neutral')
    correct_neg = sum(1 for t, p in zip(true_labels, predictions) if t == p == 'negative')
    
    total_pos = true_labels.count('positive')
    total_neu = true_labels.count('neutral')
    total_neg = true_labels.count('negative')
    
    print(f"Positive: {correct_pos}/{total_pos}")
    print(f"Neutral:  {correct_neu}/{total_neu}")
    print(f"Negative: {correct_neg}/{total_neg}")
    
    print("\nDemo completed!")

if __name__ == "__main__":
    main()
