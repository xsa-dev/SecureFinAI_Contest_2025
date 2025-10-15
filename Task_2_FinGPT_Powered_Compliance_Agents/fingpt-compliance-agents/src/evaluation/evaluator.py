#!/usr/bin/env python3
"""
FinGPT Compliance Agents Evaluator
Evaluates model performance on various financial tasks
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

class FinGPTEvaluator:
    """Evaluator for FinGPT Compliance Agents"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        self.data_dir = Path(self.config['output']['data_path'])
        self.model_dir = Path(self.config['output']['model_path'])
        self.results_dir = Path(self.config['output']['results_path'])
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.tokenizer = None
        self.model = None
        self.text_generator = None
    
    def load_model(self):
        """Load trained model and tokenizer"""
        logger.info("Loading trained model")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_config['base_model'],
                device_map="auto",
                trust_remote_code=True,
                torch_dtype=torch.float16,
            )
            
            # Load LoRA adapter if exists
            if (self.model_dir / "adapter_config.json").exists():
                self.model = PeftModel.from_pretrained(base_model, self.model_dir)
                logger.info("Loaded model with LoRA adapter")
            else:
                self.model = base_model
                logger.info("Loaded base model")
            
            # Create text generation pipeline
            self.text_generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=self.model_config['temperature'],
                top_p=self.model_config['top_p'],
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate_financial_qa(self) -> Dict[str, float]:
        """Evaluate on financial Q&A task"""
        logger.info("Evaluating financial Q&A task")
        
        # Load FinanceBench dataset
        try:
            dataset = load_dataset("PatronusAI/financebench", split="test")
        except Exception as e:
            logger.error(f"Error loading FinanceBench dataset: {e}")
            return {}
        
        predictions = []
        references = []
        
        for example in tqdm(dataset, desc="Evaluating Financial Q&A"):
            question = example['question']
            context = example.get('context', '')
            answer = example['answer']
            
            # Create prompt
            prompt = f"""Answer the following financial question based on the provided context.

Context: {context}
Question: {question}

Answer:"""
            
            try:
                # Generate answer
                response = self.text_generator(prompt)
                predicted_answer = response[0]['generated_text'].strip()
                
                predictions.append(predicted_answer)
                references.append(answer)
                
            except Exception as e:
                logger.warning(f"Error generating answer: {e}")
                predictions.append("")
                references.append(answer)
        
        # Calculate metrics (simplified - would need more sophisticated evaluation)
        accuracy = self.calculate_qa_accuracy(predictions, references)
        
        results = {
            'accuracy': accuracy,
            'num_samples': len(predictions)
        }
        
        logger.info(f"Financial Q&A Results: {results}")
        return results
    
    def evaluate_sentiment_analysis(self) -> Dict[str, float]:
        """Evaluate on sentiment analysis task"""
        logger.info("Evaluating sentiment analysis task")
        
        # Load FPB dataset
        try:
            dataset = load_dataset("ChanceFocus/en-fpb", split="test")
        except Exception as e:
            logger.error(f"Error loading FPB dataset: {e}")
            return {}
        
        predictions = []
        references = []
        label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        
        for example in tqdm(dataset, desc="Evaluating Sentiment Analysis"):
            text = example['text']
            true_label = label_map[example['gold']]
            
            # Create prompt
            prompt = f"""Analyze the sentiment of the following financial text. Respond with only one word: positive, negative, or neutral.

Text: {text}

Sentiment:"""
            
            try:
                # Generate prediction
                response = self.text_generator(prompt)
                predicted_text = response[0]['generated_text'].strip().lower()
                
                # Extract sentiment from response
                if 'positive' in predicted_text:
                    predicted_label = 'positive'
                elif 'negative' in predicted_text:
                    predicted_label = 'negative'
                else:
                    predicted_label = 'neutral'
                
                predictions.append(predicted_label)
                references.append(true_label)
                
            except Exception as e:
                logger.warning(f"Error generating sentiment: {e}")
                predictions.append('neutral')
                references.append(true_label)
        
        # Calculate metrics
        accuracy = accuracy_score(references, predictions)
        f1 = f1_score(references, predictions, average='weighted')
        precision = precision_score(references, predictions, average='weighted')
        recall = recall_score(references, predictions, average='weighted')
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'num_samples': len(predictions)
        }
        
        logger.info(f"Sentiment Analysis Results: {results}")
        return results
    
    def evaluate_xbrl_extraction(self) -> Dict[str, float]:
        """Evaluate on XBRL extraction tasks"""
        logger.info("Evaluating XBRL extraction tasks")
        
        # This would need XBRL test data
        # For now, return placeholder results
        results = {
            'tag_extraction_accuracy': 0.0,
            'value_extraction_accuracy': 0.0,
            'formula_construction_accuracy': 0.0,
            'formula_calculation_accuracy': 0.0,
            'num_samples': 0
        }
        
        logger.info(f"XBRL Extraction Results: {results}")
        return results
    
    def calculate_qa_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Calculate accuracy for Q&A task (simplified)"""
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            # Simple string matching (would need more sophisticated evaluation)
            if pred.lower().strip() in ref.lower() or ref.lower().strip() in pred.lower():
                correct += 1
        
        return correct / total if total > 0 else 0.0
    
    def run_all_evaluations(self) -> Dict[str, Dict[str, float]]:
        """Run all evaluation tasks"""
        logger.info("Starting comprehensive evaluation")
        
        all_results = {}
        
        # Load model
        self.load_model()
        
        # Run evaluations
        try:
            all_results['financial_qa'] = self.evaluate_financial_qa()
        except Exception as e:
            logger.error(f"Error in financial Q&A evaluation: {e}")
            all_results['financial_qa'] = {}
        
        try:
            all_results['sentiment_analysis'] = self.evaluate_sentiment_analysis()
        except Exception as e:
            logger.error(f"Error in sentiment analysis evaluation: {e}")
            all_results['sentiment_analysis'] = {}
        
        try:
            all_results['xbrl_extraction'] = self.evaluate_xbrl_extraction()
        except Exception as e:
            logger.error(f"Error in XBRL extraction evaluation: {e}")
            all_results['xbrl_extraction'] = {}
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(all_results)
        all_results['overall'] = overall_metrics
        
        # Save results
        results_file = self.results_dir / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Evaluation completed. Results saved to {results_file}")
        logger.info(f"Overall Results: {overall_metrics}")
        
        return all_results
    
    def calculate_overall_metrics(self, results: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        metrics = ['accuracy', 'f1_score', 'precision', 'recall']
        overall = {}
        
        for metric in metrics:
            values = []
            for task, task_results in results.items():
                if task != 'overall' and metric in task_results:
                    values.append(task_results[metric])
            
            if values:
                overall[metric] = np.mean(values)
            else:
                overall[metric] = 0.0
        
        return overall

def main():
    parser = argparse.ArgumentParser(description='Evaluate FinGPT Compliance Agents')
    parser.add_argument('--config', default='configs/config.yaml', help='Config file path')
    
    args = parser.parse_args()
    
    evaluator = FinGPTEvaluator(args.config)
    results = evaluator.run_all_evaluations()
    
    print("\nEvaluation Results:")
    print("==================")
    for task, metrics in results.items():
        print(f"\n{task.upper()}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

if __name__ == "__main__":
    main()