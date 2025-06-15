"""
textgrad_runner.py
==================

Custom TextGrad runner that implements low-level optimization using
TextualGradientDescent, adapted from archive_textgrad.py but working
with CSV files instead of databases.

This module provides:
- CSV data loading and train/test splitting
- Custom binary/multiclass classification loss functions
- Low-level TextGrad optimization with gradient descent
- Evaluation and metrics computation
- Prompt persistence after each epoch
"""

import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml

import textgrad as tg
from textgrad.loss import TextLoss
from textgrad.optimizer import TextualGradientDescent
from textgrad.engine_experimental.litellm import LiteLLMEngine


class TextGradRunner:
    """
    Custom TextGrad runner implementing low-level optimization approach.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the runner with configuration.
        
        Parameters
        ----------
        config_path : str
            Path to YAML configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize LLM engine
        import os
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
        self.engine = LiteLLMEngine(
            model_string=self.config['model']['engine'],
            is_multimodal=False,
            cache=False
        )
        
        # Set backward engine for TextGrad
        tg.set_backward_engine(self.engine)
        
        # Initialize system prompt as TextGrad variable
        self.system_prompt = tg.Variable(
            self.config['prompt']['system'],
            requires_grad=True,
            role_description="system prompt for classification"
        )
        
        # Store user prompt template
        self.user_prompt_template = self.config['prompt']['user']
        
        # Create model wrapper
        self.model = tg.BlackboxLLM(
            engine=self.engine,
            system_prompt=self.system_prompt
        )
        
        # Initialize optimizer with gradient memory to prevent context overflow
        # gradient_memory limits how many past gradients are retained
        gradient_memory = self.config['training'].get('gradient_memory', 10)
        self.optimizer = TextualGradientDescent(
            parameters=[self.system_prompt],
            engine=self.engine,
            gradient_memory=gradient_memory  # Only keep last N gradients
        )
        
        # Training parameters
        self.epochs = self.config['training']['epochs']
        self.batch_size = self.config['training']['batch_size']
        self.max_prompt_chars = self.config['training']['max_prompt_chars']
        
        # Evaluation parameters
        self.expected_classes = self.config['evaluation']['expected_classes']
        self.metric = self.config['evaluation']['metric']
        self.success_threshold = self.config['evaluation']['success_threshold']
        
        # Results storage
        self.results = {
            'train_history': [],
            'eval_history': [],
            'optimized_prompts': []
        }
    
    def load_data(self, csv_path: str) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Load CSV data and split into train/test sets.
        
        Returns
        -------
        train_set, test_set : List of tuples
            Each tuple is (input_text, label, row_id)
        """
        df = pd.read_csv(csv_path)
        
        # Handle single or multiple text columns
        if 'text_column' in self.config['data']:
            # Single text column
            text_col = self.config['data']['text_column']
            df['input_text'] = df[text_col]
        else:
            # Multiple text columns (e.g., title + body)
            text_cols = self.config['data']['text_columns']
            # Create formatted text from template
            df['input_text'] = df.apply(
                lambda row: self.user_prompt_template.format(**{
                    col: row[col] for col in text_cols
                }),
                axis=1
            )
        
        label_col = self.config['data']['label_column']
        
        # Create dataset
        dataset = [
            (row['input_text'], row[label_col], idx)
            for idx, row in df.iterrows()
        ]
        
        # Split data
        split_ratio = self.config['data']['train_test_split']
        split_idx = int(len(dataset) * split_ratio)
        
        # Shuffle before splitting
        np.random.seed(42)
        np.random.shuffle(dataset)
        
        train_set = dataset[:split_idx]
        test_set = dataset[split_idx:]
        
        # Optionally limit training samples
        if 'max_train_samples' in self.config['training']:
            max_samples = self.config['training']['max_train_samples']
            if max_samples and max_samples < len(train_set):
                train_set = train_set[:max_samples]
        
        return train_set, test_set
    
    def format_input(self, text: str, label_col_data: Optional[Dict] = None) -> str:
        """
        Format input text using the user prompt template.
        """
        if label_col_data:
            # For multi-column inputs (already formatted in load_data)
            return text
        else:
            # For single column inputs
            col_name = self.config['data']['text_column']
            return self.user_prompt_template.replace(f"{{{{{col_name}}}}}", text)
    
    def parse_model_response(self, response: str) -> str:
        """
        Parse and normalize model response to expected class label.
        """
        response = response.strip().lower()
        
        # Try exact match first
        for expected_class in self.expected_classes:
            if response == expected_class.lower():
                return expected_class
        
        # For bug/non_bug specifically, check for longer match first
        sorted_classes = sorted(self.expected_classes, key=len, reverse=True)
        for expected_class in sorted_classes:
            if expected_class.lower() in response:
                return expected_class
        
        # If no match, return the response as-is (will count as error)
        return response
    
    def classification_loss_fn(self, prediction: str, label: str) -> float:
        """
        Binary/multiclass classification loss: 0 if correct, 1 if wrong.
        """
        pred_normalized = self.parse_model_response(prediction)
        label_normalized = str(label).strip().lower()
        
        
        # Find the matching expected class for the label
        for expected_class in self.expected_classes:
            if expected_class.lower() == label_normalized:
                label_normalized = expected_class
                break
        
        return 0.0 if pred_normalized == label_normalized else 1.0
    
    def eval_sample(self, sample: Tuple) -> Tuple[float, str]:
        """
        Evaluate a single sample and return accuracy + model output.
        """
        text, label, _ = sample
        
        # Create input variable
        x_var = tg.Variable(
            text,
            requires_grad=False,
            role_description="input text for classification"
        )
        
        # Get model response
        response = self.model(x_var)
        response_str = str(response.value) if hasattr(response, 'value') else str(response)
        
        # Calculate accuracy
        loss = self.classification_loss_fn(response_str, label)
        accuracy = 1.0 - loss
        
        return accuracy, response_str
    
    def eval_dataset(self, dataset: List[Tuple], desc: str = "Evaluating") -> Dict[str, Any]:
        """
        Evaluate entire dataset and return metrics.
        """
        results = []
        predictions = []
        true_labels = []
        
        with tqdm(total=len(dataset), desc=desc, ncols=80) as pbar:
            for sample in dataset:
                acc, pred = self.eval_sample(sample)
                results.append({
                    'sample_id': sample[2],
                    'accuracy': acc,
                    'prediction': self.parse_model_response(pred),
                    'true_label': sample[1]
                })
                predictions.append(self.parse_model_response(pred))
                true_labels.append(sample[1])
                pbar.update(1)
        
        # Calculate overall metrics
        accuracy = np.mean([r['accuracy'] for r in results])
        
        # Calculate F1 if needed
        if self.metric == "macro-F1":
            from sklearn.metrics import f1_score
            f1 = f1_score(true_labels, predictions, average='macro', labels=self.expected_classes)
            metric_value = f1
        else:
            metric_value = accuracy
        
        return {
            'results': results,
            'accuracy': accuracy,
            'metric': self.metric,
            'metric_value': metric_value,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def train_one_epoch(self, train_set: List[Tuple], epoch: int) -> float:
        """
        Run one epoch of training with gradient descent.
        """
        total_loss = 0
        with tqdm(total=len(train_set), desc=f"Epoch {epoch}", ncols=80) as pbar:
            for sample in train_set:
                text, label, _ = sample
                
                # Create input variable
                x_var = tg.Variable(
                    text,
                    requires_grad=False,
                    role_description="input text for classification"
                )
                
                # Forward pass
                response = self.model(x_var)
                response_str = str(response.value) if hasattr(response, 'value') else str(response)
                
                # Calculate loss
                loss = self.classification_loss_fn(response_str, label)
                total_loss += loss
                
                # Backward pass only if loss > 0
                if loss > 0:
                    # Create a loss evaluation prompt that will generate feedback
                    loss_eval_prompt = f"""You are evaluating a classification system's performance.

The system was given this input:
{text[:200]}...

The system's current prompt is:
{self.system_prompt.value}

The system predicted: '{response_str}'
But the correct answer is: '{label}'

Provide specific feedback on how to improve the system prompt to correctly classify this type of input as '{label}'."""
                    
                    # Create TextLoss with the evaluation prompt
                    text_loss = TextLoss(
                        eval_system_prompt=loss_eval_prompt,
                        engine=self.engine
                    )
                    
                    # Apply loss to the system prompt (what we're optimizing)
                    l = text_loss(self.system_prompt)
                    l.backward()
                    
                    # Truncate prompt if too long
                    if len(self.system_prompt.value) > self.max_prompt_chars:
                        self.system_prompt.value = self.system_prompt.value[:self.max_prompt_chars]
                    
                    # Update prompt
                    self.optimizer.step()
                
                pbar.set_postfix({
                    'loss': f"{loss:.3f}",
                    'avg': f"{(total_loss/(pbar.n+1)):.3f}"
                })
                pbar.update(1)
        
        return total_loss / len(train_set)
    
    def fit(self, csv_path: str, output_dir: str):
        """
        Run the full training loop.
        """
        # Load data
        train_set, test_set = self.load_data(csv_path)
        print(f"Loaded {len(train_set)} training samples, {len(test_set)} test samples")
        
        # Initial evaluation
        print("\nInitial evaluation on test set:")
        eval_results = self.eval_dataset(test_set, desc="Initial eval")
        print(f"Initial {self.metric}: {eval_results['metric_value']:.3f}")
        self.results['eval_history'].append(eval_results)
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            print(f"\n--- Epoch {epoch}/{self.epochs} ---")
            
            # Train one epoch
            avg_loss = self.train_one_epoch(train_set, epoch)
            print(f"Average training loss: {avg_loss:.3f}")
            self.results['train_history'].append({
                'epoch': epoch,
                'avg_loss': avg_loss
            })
            
            # Evaluate on test set
            eval_results = self.eval_dataset(test_set, desc=f"Eval epoch {epoch}")
            print(f"{self.metric} after epoch {epoch}: {eval_results['metric_value']:.3f}")
            eval_results['epoch'] = epoch
            self.results['eval_history'].append(eval_results)
            
            # Save optimized prompt
            prompt_path = Path(output_dir) / f"prompt_epoch_{epoch}.txt"
            prompt_path.write_text(self.system_prompt.value)
            self.results['optimized_prompts'].append({
                'epoch': epoch,
                'path': str(prompt_path),
                'prompt': self.system_prompt.value
            })
        
        # Save final results
        self._save_results(output_dir, test_set)
        
        # Return success based on threshold
        final_metric = self.results['eval_history'][-1]['metric_value']
        return final_metric >= self.success_threshold
    
    def _save_results(self, output_dir: str, test_set: List[Tuple]):
        """
        Save all results and metrics.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results JSON
        with open(output_path / "results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save final predictions CSV
        final_eval = self.results['eval_history'][-1]
        predictions_df = pd.DataFrame({
            'sample_id': [r['sample_id'] for r in final_eval['results']],
            'true_label': [r['true_label'] for r in final_eval['results']],
            'prediction': [r['prediction'] for r in final_eval['results']],
            'correct': [r['accuracy'] for r in final_eval['results']]
        })
        predictions_df.to_csv(output_path / "predictions.csv", index=False)
        
        # Save metrics summary
        metrics_summary = {
            'experiment': self.config['experiment'],
            'final_accuracy': float(final_eval['accuracy']),
            f'final_{self.metric}': float(final_eval['metric_value']),
            'success_threshold': float(self.success_threshold),
            'passed': bool(final_eval['metric_value'] >= self.success_threshold),
            'epochs_trained': int(self.epochs),
            'train_samples': len(test_set)
        }
        
        with open(output_path / "metrics_summary.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Print final summary
        print("\n" + "="*60)
        print(f"Training completed for: {self.config['experiment']}")
        print(f"Final {self.metric}: {final_eval['metric_value']:.3f}")
        print(f"Threshold: {self.success_threshold}")
        print(f"Status: {'PASSED' if metrics_summary['passed'] else 'FAILED'}")
        print("="*60)