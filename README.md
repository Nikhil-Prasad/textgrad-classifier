# TextGrad Classifier - Prompt Optimization with GPT-4

[![TextGrad CI](https://github.com/YOUR_USERNAME/textgrad-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/textgrad-classifier/actions/workflows/ci.yml)

This repository demonstrates how to use TextGrad with GPT-4 (via LiteLLM) to optimize prompts for classification tasks. It implements a low-level TextGrad optimization approach with custom loss functions, gradient descent, and automatic prompt refinement.

## Overview

The project showcases:
- **Low-level TextGrad API usage** with TextualGradientDescent and custom loss functions
- **Binary classification** (contact notes: satisfactory/needs_follow_up)
- **Multi-class classification** (GitHub issues: bug/feature/question)
- **Automated prompt optimization** that improves classification accuracy through iterative refinement
- **CI/CD integration** that fails if model accuracy drops below 70%

## Key Features

- 🎯 Custom TextGrad runner implementing gradient-based prompt optimization
- 📊 Automatic evaluation with accuracy and F1 metrics
- 🔄 Epoch-based training with prompt versioning
- 📈 Progress tracking with tqdm
- 🧪 Synthetic data generation for testing
- 🚀 GitHub Actions CI with performance threshold checks

## Setup

### Prerequisites

- Python 3.9+ (3.12 recommended)
- OpenAI API key with GPT-4 access
- Poetry (optional, for dependency management)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/textgrad-classifier.git
   cd textgrad-classifier
   ```

2. **Install dependencies**:
   
   Using Poetry (recommended):
   ```bash
   poetry install
   ```
   
   Using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API credentials**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### 1. Generate Training Data

Create synthetic contact notes dataset:
```bash
python scripts/prepare_data.py contact --rows 40
```

Download and prepare GitHub issues dataset:
```bash
python scripts/prepare_data.py issues --limit 3000
```

### 2. Run TextGrad Optimization

Optimize prompts for contact notes classification:
```bash
python scripts/run_textgrad.py configs/contact.yaml
```

Optimize prompts for issue classification:
```bash
python scripts/run_textgrad.py configs/issues.yaml
```

### 3. Review Results

Results are saved in `outputs/{timestamp}/`:
- `prompt_epoch_*.txt` - Optimized prompts after each epoch
- `predictions.csv` - Final predictions on test set
- `metrics_summary.json` - Performance metrics
- `results.json` - Detailed training history

## How It Works

### Low-Level TextGrad Implementation

Unlike the high-level TextGrad Trainer API, this implementation uses:

1. **Custom Loss Function**: Binary/multi-class classification loss (0/1)
2. **Manual Gradient Computation**: TextLoss with feedback messages
3. **Prompt Optimization**: TextualGradientDescent updates system prompts
4. **Controlled Training Loop**: Epoch-based with prompt size management

### Training Process

```python
# Initialize TextGrad components
engine = LiteLLMEngine(model_string="gpt-4o")
system_prompt = tg.Variable(initial_prompt, requires_grad=True)
optimizer = TextualGradientDescent(parameters=[system_prompt])

# Training loop
for epoch in range(epochs):
    for sample in train_set:
        # Forward pass
        response = model(sample)
        loss = classification_loss_fn(response, true_label)
        
        # Backward pass (only if incorrect)
        if loss > 0:
            text_loss = TextLoss(system_prompt, engine=engine)
            text_loss.backward()
            optimizer.step()
```

## Configuration

### YAML Config Structure

```yaml
experiment: "Contact Notes Classification"
model:
  engine: "gpt-4o"
  max_tokens: 256
  temperature: 0.2
prompt:
  system: "Initial system prompt..."
  user: "User prompt template with {{placeholders}}"
training:
  epochs: 3
  batch_size: 5
  learning_rate: 1.0
  max_prompt_chars: 20000
evaluation:
  expected_classes: ["satisfactory", "needs_follow_up"]
  metric: "accuracy"
  success_threshold: 0.70
data:
  csv_path: "data/contact_notes.csv"
  text_column: "note_content"
  label_column: "satisfactory"
  train_test_split: 0.8
```

## CI/CD Integration

The GitHub Actions workflow:
1. Generates synthetic datasets
2. Runs TextGrad optimization
3. **Fails if accuracy < 0.70**
4. Archives results as artifacts

To run CI locally:
```bash
act -j test-prompt-optimization --secret OPENAI_API_KEY=$OPENAI_API_KEY
```

## Project Structure

```
textgrad-classifier/
├── configs/              # YAML configuration files
│   ├── contact.yaml     # Contact notes classification config
│   └── issues.yaml      # GitHub issues classification config
├── data/                # Dataset files (generated)
│   ├── contact_notes.csv
│   └── issues.csv
├── scripts/             # Main implementation
│   ├── prepare_data.py  # Data generation/download
│   ├── run_textgrad.py  # CLI entry point
│   ├── textgrad_runner.py  # Core TextGrad implementation
│   └── eval_metrics.py  # Evaluation utilities
├── outputs/             # Training results (gitignored)
├── .github/workflows/   # CI configuration
│   └── ci.yml
├── pyproject.toml       # Poetry configuration
├── requirements.txt     # Pip requirements
└── README.md           # This file
```

## Extending the Project

### Adding New Datasets

1. Create a new config in `configs/`
2. Add data preparation to `prepare_data.py`
3. Adjust `text_columns` and `label_column` in config
4. Run with: `python scripts/run_textgrad.py configs/your_config.yaml`

### Custom Loss Functions

Modify `classification_loss_fn` in `textgrad_runner.py`:
```python
def custom_loss_fn(prediction: str, label: str) -> float:
    # Your custom logic here
    return loss_value
```

### Multi-Modal Support

Enable multi-modal in the engine:
```python
engine = LiteLLMEngine(
    model_string="gpt-4o",
    is_multimodal=True
)
```

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Reduce `batch_size` in config
2. **Memory Issues**: Lower `max_prompt_chars` to prevent token overflow
3. **Low Accuracy**: Increase `epochs` or adjust initial prompts

### Debug Mode

Enable verbose logging:
```bash
export TEXTGRAD_DEBUG=1
python scripts/run_textgrad.py configs/contact.yaml
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure CI passes
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- [TextGrad](https://github.com/zou-group/textgrad) - The gradient-based prompt optimization framework
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified LLM API interface
- [NLBSE'24](https://nlbse2024.github.io/) - GitHub issues dataset

## Citation

If you use this code in your research, please cite:
```bibtex
@software{textgrad-classifier,
  title = {TextGrad Classifier: Low-Level Prompt Optimization Demo},
  year = {2025},
  url = {https://github.com/nikhil-prasad/textgrad-classifier}
}
```
