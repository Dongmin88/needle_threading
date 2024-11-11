README: NeedleThreadingTest Script
This README provides instructions and details for the NeedleThreadingTest script, which tests key-value retrieval tasks using a transformer model for causal language modeling.

Overview
The NeedleThreadingTest script is a Python-based utility designed to:

Generate synthetic data for testing (key-value pairs and threading relationships).
Use a transformer model to perform key-value retrieval tasks.
Validate the model's response against expected results.
Measure accuracy and performance metrics for various test scenarios.
It leverages Hugging Face's transformers library to load and interact with language models and requires a Hugging Face authentication token.

Features
Key-Value Generation:

Generates UUID-based key-value pairs within token constraints.
Supports creating threading relationships between keys.
Task Types:

Single Needle: Retrieve a specific value given a key.
Threading: Follow a chain of keys and values until a terminal value is reached.
Model Interaction:

Supports pre-trained causal language models (e.g., LLaMA).
Formats tasks as prompts for model inference.
Validation:

Compares model output against expected values for accuracy measurement.
Performance Logging:

Logs task duration and correctness for each trial.
Outputs results as a JSON file for further analysis.
Installation
Prerequisites
Python 3.8+
A Hugging Face account with an access token
GPU support (optional but recommended)
Required Libraries
The following Python packages are required:

transformers
torch
accelerate
huggingface_hub
To install dependencies, the script automatically installs them if missing. Alternatively, use the following command manually:

bash
pip install transformers torch accelerate huggingface_hub
Setup
Hugging Face Authentication:

Obtain an access token from Hugging Face.
Update the token variable in the setup_huggingface() function with your token.
Model Configuration:

Specify the desired model in the NeedleThreadingTest class. The default is "meta-llama/Llama-3.2-1B".
Run the Script: Execute the script using:

bash
python needle_threading_test.py
Usage
Generate Key-Value Pairs:

The script generates UUID-based key-value pairs while respecting token limits.
Test Execution:

Run tests for both "Single Needle" and "Threading" tasks.
Automatically adjusts task size and records performance metrics.
Results:

Outputs results to a JSON file in the format results_llama_<timestamp>.json.
Code Structure
setup_huggingface(): Authenticates with Hugging Face and ensures dependencies are installed.

HaystackGenerator: Generates synthetic data (key-value pairs) within token constraints.

NeedleThreadingTest: Main testing class that:

Loads the model and tokenizer.
Executes retrieval tasks.
Validates model responses.
Logs test results.
main(): Entry point that initializes testing and writes results to a JSON file.

Example Output
Sample JSON result:

{
  "model": "Llama-3.2-1B",
  "timestamp": "2024-11-12T15:30:45.123Z",
  "tests": [
    {
      "task": "single_needle",
      "size": 5,
      "trial": 0,
      "duration": 0.134,
      "is_correct": true,
      "expected": "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p",
      "received": "1a2b3c4d-5e6f-7g8h-9i0j-1k2l3m4n5o6p"
    }
  ]
}
Notes and Troubleshooting
GPU Usage:

Ensure GPU is available for faster inference. The script automatically detects and uses GPU if available.
Token Constraints:

Key-value pair generation is limited to ensure the prompt stays within the model's token limit.
Model Loading Issues:

Verify the model exists on Hugging Face and that you have permission to access it.
Validation:

The script uses regex to validate UUID format in model responses.
License
This script is provided under the MIT License. Use it as you see fit for educational or testing purposes.

For further inquiries, feel free to reach out!