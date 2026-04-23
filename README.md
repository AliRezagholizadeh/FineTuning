---
license: gemma
name: FunctionGemma-270m-it-Extended-Mobile-Actions
base_model: google/functiongemma-270m-it-270M
tags:
- function-calling
- mobile-actions
- gemma
Languages:
- English
---

## Function Gemma
Function Gemma is a Google trained light-wight model with 270M parameters to enables users to make connection between their prompt and executive actions.
You can find another variant of the model fine-tuned by Google over (google/mobile-actions)[https://huggingface.co/datasets/google/mobile-actions] here: [litert-community/FunctionGemma_270M_Mobile_Actions](https://huggingface.co/litert-community/FunctionGemma_270M_Mobile_Actions).
You can also use this FunctionGemma-270m-it-Extended-Mobile-Actions model variation fine-tuned over extended Mobile Actions tools set dataset in [AliRGHZ/Mobile-Actions](https://huggingface.co/datasets/AliRGHZ/Mobile-Actions).

For deploying the model into your application, kindly follow the instructions available [here](https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma).
For further fine-tuning you can follow the instructions suggested by Google...

###Training Configuration:
- Epochs: 2
- Batch size: 4 per device
- Gradient accumulation steps: 8
- Learning rate: 1e-5
- Scheduler: Cosine
- Optimizer: AdamW (fused)
- Precision: bfloat16
- Gradient checkpointing: Enabled
- Completion only loss: True (trains only on model outputs, not prompts)

###Training Infrastructure:
Hardware: Google Colab A100 GPU
Training time: ~24 minutes for 2 epochs
Library versions: transformers==5.2.0 torch==2.10.0 huggingface_hub==1.5.0, trl==0.29.0, accelerate==1.13.0