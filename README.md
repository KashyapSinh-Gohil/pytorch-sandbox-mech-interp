---
title: PyTorchSandbox
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: bsd-3-clause
---

# PyTorchSandbox: Agentic Mechanistic Interpretability Benchmark

PyTorchSandbox is an OpenEnv environment used to evaluate whether an LLM agent can reverse-engineer neural network internals through PyTorch code execution.

---

## Abstract

PyTorchSandbox presents an LLM agent with a **3-task curriculum** of increasing difficulty in the domain of **Mechanistic Interpretability (MI)**  the science of reverse-engineering trained neural networks into human-understandable algorithms. The agent must write and execute real PyTorch code, use forward hooks to inspect activations, and submit mathematically verifiable solutions  all within a memory-safe sandboxed execution environment.

Unlike static benchmarks (MMLU, HumanEval), this environment tests **agentic scientific reasoning**: the ability to form hypotheses, write experiments, interpret results, and iterate  the exact workflow of a human AI Safety researcher.

---

## The Curriculum

| Task | Difficulty | Domain | Model | Objective |
|------|-----------|--------|-------|-----------|
| **1. Dead Neuron Detection** | Easy | Weight Inspection | 1-layer Linear MLP (101) | Identify input indices with zero-valued weights |
| **2. Causal Ablation** | Medium | Activation Patching | 2-layer MLP computing y = x1*x2 + x3 | Identify the specific neuron index responsible for multiplication |
| **3. Fourier Analysis** | Hard | Synthetic Features | 1-layer Transformer | Extract 5 planted frequencies from the embedding matrix |

### Task 1: Dead Neuron Detection
The agent receives a simple linear model and must inspect `model.layer.weight` to find which input features have been zeroed out. This tests basic PyTorch introspection.

### Task 2: Causal Ablation
The agent must use `register_forward_hook()` on the hidden layer to systematically zero out each of the 10 hidden neurons, run test inputs through the modified model, and measure which ablation causes the largest output deviation. This mirrors the **activation patching** technique used in real MI research (Conmy et al., 2023).

### Task 3: Fourier Analysis of Planted Features
Inspired by mechanistic analysis of algorithmic tasks, the agent must analyze a Transformer engineered with synthetic modular arithmetic features. The key frequencies are synthetically planted into the embedding matrix `model.W_E.weight`. The agent must compute the Discrete Fourier Transform across token positions and identify the 5 dominant frequencies. This tests the agent's ability to rigorously extract specific mathematical structures from weight matrices.

---

## Architecture

```

                  inference.py                    
         (Agentic LLM Evaluation Loop)           
                                                  
        
    OpenAI     HF Inference Router       
    Client     (Qwen/Qwen2.5-72B)       
        
                                                 
                                                 
      
           MechInterpEnvironment               
          
          Sandbox exec() Engine              
       Isolated namespace per step          
       stdout/stderr capture                
       Hook lifecycle management            
       _forward_hooks.clear() guard         
          
          
         Mathematical Grader                 
       Exact match (Tasks 1 & 2)            
       MSE-based scoring (Task 3)           
       Progressive task advancement         
          
      

```

### Dual-Mode Action Schema

The agent interacts through `MechInterpAction`, which supports two modes:

```python
# Mode A: Execute PyTorch code in the sandbox
{"python_code": "print(model.layer.weight)"}

# Mode B: Submit a graded solution
{"solution_target": [2, 5, 8]}
```

### Memory Safety

Every `step()` call that executes code automatically purges all registered hooks from the model graph:

```python
for module in current_model.modules():
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()
```

This prevents the Out-of-Memory (OOM) failures that plague long-running PyTorch interactive sessions  a critical requirement for an agentic evaluation that may run 30+ steps.

---

## Evaluation Protocol

The environment follows the **OpenEnv** specification exactly:

- **Logging**: `[START]`, `[STEP]`, `[END]` format written to stdout
- **Scoring**: Per-task rewards averaged into a final score  [0, 1]
- **Advancement**: Agent must score  0.99 on each task to advance to the next
- **Termination**: `done=True` when all 3 tasks are solved, or `MAX_STEPS=30` reached
- **Infrastructure**: Runs within 2 vCPU / 8GB RAM Docker container

### Grading Functions

| Task | Grading | Maximum Reward Condition |
|------|---------|----------------------|
| 1 | Exact set match | All dead neuron indices found |
| 2 | Narrowing | `max(0.1, 1.0 / len(submission))` if the true ID is in the submitted list |
| 3 | MSE-based | `1.0 - MSE(predicted, ground_truth)`  0.99 |

---

## Running Locally

### Prerequisites
```bash
pip install openenv-core torch pydantic fastapi uvicorn openai
```

### Start the Server
```bash
cd mech_interp
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run the Inference Agent
```bash
export HF_TOKEN="hf_your_token_here"
python inference.py
```

### Validate Before Deployment
```bash
openenv validate
# Expected: [OK] mech_interp: Ready for multi-mode deployment
```

---

## Deployment to Hugging Face Spaces

```bash
openenv push --repo-id <your-username>/mech_interp
```

After deployment, add your `HF_TOKEN` as a **Secret** in the Space Settings to enable the LLM inference loop.

---

## Project Structure

```
mech_interp/
 Dockerfile                          # Simple Docker build (root-level for HF Spaces)
 README.md                           # This document
 openenv.yaml                        # OpenEnv manifest
 pyproject.toml                      # Dependencies and project metadata
 inference.py                        # Agentic LLM evaluation loop
 models.py                           # MechInterpAction / MechInterpObservation schemas
 client.py                           # OpenEnv client wrapper
 artifacts/
    task1.pt                        # Dead Neuron MLP (101 Linear)
    task2.pt                        # Causal Ablation MLP (3101)
    task3.pt                        # Synthetic Transformer (planted mod-97 frequencies)
 server/
     app.py                          # FastAPI + WebSocket server
     mech_interp_environment.py      # Core physics engine (sandbox + grader)
     model_architectures.py          # PyTorch nn.Module definitions
     gen_art.py                      # Offline artifact generation script
```

---

## Research References

- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Conmy, A., Mavor-Parker, A. N., Lynch, A., Heimersheim, S., & Garriga-Alonso, A. (2023). *Towards Automated Circuit Discovery for Mechanistic Interpretability.* NeurIPS 2023. [arXiv:2304.14997](https://arxiv.org/abs/2304.14997)

---

## License

This project is licensed under the BSD-style license. See `LICENSE` for details.

Built with [OpenEnv](https://github.com/meta-pytorch/openenv) by Meta.
