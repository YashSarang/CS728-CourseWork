# CS728 – Programming Assignment 2: Training Dynamics of RNNs and GRUs

## 1. Introduction
This report analyzes the training dynamics of Recurrent Neural Networks (RNNs) and Gated Recurrent Units (GRUs) on long-range dependency tasks. We focus on the phenomena of vanishing/exploding gradients and saturation of activation functions/gates.

## 2. Methodology
We implemented a Vanilla RNN and a GRU from scratch using PyTorch. Training was conducted on two synthetic tasks:
- **Memorization Task:** Classification of long-range sequences.
- **Multiplication Task:** Regression on product of values at specific indices.

We evaluated different gradient clipping regimes and compared the performance and stability of RNN vs. GRU.

## 3. Results - Task 1: Memorization

### 3.1 RNN (No Clipping) - Experiment A1
[A1 Results Analysis]
- **Gradient Flow:** [Describe log10(g_t) behavior]
- **Saturation:** [Describe d(h) behavior]

### 3.2 RNN (Clipping 0.05) - Experiment A2
[A2 Results Analysis]
- **Effect of Clipping:** [Compare with A1]

### 3.3 RNN (Clipping 0.01) - Experiment A3
[A3 Results Analysis]

### 3.4 GRU (No Clipping/Clipping 0.05) - Experiments A4 & A5
[GRU Results Analysis]
- **Gating Mechanism:** [Explain how gates affect gradient flow]
- **Gate Saturation:** [Analyze z and r gate behavior]

## 4. Results - Task 2: Multiplication

### 4.1 RNN vs. GRU (B1 vs. B2)
[Regression task comparison]

## 5. Summary & Interpretation
[Summarize findings on vanishing/exploding gradients and architectural differences]

## 6. Appendix: Plots
(Plots will be generated and attached here)
