# Explainable AI Commentary Generation for Deep RL Chess Agent

This project aims to bridge the gap between Deep Reinforcement Learning (DRL) decision-making and human interpretability in chess by combining **CNN probe-based feature detection** with **Large Language Model (LLM)**-driven natural language explanations.

## Objective
To develop an explainable AI framework that can interpret and articulate the reasoning behind a deep RL chess engine’s moves through automated commentary.

## Methodology

### 1. **Model Integration**
- Used a deep RL chess engine trained with self-play.
- Extracted intermediate feature maps from convolutional layers during inference.

### 2. **CNN Probe Training**
- Trained 22 convolutional probes to detect tactical and positional motifs (e.g., forks, pins, checkmates).
- Each probe used labeled examples of chess positions.
- Achieved **AUC-ROC ≈ 0.92** on validation data.

### 3. **Feature Interpretation**
- Converted probe outputs to numeric metrics to feed into a LLM prompt.

### 4. **Natural Language Generation**
- Generated commentary by feeding structured templates into **GPT-4o** prompt templates.
- Commentary balanced **accuracy** (tactical correctness) and **conceptual clarity**.

### 5. **Evaluation**
- Benchmarked commentary against baseline textual explanations.
- Observed **>40% improvement** in both concept recognition and tactical accuracy metrics.

## Results
- Successfully integrated DRL interpretability with generative commentary.
- Demonstrated interpretability gains while maintaining chess engine performance.
- Paved the way for explainable reinforcement learning in strategic games.

