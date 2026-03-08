# Attention Analysis in Reasoning LLMs

This repository contains code and analysis for studying attention mechanisms in large language models during mathematical reasoning tasks.

## Overview

We investigate how attention patterns change in Llama models when solving GSM8K math problems. The project implements:
- self-attention regression model for predicting attention scores, used for probing.
- Implementation of a custom masking strategy called "semantic-causal".
- Reasoning decomposition into semantic categories.
- Analysis of attention flows between reasoning components.
