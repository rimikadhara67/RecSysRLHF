# RLHF for Recommender Systems: Understanding User Preference Stability

This repository implements a Proximal Policy Optimization (PPO) algorithm with Reinforcement Learning from Human Feedback (RLHF). The goal is to study user preference stability over time and across contexts, focusing on key recommendation attributes like diversity, serendipity, and popularity. The project aims to improve recommendation quality and address the cold-start problem in recommender systems.

---

## Motivation

User preferences are dynamic and influenced by context and time. While prior research explores short-term changes or links preferences to personality traits, there’s a lack of longitudinal studies on user preference stability across recommendation attributes beyond accuracy. This project seeks to:

1. Evaluate user preference **stability** over time and changing contexts.
2. Use preference data to enhance recommendation algorithms.
3. Leverage RLHF for interactive feedback collection and fast personalization.

---

## Research Questions

1. **RQ1**: How stable are users’ preferences on recommendation diversity, serendipity, and popularity across longitudinal sessions?
2. **RQ2**: How stable are users’ preferences on these attributes across changing contexts?
3. **RQ3**: How can preference data be used to improve recommendation tasks?
