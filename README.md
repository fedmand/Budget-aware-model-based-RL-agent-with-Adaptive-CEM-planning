# Budget-Aware Model-Based RL Agent with Adaptive CEM Planning

A model-based reinforcement learning agent that learns to navigate a 2D environment under strict resource constraints. The agent trains neural network dynamics and distance predictor models online, and plans actions using a receding-horizon Cross-Entropy Method (CEM) planner with an adaptive exploration mechanism for escaping local traps.

## Method Overview

- **Dynamics model**: predicts the next observation given the current observation and action
- **Distance predictor**: estimates the distance to the goal from an observation
- **CEM planner**: samples action sequences, rolls them out through the learned dynamics, and selects elite trajectories minimising predicted terminal distance
- **Adaptive exploration**: monitors distance stagnation and observation-space displacement to dynamically switch to a high-variance CEM regime, escaping local traps without costly resets
- **Budget management**: all operations (environment steps, resets, compute time) consume a shared monetary budget; training ends automatically before the budget is exhausted

## Setup

Requires Python 3.12.
```bash
pip install -r requirements.txt
```

## Usage
```bash
python robot-learning.py
```

This runs the full training and testing pipeline. The agent trains until the budget is exhausted, then switches to testing mode and attempts to reach the goal as quickly as possible.

## Project Structure
```
robot-learning.py   # Main loop: training and testing orchestration
robot.py            # Agent implementation (dynamics model, CEM planner, replay buffer)
environment.py      # Environment dynamics
config.py           # Mode selection (development / evaluation)
constants.py        # Budget costs and action constraints
```
