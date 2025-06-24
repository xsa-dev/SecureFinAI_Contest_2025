# Task 3 — DeFi Liquidity Provisioning

## Objective of the Task

Participants will develop intelligent agents for decentralized liquidity provisioning on Uniswap v3. The task simulates a Liquidity Provider (LP) who must decide, at each time step, whether to maintain or rebalance their liquidity position. This is framed as a reinforcement learning (RL) control problem, where the goal is to maximize trading fee revenue while minimizing impermanent loss (IL) and rebalancing (gas) costs.

The agents must learn to adjust price intervals dynamically in response to market changes, identifying optimal rebalancing strategies that outperform passive LP baselines. Participants are encouraged to focus on designing effective state representations, reward functions, and policies under uncertainty, using Deep RL or other sequential decision-making approaches.

This competition is based on the research paper:  
*"Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning"* ([arXiv:2501.07508](https://arxiv.org/abs/2501.07508)).  
An accompanying codebase implementing the baseline strategy and environment is available here:  
https://github.com/Alessiobrini/deeprl-liquidity-provision-uniswapv3

Participants may use this implementation as a starting point for their solutions.

---

## Datasets to Be Provided

### Training Dataset

Hourly data for the WETH/USDC Uniswap v3 pool (0.05% fee tier), covering the period from **May 5, 2021 at 1:00 am to January 29, 2024 at 7:00 pm**. The dataset is sourced from the Uniswap Ethereum subgraph and resampled into an evenly spaced hourly series.

Participants will be provided with the market price time series and a starter reinforcement learning environment (`Uniswapv3Env`) that constructs the state vector automatically at each time step. The environment internally computes the following state variables:

- **Spot price**
- **Tick index**  
  (Mapped from price using log spacing: *i = floor[ log(pₜ) / log(1.0001) ]*)
- **Liquidity interval width**  
  (Selected in the previous step by the agent)
- **Liquidity level**  
  (Capital deployed, computed from the AMM formula given the selected price range)
- **Exponentially weighted volatility**  
  (Estimated using α = 0.05)
- **Moving averages**:
  - 24-period MA (short-term trend)
  - 168-period MA (long-term trend)
- **Bollinger Bands**  
  (Upper, middle, and lower bands using T3 smoothing)
- **Technical indicators computed with TA-Lib**:
  - ADXR (Average Directional Movement Index Rating)
  - BOP (Balance of Power)
  - DX (Directional Movement Index)

### Testing Dataset (Held-Out)

The final out-of-sample window, from **January 29, 2024 at 8:00 pm to April 29, 2024 at 7:00 pm**, will be held out for evaluation.

### Environment Code

A simplified RL environment will be provided based on the codebase at  
https://github.com/Alessiobrini/deeprl-liquidity-provision-uniswapv3,  
using the Gymnasium interface and Stable Baselines3 PPO implementation.

---

## What Participants Should Submit

- Complete code for training and inference (scripts, classes, and configurations)
- Trained model weights
- An evaluation script (e.g., `evaluate_agent.py`) that runs inference on the provided test environment on the test dataset (**January 29, 2024 at 8:00 pm to April 29, 2024 at 7:00 pm**)
- A short README file with setup and usage instructions
- Optionally: Docker container with all dependencies for reproducibility

---

## Evaluation Metrics and Methodology

Submissions will be evaluated on a **fixed held-out testing period**:  
**November 28, 2023 to January 29, 2024**, using the environment and data provided in the starter kit.

The evaluation will focus on the **cumulative performance** of the agent over this entire test window.

The main metric is the **cumulative reward**, computed at each time step as:

> **Reward = Trading Fees − LVR Penalty − Gas Cost**

Where:
- **Trading Fees** are collected when swaps occur within the active price range;
- **LVR Penalty** (Loss-Versus-Rebalancing) represents the opportunity cost of capital locked in the AMM compared to alternative uses;
- **Gas Cost** is incurred when rebalancing the liquidity position.

The agent's performance will be compared against a **passive LP baseline**, which rebalances its price range at fixed 500-hour intervals without adapting to market dynamics.

**Only the final cumulative reward over the evaluation window will be used to rank submissions.**

---

## Additional Requirements or Assumptions

### Action Space

Agents must operate in a **discrete action space**, choosing from predefined tick-width intervals (e.g., {0, 20, 50}). While the action space must remain discrete, **participants are free to customize** the number and values of tick-width options provided to the agent if they believe this improves performance.

### Environment and Reward Function

The environment implementation provided in the starter kit **must remain unchanged**. In particular:

- The **reward function must not be modified**. It is already implemented in the environment and reflects the net value of an LP position after fees, losses, and gas costs.
- The **gas fee is fixed at $5 per rebalancing** and must remain unchanged.

### Feature Engineering and External Data

Participants are encouraged to extend the **state (feature) space**, including integrating external data such as mempool snapshots or centralized exchange prices. However, they must:

- Preserve the **existing state variables from the baseline as a subset** of their feature space.
- Ensure **reproducibility** and clearly document any external data used.

### No Lookahead Bias

Agents may only use information available at the current or past timesteps.  
**Future information from the test set is strictly prohibited.**
