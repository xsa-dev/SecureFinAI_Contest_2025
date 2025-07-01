# FinAI Contest 2025
This repository contains the starter kit and tutorials for the FinAI Contest 2025.

## Outline
  - [Tutorial](#tutorial)
  - [Task 1 FinRL DeepSeek for Crypto Trading](#task-1-finrl-deepseek-for-crypto-trading)
  - [Task 2 FinGPT Agents in Real Life](#task-2-fingpt-agents-in-real-life)
  - [Task 3 FinRL-DeFi](#task-2-finrl-defi)
  - [Paper Submission Requirement](#paper-submission-requirement)
  - [Resources](#resources)

## Tutorial
Please explore [FinAI Contest Documentation](https://finrl-contest.readthedocs.io/en/latest/) for task 1, 2 and 3, and [FinLoRA](https://finlora-docs.readthedocs.io/en/latest/) Documentation for task 2.

We also welcome questions for these documentations and will update their FAQs.

Here we also provide some demos for FinRL:
| Task | Model | Environment | Dataset | Link |
| ---- |------ | ----------- | ------- | ---- |
| FinRL-AlphaSeek for Crypto Trading @ [FinRL Contest 2025](https://open-finance-lab.github.io/FinRL_Contest_2025/)| Ensemble | Crypto Trading Environment | LOB | [Baseline solution](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Task_2_FinRL_AlphaSeek_Crypto) |
| FinRL-DeepSeek for Stock Trading @ [FinRL Contest 2025](https://open-finance-lab.github.io/FinRL_Contest_2025/)| PPO, CPPO | Stock Trading Environment | OHLCV, news | [Baseline solution](https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Task_1_FinRL_DeepSeek_Stock) |
| Crypto Trading @ [FinRL Contest 2024](https://open-finance-lab.github.io/finrl-contest-2024.github.io/)| Ensemble | Crypto Trading Environment | LOB | [Baseline solution](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_1_starter_kit) |
| Stock Trading @ NeurIPS 2018 Workshop | DDPG | Stock Trading Environment | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinAI_Contest_2025/tree/main/Tutorials/Stock_Trading_NeurIPS2018) for [Paper](https://arxiv.org/abs/1811.07522) |
| Stock Trading @ [FinRL Contest 2023](https://open-finance-lab.github.io/finrl-contest.github.io/)| PPO | Stock Trading Environment | OHLCV | [Baseline solution](https://github.com/Open-Finance-Lab/FinAI_Contest_2025/tree/main/Tutorials/FinRL_Contest_2023_Task_1_baseline_solution) |
| Stock Trading | PPO | Stock Trading Environment | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinAI_Contest_2025/blob/main/Tutorials/FinRL_stock_trading_demo.ipynb) |
| Stock Trading | Ensemble | Stock Trading Environment | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinAI_Contest_2025/blob/main/Tutorials/FinRL_Ensemble_StockTrading_ICAIF_2020.ipynb) for [paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3690996)|
| Sentiment Analysis with RLMF @ [FinRL Contest 2024](https://open-finance-lab.github.io/finrl-contest-2024.github.io/) | / | Stock Sentiment Environment | OHLCV, News | [Starter-Kit](https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Task_2_starter_kit)|
| Sentiment Analysis with Market Feedback | ChatGLM2-6B | -- | Eastmoney News | [Code](https://github.com/AI4Finance-Foundation/FinGPT/tree/master/fingpt/FinGPT_Sentiment_Analysis_v1/FinGPT_v1.0) |
| Stock Price Prediction | Linear Regression | -- | OHLCV | [Demo](https://github.com/Open-Finance-Lab/FinAI_Contest_2025/blob/main/Tutorials/Example_Linear_Regression.ipynb) |

## Task 1 FinRL-DeepSeek for Crypto Trading
This task is to develop crypto trading agents by integrating LLM-generated signals in FinRL, using financial news and market data. 

Task 1 Starter Kit is released [here](./Task_1_FinRL_DeepSeek_Crypto_Trading). It contains example code to run a baseline solution. 

## Task 2 FinGPT Agents in Real Life
This task encourages participants to fine-tune open LLMs and develop FinAgents for financial analytics, including the CFA exam, BloombergGPT’s public benchmark tasks, and XBRL tasks.

Task 2 Starter Kit is released [here](./Task_2_FinGPT_Agents_Real_Life). It provides the summary and statistics of question datasets which will be used to evaluate the submitted LLMs. Participants can collect raw data themselves according to data sources or utilize other datasets to train or fine-tune their models.



## Task 3 FinRL-DeFi
The task is to develop RL agents for decentralized liquidity provisioning on Uniswap v3. The agent simulates a Liquidity Provider (LP) who must decide, at each time step, whether to maintain or rebalance their liquidity position.

Task 3 Starter Kit is released [here](./Task_3_FinRL_DeFi). 

## Paper Submission Requirements
Each team should submit short papers with 3 complimentary pages and up to 2 extra pages, including all figures, tables, and references. The paper submission is through [the special track SecureFinAI](https://www.cloud-conf.net/cscloud/2025/cscloud/cfp_files/SecureFinAI_CFP.pdf) and should follow its instructions. Please include “FinAI Contest Task 1/2/3” in your abstract.

## Resources
Useful materials and resources for contestants:
* [FinAI Contest Docs](https://finrl-contest.readthedocs.io/en/latest/index.html)
* [FinRL Trading Agents Blog](https://berylventures.com/spotlights)
* FinRL Contests
  * FinRL Contest 2025: [Contest Website](https://open-finance-lab.github.io/FinRL_Contest_2025/); [Github](https://github.com/Open-Finance-Lab/FinRL_Contest_2025)
  * FinRL Contest 2024: [Contest Website](https://open-finance-lab.github.io/finrl-contest-2024.github.io/); [Github](https://github.com/Open-Finance-Lab/FinRL_Contest_2024)
  * FinRL Contest 2023: [Contest Website](https://open-finance-lab.github.io/finrl-contest.github.io/); [Github](https://github.com/Open-Finance-Lab/FinRL_Contest_2023)
* [FinLoRA](https://finlora-docs.readthedocs.io/en/latest/)
* [FinRL-DeepSeek](https://github.com/benstaf/FinRL_DeepSeek)
* Regulations Challenges at COLING 2025: [Contest Website](https://coling2025regulations.thefin.ai/); [Github](https://github.com/Open-Finance-Lab/Regulations_Challenge_COLING_2025)
* [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
* [FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta)
* [FinRL Tutorials](https://github.com/AI4Finance-Foundation/FinRL-Tutorials)
