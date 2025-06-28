---
layout: page
title: Overview
permalink: /
weight: 1
---

<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: center; gap: 1em; padding: 2em">
  <img style="width: 30%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/ieee-logo.png?raw=true" alt="IEEE Logo">
  <img style="width: 20%;" src="https://github.com/Open-Finance-Lab/FinRL_Contest_2025/blob/main/docs/assets/logos/columbiau.jpeg?raw=true" alt="Columbia Logo">
</div>

### Thanks to the AI4Finance Foundation open source community for their support.

## Introduction

As AI continues to advance at a fast pace, more FinAI agents are being developed for the finance sector, such as [FinRL trading agents](https://berylventures.com/spotlights) [1,2,3], FinGPT agents [4,5] with multimodal capabilities [6], and regulatory reporting agents [7]. The FinAI Contest 2025 encourages the development of open FinAgents based on the frameworks FinRL [2,3] and FinGPT [4].


The FinAI Contest 2025 explores and evaluates the capability of machine learning methods in finance, with the following features:
1. **FinRL-DeepSeek**. Seeking alpha signals is crucial for trading strategies, in particular for strategies driven by alternative data and quantamental approach, respectively. Is it possible for an individual to ask Warren Buffett for value-investing advice, consult a risk manager to identify red flags in SEC filings, or engage a sentiment analyst to interpret the tone of market news — all timely and on demand?  AI agents are making this happen. These FinGPT-powered agents, such as a Buffett agent, a sentiment analysis agent, and a risk management agent, form a professional investment team to extract actionable signals from financial documents. In this task, we encourage participants to explore FinGPT-engineered signals and integrate them into a FinRL trading agent for crypto trading.
2. **FinGPT Agents in Real Life**. AI agents have seen rapid development and have been applied to various financial tasks recently. They have been applied to [financial analysis and accounting](https://openai.com/solutions/ai-for-finance/) and are capable of [analyzing SEC filings](https://fintool.com/press/fintool-outperforms-analysts-sec-filings). Researchers also show that [large language models (LLMs) can pass CFA Level I and II exams](https://aclanthology.org/2024.emnlp-industry.80/), achieving performance above the human average. While BloombergGPT is the first financial LLM pre-trained on large-scale financial data, it is no longer unmatched. Many open FinLLMs, such as FinGPT [4], have outperformed BloombergGPT on public benchmarks. It is not hard to build your own FinGPT agent that rivals or surpasses BloombergGPT and serves as professional financial assistant. This task encourages participants to fine-tune open LLMs and develop FinAgents for financial analytics, including the CFA exam, BloombergGPT's public benchmark tasks, and XBRL tasks.


3. **FinRL-DeFi**. Decentralized Finance (DeFi) is reshaping the crypto economy by enabling peer-to-peer trading, lending, and liquidity provision without banks, brokers, or intermediaries. As a core component of DeFi, the automated market makers (AMMs) act as liquidity providers (LPs) and replace order books with liquidity pools. However, liquidity provision is complex and risky. For example, impermanent loss can occur for LPs when the price of assets in a liquidity pool diverges from their initial value. LPs must actively manage price ranges, balance transaction fees, and mitigate impermanent loss. How can we develop an intelligent LP that adapts to market dynamics in DeFi? In this contest, we challenge participants to develop reinforcement learning agents that act as LPs [8], dynamically adjusting their liquidity positions in response to market conditions. 

We design three tasks: (1) FinRL-DeepSeek for Crypto Trading, (2) FinGPT Agents in Real Life, and (3) FinRL-DeFi. These challenges allow contestants to participate in various financial tasks and contribute to open finance using state-of-the-art technologies. We welcome students, researchers, and engineers who are passionate about finance and machine learning to partake in the contest.

## Tasks
Each team can choose to participate in one or more tasks. The prizes will be awarded for each task.

### Task I FinRL-DeepSeek for Crypto Trading:
This task is to develop crypto trading agents by integrating LLM-generated signals in FinRL, using financial news and market data. Participants can build upon the FinRL-DeepSeek project (e.g., with new prompts, new ways to inject LLM-processed news signals into the RL agent, new RL algorithms like GRPO) or explore more computationally intensive directions, such as adapting variants of the DeepSeek R1 training method to this crypto trading task.

**Datasets**
We will provide the second-level LOB data and financial news for Bitcoin. Participants are permitted to use additional external datasets.


### Task II FinGPT Agents in Real Life:
This task is to fine-tune LLMs and develop financial agents to interpret the professional language of finance. Participants are expected to train or fine-tune their LLMs to perform tasks in the three domains: the CDM, the MOF, and XBRL. We encourage participants to use LoRA and reinforcement fine-tuning.
* **CFA exam**: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
* **BloombergGPT** [6]: Compare the performance of your model with BloombergGPT on its public financial benchmarks.
* **XBRL**: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.

**Datasets**
We will provide the data sources for the three domains so that participants can collect data themselves. Participants can collect these data to fine-tune their LLMs and develop their agents. The full question sets for evaluation will be released during the evaluation period.


### Task III FinRL-DeFi:
The task is to develop RL agents for decentralized liquidity provisioning on Uniswap v3. The agent simulates a Liquidity Provider (LP) who must decide, at each time step, whether to maintain or rebalance their liquidity position. 

**Datasets**
We will provide hourly data for the WETH/USDC Uniswap v3 pool (0.05% fee tier), covering the period from May 5, 2021 at 1:00 am to 29 January 2024 at 7:00 pm. The dataset is sourced from the Uniswap Ethereum subgraph and resampled into an evenly spaced hourly series.


<p style="font-size: 10px;">
[1] Keyi Wang, Nikolaus Holzer, Ziyi Xia, Yupeng Cao, Jiechao Gao, Anwar Walid, Kairong Xiao, and  Xiao-Yang Liu Yanglet. FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents. arXiv preprint arxiv.org/abs/2504.02281, 2025.
</p>
<p style="font-size: 10px;">
[2] Xiao-Yang Liu, Ziyi Xia, Jingyang Rui, Jiechao Gao, Hongyang Yang, Ming Zhu, Christina Wang, Zhaoran Wang, and Jian Guo. FinRL-Meta: Market environments and benchmarks for data-driven financial reinforcement learning. Advances in Neural Information Processing Systems 35, 1835-1849, 2022.
</p>
<p style="font-size: 10px;">
[3] Xiao-Yang Liu, Hongyang Yang, Qian Chen, Runjia Zhang, Liuqing Yang, Bowen Xiao, and Christina Dan Wang. FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance. Deep Reinforcement Learning Workshop, NeurIPS. 2020.
</p>
<p style="font-size: 10px;">
[4] Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, and Daochen Zha. FinGPT: Democratizing internet-scale data for financial large language models. Workshop on Instruction Tuning and Instruction Following, NeurIPS 2023.
</p>
<p style="font-size: 10px;">
[5] Felix Tian, Ajay Byadgi, Daniel Kim, Daochen Zha, Matt White, Kairong Xiao, Xiao-Yang Liu. Customized FinGPT Search Agents Using Foundation Models. ACM International Conference on AI in Finance, 2024.
</p>
<p style="font-size: 10px;">
[6] Xiao-Yang Liu Yanglet, Yupeng Cao, and Li Deng. Multimodal financial foundation models (MFFMs): Progress, prospects, and challenges.  arXiv preprint arxiv.org/abs/2506.01973, 2025.
</p>
<p style="font-size: 10px;">
[7] Shijie Han, Haoqiang Kang, Bo Jin, Xiao-Yang Liu, Steve Yang. XBRL Agent: Leveraging Large Language Models for Financial Report Analysis. ACM International Conference on AI in Finance, 2024.
</p>
<p style="font-size: 10px;">
[8] Haonan Xu and Alessio Brini. Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning. arXiv: 2501.07508, 2025.
</p>
<p style="font-size: 10px;">
[9] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann. BloombergGPT: A Large Language Model for Finance. arXiv: 2303.17564, 2023.
</p>

## Contact
Contact email: [finrlcontest@gmail.com](mailto:finrlcontest@gmail.com)

Contestants can communicate any questions on 
* [Discord](https://discord.gg/dJY5cKzmkv).
* WeChat Group:
<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: left; gap: 1em; padding: 2em">
  <img style="width: 20%;" src="https://github.com/Open-Finance-Lab/FinAI_Contest_2025/blob/main/docs/assets/pictures/wechat_group.jpg?raw=true" alt="wechat group">
</div>




