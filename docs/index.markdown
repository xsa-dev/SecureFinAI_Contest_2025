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

### Thanks to the AI4Finance Foundation Open-Source Community support.

## Introduction

As artificial intelligence (AI) continues to advance rapidly, more and more AI agents are being developed and applied to various financial tasks, such as trading agents [1,2,3], search agents [4], and regulatory reporting agents [5]. The FinAI Contest 2025 aims to encourage the development of advanced financial agents and benchmark their performance across different financial tasks.

The FinRL Contest 2025 explores and evaluates the capability of machine learning methods in finance, with the following features:
1. **FinRL-DeepSeek**. Generating alpha signals is crucial for making informed trading decisions. As individual investors without resources, what if we can ask Warren Buffett for value-investing advice, consult a risk manager to identify red flags in SEC filings, or engage a sentiment analyst to interpret the tone of market news — all timely and on demand. AI agents make this possible. These LLM-powered agents, such as a Warren Buffett agent, a sentiment analysis agent, and a risk management agent, form a professional investment team to extract actionable signals from financial documents. In this contest, we encourage participants to explore LLM-generated signals and integrate them into FinRL for crypto trading.
2. **FinGPT Agents**. AI agents have seen rapid development and have been applied to various financial tasks recently. However, can they truly serve as professional financial assistants in real life? Imagine an AI agent that can help us prepare for CFA exams, analyze the SEC filings, or navigate financial contracts. Despite this promise, there still remain [doubts that AI cannot perform financial analytics](https://www.cnbc.com/2023/12/19/gpt-and-other-ai-models-cant-analyze-an-sec-filing-researchers-find.html). This task encourages participants to take on that challenge: fine-tune LLMs and develop financial agents for financial analytics, including the CFA exam, BloombergGPT's public benchmarks, and XBRL.
3. **Decentralized Finance (DeFi)**. DeFi is an emerging blockchain-based peer-to-peer financial ecosystem and has transformed the crypto market. It enables users to trade, lend, and borrow assets without intermediaries like brokers. Instead, smart contracts automate these processes in a trustless and permissionless manner. Unlike traditional centralized crypto exchanges, DeFi protocols (e.g., Uniswap v3) employ automated market maker (AMM) models, which replace order books with liquidity pools. The introduction of concentrated liquidity in Uniswap v3 significantly improved capital efficiency but made liquidity provision more complex and risky. LPs must actively manage price ranges, balance transaction fees, and mitigate impermanent loss. In this contest, we encourage participants to develop reinforcement learning agents that act as LPs [7], dynamically adjusting liquidity positions in response to market conditions.

We design two tasks: (1) FinRL-DeepSeek for Crypto Trading and (2) FinAgent using Fine-Tuning. These challenges allow contestants to participate in various financial tasks and contribute to open finance using state-of-the-art technologies. We welcome students, researchers, and engineers who are passionate about finance and machine learning to partake in the contest.

## Tasks
Each team can choose to participate in one or more tasks. The prizes will be awarded for each task.

### Task I FinRL-DeepSeek for Crypto Trading:
This task is to develop crypto trading agents by integrating LLM-generated signals in FinRL, using financial news and market data. Participants can build upon the FinRL-DeepSeek project (e.g., with new prompts, new ways to inject LLM-processed news signals into the RL agent, new RL algorithms like GRPO) or explore more computationally intensive directions, such as adapting variants of the DeepSeek R1 training method to this crypto trading task.

**Datasets**



### Task II FinGPT Agents in Real Life:
This task is to fine-tune LLMs and develop financial agents to interpret the professional language of finance. Participants are expected to train or fine-tune their LLMs to perform tasks in the three domains: the CDM, the MOF, and XBRL. We encourage participants to use LoRA and reinforcement fine-tuning.
* **CFA exam**: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
* **BloombergGPT** [6]: Compare the performance of your model with BloombergGPT on its public financial benchmarks.
* **XBRL**: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.

**Datasets**

### Task III FinRL for DeFi:
The task is to develop RL agents for decentralized liquidity provisioning on Uniswap v3. The agent simulates a Liquidity Provider (LP) who must decide, at each time step, whether to maintain or rebalance their liquidity position. 

**Datasets**


<p style="font-size: 10px;">
[1] X.-Y. Liu, Z. Xia, H. Yang, J. Gao, D. Zha, M. Zhu, Christina D. Wang*, Zhaoran Wang, and Jian Guo. Dynamic datasets and market environments for financial reinforcement learning. Machine Learning Journal, Springer Nature, 2023.
</p>
<p style="font-size: 10px;">
[2] X.-Y. Liu, Z. Xia, J. Rui, J. Gao, H. Yang, M. Zhu, C. Wang, Z. Wang, J. Guo. FinRL-Meta: Market environments and benchmarks for data-driven financial reinforcement learning. NeurIPS, Special Track on Datasets and Benchmarks, 2022.
</p>
<p style="font-size: 10px;">
[3] Keyi Wang, Nikolaus Holzer, Ziyi Xia, Yupeng Cao, Jiechao Gao, Anwar Walid, Kairong Xiao, and Xiao-Yang Liu Yanglet. Parallel Market Environments for FinRL Contests. arXiv: 2504.02281 (2025).
</p>
<p style="font-size: 10px;">
[4] Felix Tian, Ajay Byadgi, Daniel Kim, Daochen Zha, Matt White, Kairong Xiao, Xiao-Yang Liu. Customized FinGPT Search Agents Using Foundation Models. ACM International Conference on AI in Finance, 2024.
</p>
<p style="font-size: 10px;">
[5] Shijie Han, Haoqiang Kang, Bo Jin, Xiao-Yang Liu, Steve Yang. XBRL Agent: Leveraging Large Language Models for Financial Report Analysis. ACM International Conference on AI in Finance, 2024.
</p>
<p style="font-size: 10px;">
[6] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann. BloombergGPT: A Large Language Model for Finance. arXiv: 2303.17564 (2023).
</p>
<p style="font-size: 10px;">
[7] Haonan Xu and Alessio Brini. Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning. arXiv: 2501.07508 (2025).
</p>
## Contact
Contact email: [finrlcontest@gmail.com](mailto:finrlcontest@gmail.com)

Contestants can communicate any questions on 
* [Discord](https://discord.gg/dJY5cKzmkv).




