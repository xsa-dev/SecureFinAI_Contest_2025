---
layout: page
title: Overview
permalink: /
weight: 1
---

<div style="text-align: center; display: flex; width: 100%; justify-content: space-evenly; align-items: center; gap: 1em; padding: 2em">
  <img style="width: 30%;" src="https://github.com/Open-Finance-Lab/SecureFinAI_Contest_2025/blob/main/docs/assets/logos/acm_icaif.png?raw=true" alt="ACM ICAIF Logo">
  <img style="width: 20%;" src="https://github.com/Open-Finance-Lab/SecureFinAI_Contest_2025/blob/main/docs/assets/logos/columbiau.jpeg?raw=true" alt="Columbia Logo">
</div>

### Thanks to the AI4Finance Foundation open source community for their support.



## Introduction

As AI continues to advance at a fast pace, more FinAI agents are being developed for the finance sector, such as FinRL trading agents [1,2,3], FinGPT agents [4,5] with multimodal capabilities [6], and regulatory reporting agents [7]. The **Secure FinAI Contest 2025** encourages the development of FinAI agents based on the frameworks FinRL [2,3] and FinGPT [4].

The Secure FinAI Contest 2025 explores and evaluates the capability of machine learning methods in finance, with following features:

1. **FinRL-Transformer for Cryptocurrency Trading**. In the volatile cryptocurrency markets, timely interpretation of market sentiment and secure execution of trading strategies are critical. Cryptocurrency markets are highly sensitive to news headlines, tweets, regulatory shifts, and viral narratives. However, the massive amount of news, social media posts, and documents can overwhelm individual traders. Is it possible for an individual trader to ask a sentiment analyst to interpret market news and consult a risk manager to detect red flags in blockchain activity — all timely and on demand? AI agents are making this happen. These secure FinGPT-powered agents, such as a sentiment analysis agent and a risk management agent, form a professional investment team to extract actionable signals from financial news, tweets, and filings. 

2. **FinGPT Agents for SEC Filings Analysis**. AI agents have seen rapid development and have been applied to various financial tasks recently. They have been applied to financial analysis and accounting and are capable of analyzing SEC filings. Researchers show that large language models (LLMs) can pass CFA Level I and II exams, achieving performance above the human average. Many open FinLLMs, such as FinGPT [4], have outperformed BloombergGPT on public benchmarks. With our framework, it is possible to build your own secure FinGPT agent that rivals or surpasses BloombergGPT and serves as a professional financial assistant while maintaining data confidentiality and result verifiability.

3. **FinGPT Agents for Regulation and Compliance**. The regulation landscape in finance demands the highest levels of security, privacy, and compliance. Financial institutions must follow the complex regulatory requirements and protect sensitive data. Our contest challenges participants to develop FinGPT agents that can handle regulatory scenarios with zero-knowledge proof verification, ensuring that compliance checks and regulatory analysis can be performed without exposing underlying sensitive data.

We design three tasks: (1) FinRL-Transformer for Cryptocurrency Trading, (2) FinGPT Agents for SEC Filings Analysis, and (3) FinGPT Agents for Regulation and Compliance. These challenges allow contestants to participate in various financial tasks and contribute to secure finance using state-of-the-art technologies with privacy-preserving and verifiable computation frameworks. We welcome students, researchers, and engineers who are passionate about finance, machine learning, and security to partake in the contest.

## Tasks
Each team can choose to participate in one or more tasks. The prizes will be awarded for each task.

### Task I: FinRL-Transformer for Cryptocurrency Trading

This task is to develop crypto trading agents by integrating transformer-based architectures and LLM-generated signals in FinRL, using financial news and market data. Participants can build upon transformer architectures (e.g., with new attention mechanisms, new ways to inject LLM-processed news signals into the transformer agent, new optimization algorithms) or explore more computationally intensive directions, such as adapting variants of modern transformer training methods to this crypto trading task.

**Datasets**: We will provide the second-level LOB data and financial news for Bitcoin. Participants are permitted to use additional external datasets.

### Task II: FinGPT Agents for SEC Filings Analysis

This task is to fine-tune LLMs and develop financial agents to interpret the professional language of finance. Participants are expected to train or fine-tune their LLMs to perform tasks in three domains: the CFA exam, the BloombergGPT benchmarks, and XBRL. We encourage participants to use LoRA and reinforcement fine-tuning.

* **CFA exam**: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
* **BloombergGPT** [9]: Compare the performance of your model with BloombergGPT on its public financial benchmarks.
* **XBRL**: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.

**Datasets**: We will provide the data sources for the three domains so that participants can collect data themselves. Participants can collect these data to fine-tune their LLMs and develop their agents. The full question sets for evaluation will be released during the evaluation period.

### Task III: FinGPT Agents for Regulation and Compliance

This task focuses on developing FinGPT agents for regulatory and compliance scenarios in finance with zero-knowledge proof verification. The task covers multiple regulatory domains including financial data retrieval, sentiment analysis, antitrust and copyright compliance verification, patent and IP protection analysis, and financial audio processing for compliance monitoring.

**Datasets**: We will provide regulatory datasets covering various compliance scenarios including AML data, regulatory filings, compliance questionnaires, and audit trails, along with evaluation frameworks that incorporate zero-knowledge proof verification.


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
  <img style="width: 20%;" src="https://github.com/Open-Finance-Lab/SecureFinAI_Contest_2025/blob/main/docs/assets/pictures/wechat_group.jpeg?raw=true" alt="wechat group">
</div>




