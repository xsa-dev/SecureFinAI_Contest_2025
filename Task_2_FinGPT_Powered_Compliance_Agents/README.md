## FinAI Contest Task 2 - FinGPT-Powered Compliance Agents

### 🧠 Task Overview

This task focuses on developing FinGPT agents for financial compliance and regulatory scenarios. Participants will build LLM-based agents capable of handling SEC filings analysis, regulatory compliance, sentiment analysis, antitrust reasoning, patent analysis, and financial audio processing. We encourage participants to use LoRA and reinforcement fine-tuning.

- **SEC Filings Analysis**: Analyze and extract insights from SEC filings, including XBRL data processing, financial statement Q&A, and mathematical reasoning.
- **Regulatory Compliance**: Handle real-time financial data retrieval, sentiment analysis, antitrust and copyright reasoning, patent and IP protection analysis.
- **Multimodal Processing**: Process various data types including text, structured data, and audio for comprehensive financial analysis.

#### 🎯 Objective

Your primary goal is to fine-tune or train a language model for financial compliance and regulatory tasks, covering SEC filings analysis, regulatory compliance, sentiment analysis, antitrust reasoning, patent analysis, and financial audio processing. You may also enhance your agent by integrating external tools, such as a retrieval-augmented knowledge base (RAG), to improve its analytical and question-answering capabilities.

#### 💡 What You Need To Do

1. **Collect and Prepare Your Raw Training Data**  
   Participants need to collect raw data given the sources provided below.

2. **Develop FinGPT Compliance Agents**  
   Use your collected data to fine-tune your own LLM for financial compliance and regulatory tasks. You can use FinGPT framework to fine-tune your model. We encourage participants to use LoRA and reinforcement fine-tuning. You can also enhance your agent by integrating external tools, such as RAG. You can view [**FinLoRA documentation**](https://finlora-docs.readthedocs.io/en/latest/index.html) to learn more about LoRA and some financial tasks.

3. **Submit Your Agent**  
   Submit your agent following the competition guidelines. Make sure your model is:

   - Capable of analyzing SEC filings and extracting structured information.
   - Proficient in regulatory compliance reasoning and analysis.
   - Robust in interpreting multimodal financial data and reasoning over it.

4. **Benchmarking Phase**  
   After submission, we will use our question sets to evaluate your model's performance across SEC filings analysis and regulatory compliance tasks.

---

### 📊 Question Set Overview

These question sets contain question-answer pairs collected and organized for evaluating model capabilities across SEC filings analysis and regulatory compliance tasks. These question sets are sampled from the test split of the datasets, which are used to benchmark your agent's performance. You **SHOULD NOT** use it or the entire test split for fine-tuning or training.

#### SEC Filings Analysis

| **Task** | **Dataset** | **Size** | **Metrics** | **Description** | **Source** |
| -------- | ----------- | -------- | ----------- | --------------- | ----------|
| Financial Q&A | FinanceBench | 150 | BERTScore | Open-book financial Q&A on company filings based on OCR-processed annual reports. | https://huggingface.co/datasets/PatronusAI/financebench |
| XBRL Tag Extraction | XBRL Analysis | 1k | Accuracy, F1-Score | Extract specific XBRL tags from raw XBRL text segments given natural language descriptions. | https://huggingface.co/datasets/wangd12/XBRL_analysis |
| XBRL Value Extraction | XBRL Analysis | 12k | Accuracy, F1-Score | Extract numeric values from XBRL text segments given natural language descriptions. | https://huggingface.co/datasets/wangd12/XBRL_analysis |
| XBRL Formula Construction | XBRL Analysis | 1k | Accuracy, F1-Score | Select relevant facts and tags from XBRL data and construct standard financial formulas. | https://huggingface.co/datasets/wangd12/XBRL_analysis |
| XBRL Formula Calculation | XBRL Analysis | 1k | Accuracy, F1-Score | Substitute actual values into constructed formulas and compute final results. | https://huggingface.co/datasets/wangd12/XBRL_analysis |
| General Mathematics | Math Problems | 1k | Accuracy | Solve general mathematical problems related to ratio calculation and algebra. | https://github.com/Open-Finance-Lab/SecureFinAI_Contest_2025/tree/main/Task_2_FinGPT_Powered_Compliance_Agents |

#### Regulatory Compliance

| **Task** | **Dataset** | **Size** | **Metrics** | **Description** | **Source**                  |
| -------- | ----------- | -------- | ----------- | --------------- |-----------------------------|
| Financial Data Retrieval | Real-time Data | 331 | Accuracy, F1-Score | Real-time retrieval from active web pages and open-domain search on company financials. | Yahoo Finance and Bloomberg |
| Sentiment Analysis | Financial Sentiment | 4.8k | Accuracy, F1-Score | Aspect-specific sentiment classification for financial texts (news, social media, transcripts, ESG, macro). | BloombergGPT FPB, FiQA SA   |
| Financial Audio | FinAudio | 1k | Word Error Rate | Automatic speech recognition for financial audio content. | SPGISpeech                        |

We will sample questions from the test split for each dataset for our evaluation.

---

### 📁 Data Sources and Collection

#### 📥 1. SEC Filings Data

You can manually retrieve XBRL filings for individual companies via the U.S. Securities and Exchange Commission (SEC):

1. Visit the [SEC EDGAR Company Search](https://www.sec.gov/edgar/searchedgar/companysearch).
2. Search by company name or ticker symbol.
3. Filter by filing types such as 10-K, 10-Q, etc.
4. Click on a specific filing.
5. Look for files with extensions like:
   - `.xml`
   - `.xsd`
   - `.xbrl`
   - or links labeled "Interactive Data".
6. Download the corresponding XBRL instance and taxonomy files.

> 💡 This method is ideal for collecting filings from specific companies or filing types in a controlled manner.

#### ⚙️ 2. XBRL Terminology & Standards

As a starting point, you may also use the provided web crawling script to automate the retrieval of XBRL-related documents from the [XBRL International Glossary](https://www.xbrl.org/guidance/xbrl-glossary/). This source provides standardized definitions and explanations of XBRL terms.

- 📎 Provided Code: [xbrl_webcrawl.ipynb](./xbrl_webcrawl.ipynb)

This script offers a basic template to:

- Scrape and parse glossary terms.
- Crawl linked resources or downloadable attachments related to XBRL filings.
- Extend it further for large-scale automated crawling from additional sources.

> 💡 This data helps build XBRL term comprehension tasks, enabling models to understand and explain technical terms used in filings.

#### 📊 3. Financial Data APIs

For real-time financial data retrieval tasks, participants can utilize various financial data APIs and web sources:

- Yahoo Finance API
- Alpha Vantage API
- Financial news websites
- Company investor relations pages

#### 🎵 4. Financial Audio Data

For the FinAudio task, participants could collect financial audio content such as:

- Earnings call recordings
- Financial news broadcasts
- Investor presentations
- Financial podcasts

---

### 🔧 Training and Fine-tuning

- 📎 Provided Code: [task_2_finetune.ipynb](./task_2_finetune.ipynb)

This script offers a basic template for fine-tuning:

- The notebook is simplified for quick start.
- For more detailed instructions, please check the tutorials under the FinLoRA docs here: https://finlora-docs.readthedocs.io/en/latest/index.html.
- The full process for a **simplified** Buffett Agent model we created can be found here: https://finlora-docs.readthedocs.io/en/latest/tutorials/buffett_agent.html.

Note: We will additionally test on subsets of various financial datasets. Please use the batched versions provided in this folder for fine-tuning to avoid overfitting.

---

### 📥 Submission Requirement

Submit a Hugging Face repository with model weights, scripts, and all necessary files for inference. Make sure your submission includes:

- Model weights and configuration files
- Inference scripts for all subtasks
- Requirements.txt or environment.yml
- Clear documentation on how to run inference
- Any custom libraries or preprocessing code

### 📊 Metrics

The model evaluation in each domain is the average score of all tasks.

#### 📘 Note for Participants

Participants are encouraged to use the above sources as a starting point to construct their own training/fine-tuning datasets. Your model's performance will strongly depend on the quality and comprehensiveness of your self-collected training data. These sources can help you build a rich and task-aligned dataset for model training, ensuring better performance on regulatory reasoning and question answering.

To ensure fair comparison and practical deployment, it is recommended that the model size should not exceed 8B parameters.

---

### 📚 References

[1] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann. BloombergGPT: A Large Language Model for Finance. arXiv: 2303.17564, 2023.

[2] Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, Daochen Zha. FinGPT: Democratizing internet-scale data for financial large language models. Workshop on Instruction Tuning and Instruction Following, NeurIPS, 2023.

[3] Felix Tian, Ajay Byadgi, Daniel S Kim, Daochen Zha, Matt White, Kairong Xiao, Xiao-Yang Liu. Customized FinGPT search agents using foundation models. Proceedings of the 5th ACM International Conference on AI in Finance, pages 469--477, 2024.

[4] Yinheng Li, Shaofei Wang, Han Ding, Hang Chen. Large Language Models in Finance: A Survey. ACM International Conference on AI in Finance, pages 374–382, 2023.

[5] Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M. Mulvey, H. Vincent Poor, Qingsong Wen, Stefan Zohren. A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges. arXiv preprint arXiv:2406.11903, 2024.

[6] Shengyuan Colin Lin, Felix Tian, Keyi Wang, Xingjian Zhao, Jimin Huang, Qianqian Xie, Luca Borella, Matt White, Christina Dan Wang, Kairong Xiao, Xiao-Yang Liu Yanglet, Li Deng. Open FinLLM Leaderboard: Towards Financial AI Readiness. International Workshop on Multimodal Financial Foundation Models (MFFMs) at 5th ACM International Conference on AI in Finance, 2024.

[7] Haochen Sun, Jason Li, Hongyang Zhang. zkLLM: Zero Knowledge Proofs for Large Language Models. ACM SIGSAC Conference on Computer and Communications Security, 2024.

[8] Xiao-Yang Liu, Ziyi Xia, Hongyang Yang, Jiechao Gao, Daochen Zha, Ming Zhu, Christina Dan Wang, Zhaoran Wang, Jian Guo. Dynamic datasets and market environments for financial reinforcement learning. Machine Learning - Nature, 2024.

[9] Cao, Yupeng, et al. "FinAudio: A Benchmark for Audio Large Language Models in Financial Applications." arXiv preprint arXiv:2503.20990 (2025).

