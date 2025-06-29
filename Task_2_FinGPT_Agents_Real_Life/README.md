## # FinAI Contest Task 12 - FinGPT Agents in Real Life

### 🧠 Task Overview

This task is to fine-tune LLMs and develop financial agents for financial analytics, including the CFA exam, BloombergGPT’s public benchmark tasks, and XBRL tasks. We encourage participants to use LoRA and reinforcement fine-tuning.

* **CFA exam**: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
* **BloombergGPT**: Compare the performance of your model with BloombergGPT on its public financial benchmarks.
* **XBRL**: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.

#### 🎯 Objective
Your primary goal is to fine-tune or train a language model for financial analytics across the CFA exam, BloombergGPT benchmark tasks, and XBRL tasks. You may also enhance your agent by integrating external tools, such as a retrieval-augmented knowledge base (RAG), to improve its analytical and question-answering capabilities.

#### 💡 What You Need To Do

1. **Collect and Prepare Your Raw Training Data**  
   Participants need to collect raw data given the source provided below. 

2. **Develop FinGPT Agents**  
   Use your collected data to fine-tune your own LLM for financial analytics. You can use FinGPT framework to fine-tune your model. We encourage participants to use LoRA and reinforcement fine-tuning. You can also enhance your agent by integrating external tools, such as RAG.

3. **Submit Your Agent**  
   Submit your agent following the competition guidelines. Make sure your model is:
   - Capable of answering complex domain-specific questions.
   - Robust in interpreting structured data and reasoning over it.

4. **Benchmarking Phase**  
   After submission, we will use our question sets to evaluate your model's performance on CFA exams, BloombergGPT benchmark tasks, and XBRL tasks.

---

### 📊 Question Dataset Overview

This dataset contains question-answer pairs collected and organized for evaluating model capabilities across CFA exams, BloombergGPT benchmark tasks, and XBRL tasks.

#### CFA Exams

#### BloombergGPT Public Benchmark Datasets
| **Data Category**                         | **Size** | **Metrics**     | **Data Source**                                                                 |
|------------------------------------------|----------|----------------|---------------------------------------------------------------------------------|

#### 📁 XBRL Dataset

| **Data Category**                         | **Size** | **Metrics**     | **Data Source**                                                                 |
|------------------------------------------|----------|----------------|---------------------------------------------------------------------------------|

##### 📂 How to Download XBRL Filings

To construct or extend your training dataset with real-world XBRL filings, participants may utilize the following data sources:

##### 📥 1. Company-Level Financial Statements

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

##### ⚙️ 2. XBRL Terminology & Standards

As a starting point, you may also use the provided web crawling script to automate the retrieval of XBRL-related documents from the [XBRL International Glossary](https://www.xbrl.org/guidance/xbrl-glossary/). This source provides standardized definitions and explanations of XBRL terms. These documents help the model better understand the semantics and structure of XBRL as a framework.

- 📎 Provided Code: [xbrl_webcrawl.ipynb](./xbrl_webcrawl.ipynb)

This script offers a basic template to:
- Scrape and parse glossary terms.
- Crawl linked resources or downloadable attachments related to XBRL filings.
- Extend it further for large-scale automated crawling from additional sources (e.g., SEC bulk data feeds, company repositories, etc.).

> 💡 This data helps build XBRL term comprehension tasks, enabling models to understand and explain technical terms used in filings. Participants are encouraged to adapt and extend the script to suit their own dataset construction needs.

Note: We will additionally test on a subset of the FiNER-139 and FNXL datasets. Please use the batched versions provided in this folder for fine-tuning to avoid overfitting. To test, please use the code from https://github.com/Open-Finance-Lab/FinLoRA/blob/main/test/xbrl.py.

---

#### 📦 Dataset Summary

| **Domain** | **Total QA Pairs** |
|------------|--------------------|
| CDM        |                 |
| BloombergGPT  |                 |
| XBRL       |                |

### 📊 Metrics
The model evaluation is the average score of all tasks. 

#### 📘 Note for Participants

Participants are encouraged to use the above sources as a starting point to construct their own training datasets. Your model's performance will strongly depend on the quality and comprehensiveness of your self-collected training data. These sources can help you build a rich and task-aligned dataset for model training, ensuring better performance on regulatory reasoning and question answering.

To ensure fair comparison and practical deployment, it is recommended that the model size should not exceed 8B parameters.

