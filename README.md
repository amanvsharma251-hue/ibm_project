# 🎓 College Admission Agent — RAG-Based AI Assistant
### Powered by IBM Granite on IBM watsonx.ai (Cloud Lite)

> **Problem Statement #4 — AICTE IBM Internship**
> A College Admission Agent powered by RAG (Retrieval-Augmented Generation) that streamlines the student admission process using IBM Cloud services and IBM Granite LLM.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [IBM Cloud Configuration](#ibm-cloud-configuration)
- [Running the Application](#running-the-application)
- [Knowledge Base](#knowledge-base)
- [Environment Variables](#environment-variables)

---

## Overview

The College Admission Agent is an AI-powered chatbot that helps prospective students navigate the college admission process. It uses **Retrieval-Augmented Generation (RAG)** to retrieve accurate, context-grounded information from institutional documents and generates natural language answers using **IBM Granite** — IBM's enterprise-grade LLM hosted on **IBM watsonx.ai**.

### Key Features
- 🔍 **RAG Pipeline** — retrieves relevant policy/FAQ chunks before answering
- 🤖 **IBM Granite LLM** — generates grounded, accurate responses
- 💬 **Conversational Memory** — maintains context across follow-up questions
- 📚 **Rich Knowledge Base** — covers policies, fees, deadlines, courses & FAQs
- 🖥️ **Streamlit UI** — clean, responsive chat interface
- 🔄 **Live KB Rebuild** — update knowledge base without code changes

---

## Architecture

```
Student Question
      │
      ▼
┌─────────────────────┐
│  Streamlit Web UI   │  ← app.py
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Agent Orchestrator│  ← src/agent.py
└────────┬────────────┘
         │
    ┌────┴──────────────┐
    │                   │
    ▼                   ▼
┌──────────┐    ┌────────────────────┐
│ RAG Core │    │  IBM Granite LLM   │
│(ChromaDB)│    │  via watsonx.ai    │
└──────────┘    └────────────────────┘
    │
    ▼
┌──────────────────────────┐
│  Knowledge Base (.txt)   │
│  - admission_policy.txt  │
│  - fee_structure.txt     │
│  - important_dates.txt   │
│  - courses_programs.txt  │
│  - faqs.txt              │
└──────────────────────────┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | IBM Granite (ibm/granite-3-8b-instruct) |
| **LLM Hosting** | IBM watsonx.ai (Cloud Lite — Free) |
| **LLM Framework** | LangChain + langchain-ibm |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 (local) |
| **Vector Store** | ChromaDB (local persistence) |
| **Document Loader** | LangChain Community loaders |
| **Web UI** | Streamlit |
| **Config** | python-dotenv |

---

## Project Structure

```
college_admission_agent/
├── app.py                    # Streamlit web application (entry point)
├── config.env                # ⚠️ Rename to .env — fill in your IBM credentials
├── .env.example              # Template for credentials
├── requirements.txt          # Python dependencies
│
├── src/
│   ├── agent.py              # Orchestrates RAG + LLM; AdmissionChatSession
│   ├── rag_engine.py         # Document loader, splitter, embeddings, ChromaDB
│   └── granite_llm.py        # IBM Granite LLM wrapper + system prompt
│
└── knowledge_base/           # Add your .txt / .pdf documents here
    ├── admission_policy.txt
    ├── fee_structure.txt
    ├── important_dates.txt
    ├── courses_programs.txt
    └── faqs.txt
```

---

## Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip
- An IBM Cloud account (free Lite tier)

### 1. Clone / Navigate to the project
```bash
cd college_admission_agent
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## IBM Cloud Configuration

### Step 1 — Create a free IBM Cloud account
Go to: https://cloud.ibm.com/registration

### Step 2 — Create a Watson Machine Learning service (Lite plan)
1. In IBM Cloud catalog, search for **"Watson Machine Learning"**
2. Select the **Lite (free)** plan → click **Create**

### Step 3 — Create a watsonx.ai project
1. Go to https://dataplatform.cloud.ibm.com
2. Click **New project** → **Create an empty project**
3. Copy the **Project ID** from the project Settings tab

### Step 4 — Get your IBM Cloud API Key
1. Go to https://cloud.ibm.com/iam/apikeys
2. Click **Create an IBM Cloud API key**
3. Copy and save the key securely

### Step 5 — Configure credentials
Edit `config.env` (or rename it to `.env`):

```env
WATSONX_API_KEY=your_actual_api_key_here
WATSONX_PROJECT_ID=your_actual_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
GRANITE_MODEL_ID=ibm/granite-3-8b-instruct
```

> ⚠️ **Never commit your API key to version control.**

---

## Running the Application

### Streamlit Web UI (recommended)
```bash
cd college_admission_agent
streamlit run app.py
```
Open http://localhost:8501 in your browser.

### Command-line Interactive Mode
```bash
cd college_admission_agent
python src/agent.py
```

### Rebuild Knowledge Base
```bash
cd college_admission_agent
python src/rag_engine.py
```

### Test IBM Granite Connection
```bash
cd college_admission_agent
python src/granite_llm.py
```

---

## Knowledge Base

The `knowledge_base/` directory contains the documents the agent retrieves from:

| File | Contents |
|------|----------|
| `admission_policy.txt` | Eligibility criteria, reservation policy, documents checklist, appeals |
| `fee_structure.txt` | Tuition fees for all programs, scholarships, payment schedule |
| `important_dates.txt` | Application deadlines, exam dates, semester calendar |
| `courses_programs.txt` | All UG/PG programs, seats, specializations, placement stats |
| `faqs.txt` | 30 frequently asked questions with detailed answers |

**Adding custom documents:** Drop any `.txt` or `.pdf` file into `knowledge_base/` and click **Rebuild Knowledge Base** in the sidebar (or run `python src/rag_engine.py`).

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `WATSONX_API_KEY` | IBM Cloud API key | *(required)* |
| `WATSONX_PROJECT_ID` | watsonx.ai Project ID | *(required)* |
| `WATSONX_URL` | watsonx.ai region URL | `https://us-south.ml.cloud.ibm.com` |
| `GRANITE_MODEL_ID` | IBM Granite model ID | `ibm/granite-3-8b-instruct` |
| `GRANITE_MAX_NEW_TOKENS` | Max tokens in response | `1024` |
| `GRANITE_TEMPERATURE` | LLM temperature | `0.1` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `all-MiniLM-L6-v2` |
| `CHROMA_PERSIST_DIR` | ChromaDB storage path | `./chroma_db` |
| `RAG_TOP_K` | Chunks retrieved per query | `5` |
| `CHUNK_SIZE` | Document chunk size (chars) | `500` |
| `CHUNK_OVERLAP` | Chunk overlap (chars) | `50` |

---

## Sample Questions

Try asking the agent:
- *"What is the eligibility criteria for B.Tech CSE?"*
- *"What is the total fee for MBA including hostel?"*
- *"When is the last date to apply for undergraduate programs?"*
- *"Is work experience required for MBA?"*
- *"What scholarships are available for SC/ST students?"*
- *"How do I apply for lateral entry admission?"*
- *"What GATE score is required for M.Tech?"*

---

## Contact

**Springfield Institute of Technology — Admissions Office**
📧 admissions@sit.edu.in | 📞 +91-98765-43210 | Mon–Fri 9 AM–5 PM IST

---

*Built for AICTE–IBM Internship · Problem Statement #4 · IBM Granite + watsonx.ai*
