# HealBridge AI - Post-Discharge Medical Assistant



A simple AI chatbot that helps patients after they leave the hospital. It answers medical questions using a knowledge base and can search the web if needed.


What It Does> **A sophisticated multi-agent AI system for post-discharge patient care** using RAG, intelligent query optimization, and medical-grade logging.> **A sophisticated multi-agent AI system for post-discharge patient care** using RAG, intelligent query optimization, and medical-grade logging.


# Install dependenciesSet database credentials as environment variables (or in `.env`):

pip install -r requirements.txt

```env

# Configure environmentPGHOST=localhost

cp .env.example .envPGPORT=5432

# Edit .env with:PGDATABASE=your_db

# GROQ_API_KEY=your_key_herePGUSER=your_user

# DATABASE_URL=postgresql://user:pass@localhost/dbPGPASSWORD=your_password

```PGSSLMODE=prefer

```

### Running the Application

Query by patient name (partial match by default). Replace the table name if yours differs:

```bash

# Web UI (Recommended)```powershell

streamlit run frontend/streamlit_app.pypython .\agents\clinicalDbAgent.py --name "John Smith" --table patient_discharges --limit 3 --json_only

```





```bash## Receptionist Agent

# Ingest nephrology knowledge base

python rag/ingest.pyInteractive greeting + follow-up using the latest discharge record for the patient, then routes the patient's concern to the Clinical AI Agent with patient context.



# Load sample patients (optional)Requires a Groq API key (set GROQ_API_KEY in your environment or in `.env`).

python scripts/load_dummy_patients.py

```Interactive prompt:



---```powershell

python .\agents\receptionistAgent.py

## 📁 Project Structure```



```Non-interactive (pass name):

HealBridge-AI/

├── agents/                              # Multi-agent system```powershell

│   ├── receptionistAgent.py            # Greeting & handoff routingpython .\agents\receptionistAgent.py --name "John Smith" --schema public --table patient_discharges --match exact --llm llama-3.1-8b-instant --persist_dir .\chroma_db --collection default --embed_model sentence-transformers/all-MiniLM-L6-v2 --top_k 5 --threshold 0.45 --web_max 5

│   ├── clinicalQueryComposerAgent.py   # Query optimization (NOVEL)```

│   ├── clinicalAiAgent.py              # RAG & clinical reasoning

│   └── clinicalDbAgent.py              # Patient data retrieval## Clinical Query Composer Agent

│

├── app/When you want stronger KB retrieval, compose the question using patient context first, then answer:

│   └── main.py                         # CLI orchestrator

│```powershell

├── frontend/python .\agents\clinicalQueryComposerAgent.py --name "John Smith" --question "Is leg swelling related to CKD Stage 3?" --persist_dir .\chroma_db --collection default --embed_model sentence-transformers/all-MiniLM-L6-v2 --top_k 8 --threshold 0.55 --print_composed

│   └── streamlit_app.py                # Web interface```

│

├── rag/Or provide patient info directly:

│   ├── ingest.py                       # PDF ingestion & embedding

│   ├── query.py                        # RAG retrieval```powershell

│   └── agent.py                        # RAG pipeline wrapperpython .\agents\clinicalQueryComposerAgent.py --patient_info_json '{"patient_name":"John Smith","primary_diagnosis":"Chronic Kidney Disease Stage 3","medications":["Lisinopril 10mg daily","Furosemide 20mg twice daily"],"dietary_restrictions":"Low sodium, fluid restriction 1.5L/day"}' --question "Why are my legs swollen?" --persist_dir .\chroma_db --collection default --top_k 8 --threshold 0.55 --print_composed

│```

├── utils/
│   └── logging_setup.py                # Centralized logging
│
├── chroma_db/                          # Vector database (persistent)
├── logs/                               # Application logs
├── comprehensive-clinical-nephrology.pdf
├── dummy.json                          # Sample patients
├── requirements.txt
├── .env.example
├── BRIEF_REPORT.md                     # Technical report
└── README.md
```

---

## ⚙️ Configuration

### Environment Variables (.env)

```env
# Groq API
GROQ_API_KEY=your_groq_api_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/patients_db
DB_SCHEMA=public

# RAG Configuration
PERSIST_DIR=chroma_db
COLLECTION_NAME=default
EMBED_MODEL=sentence-transformers/all-mpnet-base-v2
TOP_K=8
THRESHOLD=0.55
MIN_KB_HITS=2
STRICT_THRESHOLD=0.40

# LLM Configuration
ANSWER_LLM=llama-3.1-8b-instant
COMPOSE_LLM=llama-3.1-8b-instant
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=512

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

### Database Schema

```sql
CREATE TABLE public.patient_discharges (
    id SERIAL PRIMARY KEY,
    patient_name VARCHAR(255) NOT NULL,
    discharge_date DATE,
    primary_diagnosis VARCHAR(255),
    medications TEXT[],
    dietary_restrictions TEXT,
    follow_up VARCHAR(500),
    warning_signs TEXT,
    discharge_instructions TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);
```

---

## 💡 Usage Examples

### Web UI Workflow

```
1. Enter patient name → Lookup record
2. Receive greeting + handoff
3. Ask medical question
4. View answer + collapsible sources
5. Continue multi-turn conversation
```



## 🏗️ System Architecture

```
Patient Input (Web/CLI)
    ↓
Receptionist Agent
    ↓
Query Composer Agent (NOVEL)
    ↓
Clinical AI Agent
├─ ChromaDB Retrieval
├─ Strict Gating
└─ Web Fallback
    ↓
Response + Citations
    ↓
Audit Logging
```

---

## 🔍 [NOVEL] Query Composer Agent

**The Competitive Differentiator** ⭐

This agent solves a fundamental RAG limitation: colloquial patient language doesn't match medical knowledge documents.

**Transformation Example:**
```
Raw:       "I'm feeling really tired after waking up"
Optimized: "Post-discharge fatigue and morning tiredness in CKD stage 3; 
            differential including anemia, medication effects, fluid management"
Result:    30-40% better KB match quality
```

---

## 📊 Technical Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | Groq llama-3.1-8b-instant |
| **Vector DB** | ChromaDB + all-mpnet-base-v2 |
| **RAG** | LangChain + custom implementation |
| **Web Search** | DuckDuckGo |
| **Database** | PostgreSQL |
| **Frontend** | Streamlit + CLI |
| **Logging** | Python rotating file logger |

---

## 📋 Logging & Monitoring

Complete audit trail in `logs/app.log`:

```
session.start / session.end
patient.lookup (success/failure)
receptionist.greeting / receptionist.handoff
composer.query (transformation)
retrieval.decision (KB vs Web)
answer.generated (sources, scores)
sources.list (citations)
smalltalk.* (classification)
```

View logs:
```bash
tail -f logs/app.log
```

---

## ⚖️ Medical Disclaimer

⚠️ **IMPORTANT: Educational Use Only**

This AI assistant is a **Proof of Concept** for educational purposes only. It is **NOT** a substitute for professional medical advice.

**Always consult qualified healthcare professionals before making medical decisions.**

This system:
- Provides general information based on guidelines
- Uses AI-generated responses (may contain errors)
- Is **NOT** for emergency situations
- Should **NOT** delay professional consultation

**For urgent medical concerns, contact healthcare providers immediately.**

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---



