# Ralph Agent Instructions - Q for Mortals RAG Agent

You are an autonomous coding agent implementing a RAG system for Q for Mortals documentation.

## Your Workflow

Execute these steps in order:

### 1. Read Project State
```bash
# Read the PRD
cat prd.json

# Read progress log for context
cat progress.txt

# Check git status
git status
```

### 2. Select Next Story
- Find the highest-priority story where `passes: false`
- Priority is determined by the `priority` field (lower = higher priority)
- Work on exactly ONE story per iteration

### 3. Implement the Story
- Write clean, simple Python code
- Follow existing patterns in the codebase
- Keep changes focused on the single story
- Use type hints and docstrings

### 4. Run Quality Checks
Before marking complete, verify:
```bash
# Check Python syntax
python -m py_compile src/**/*.py

# Run type checks if mypy is available
mypy src/ --ignore-missing-imports || true

# Run tests if they exist
pytest tests/ -v || true
```

### 5. Update Progress Log
Append (never overwrite) to `progress.txt`:
```markdown
### Story: {story_id} - {title}
**Completed:** {timestamp}

**What was implemented:**
- Bullet points of changes

**Files modified:**
- path/to/file.py

**Learnings for future iterations:**
- Any patterns discovered
- Gotchas encountered
- Useful context for later stories
```

### 6. Mark Story Complete
Update `prd.json` to set `passes: true` for the completed story.

### 7. Commit Changes
```bash
git add -A
git commit -m "Complete {story_id}: {title}"
```

### 8. Signal Completion
If ALL stories have `passes: true`, output:
```
<promise>COMPLETE</promise>
```

---

## Project-Specific Context

### KDB.AI Connection
- Endpoint: `http://192.168.1.68:8082`
- Use environment variable: `KDBAI_ENDPOINT`

### KDB.AI Client Patterns
```python
import kdbai_client as kdbai

# Connect
session = kdbai.Session(endpoint=os.environ.get("KDBAI_ENDPOINT", "http://192.168.1.68:8082"))
db = session.database("default")

# Create table with schema
schema = [
    {"name": "chunk_id", "type": "str"},
    {"name": "text", "type": "str"},
    {"name": "chapter", "type": "str"},
    {"name": "heading", "type": "str"},
    {"name": "url", "type": "str"},
    {"name": "embeddings", "type": "float32s"}
]
indexes = [
    {"name": "flat_index", "type": "flat", "column": "embeddings",
     "params": {"dims": 384, "metric": "CS"}}
]
table = db.create_table("q4m_chunks", schema=schema, indexes=indexes)

# Insert with pandas DataFrame
import pandas as pd
df = pd.DataFrame(data)
table.insert(df)

# Search
results = table.search(vectors={"flat_index": [query_vector]}, n=5)

# Search with filter
results = table.search(
    vectors={"flat_index": [query_vector]},
    n=5,
    filter=[("=", "chapter", "Tables")]
)
```

### FastEmbed Patterns
```python
from fastembed import TextEmbedding

# Initialize model (384 dimensions)
model = TextEmbedding("BAAI/bge-small-en-v1.5")

# Generate embeddings (returns generator)
texts = ["Hello", "World"]
embeddings = list(model.embed(texts))  # List of numpy arrays
```

### SQLite Patterns
```python
import sqlite3
from pathlib import Path

def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            file_id TEXT PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            scraped_at TEXT,
            content_hash TEXT,
            status TEXT DEFAULT 'pending',
            error_message TEXT
        )
    """)
    conn.commit()
    return conn
```

### Web Scraping Patterns
```python
import requests
from bs4 import BeautifulSoup
import markdownify
import time

def fetch_page(url: str, delay: float = 1.0) -> str:
    time.sleep(delay)  # Be polite
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text

def extract_body(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    # Remove nav, header, footer, toc
    for tag in soup.find_all(['nav', 'header', 'footer', 'aside']):
        tag.decompose()
    main = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
    return str(main) if main else str(soup.body)

def to_markdown(html: str) -> str:
    return markdownify.markdownify(html, heading_style="ATX", code_language="q")
```

### Directory Structure
```
kdbaiClient/
├── src/
│   ├── __init__.py
│   ├── db.py              # SQLite operations
│   ├── scraper/
│   │   ├── __init__.py
│   │   ├── discover.py    # URL discovery
│   │   ├── fetcher.py     # HTTP fetching
│   │   ├── extractor.py   # Content extraction
│   │   ├── storage.py     # File saving
│   │   └── main.py        # Scraper orchestrator
│   ├── embedding/
│   │   ├── __init__.py
│   │   ├── chunker.py     # Text chunking
│   │   ├── embedder.py    # FastEmbed wrapper
│   │   └── main.py        # Embedding pipeline
│   ├── kdbai/
│   │   ├── __init__.py
│   │   └── client.py      # KDB.AI operations
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py        # FastAPI app
│   │   └── query.py       # Query logic
│   └── chat/
│       ├── __init__.py
│       └── main.py        # Gradio interface
├── config/
│   └── config.yaml
├── data/
│   ├── raw/               # Scraped markdown files
│   └── processed/         # Chunked data
├── tests/
├── prd.json
├── prd.txt                # Original PRD (reference)
├── progress.txt
├── prompt.md
├── ralph.sh
├── requirements.txt
└── KDBAI_REFERENCE.md
```

---

## Quality Guidelines

### Code Style
- Use Python 3.11+ features where appropriate
- Add type hints to function signatures
- Include docstrings for public functions
- Use `pathlib.Path` for file operations
- Use `logging` module (not print) for status messages

### Error Handling
- Catch specific exceptions, not bare `except:`
- Log errors with context
- Allow caller to decide recovery strategy

### Testing
- Write unit tests for core logic
- Use pytest fixtures for setup
- Mock external dependencies (HTTP, DB)

---

## Important Rules

1. **One story per iteration** - Do not try to complete multiple stories
2. **Small, focused changes** - Each commit should be reviewable
3. **Update progress.txt** - Document what you learned
4. **Check before marking done** - Run quality checks first
5. **Fresh context** - Don't assume previous iterations' state; always read files

---

## Q for Mortals Source URLs

- Main index: https://code.kx.com/q4m3/
- Also consider: https://code.kx.com/q/ (Q reference docs)

The site uses a consistent structure. Each chapter is at a predictable URL pattern.

---

## Signal Words

- Output `<promise>COMPLETE</promise>` when ALL stories are done
- This signals ralph.sh to exit the loop successfully
