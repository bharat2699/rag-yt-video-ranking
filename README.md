# YouTube Video Ranking System

This project implements a semantic search and ranking system for YouTube videos using transcripts and embeddings.

## Features

- YouTube video search using YouTube Data API v3
- Transcript extraction from videos
- Semantic search using embeddings
- Video ranking based on content relevance
- Optional summarization using LLM

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Create a `.env` file with your API keys:
```
YOUTUBE_API_KEY=your_youtube_api_key
OPENAI_API_KEY=your_openai_api_key
```

3. Run the script:
```bash
python main.py
```

## Usage

The script can be used in two ways:

1. Command line:
```bash
python main.py --query "your search query" --max_results 10
```

2. As a module:
```python
from youtube_ranker import YouTubeRanker

ranker = YouTubeRanker()
results = ranker.search_and_rank("your search query", max_results=10)
```

## Requirements

- Python 3.8+
- YouTube Data API v3 key
- OpenAI API key (for embeddings and optional summarization) 