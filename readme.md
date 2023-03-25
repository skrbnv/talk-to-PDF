# Talk-to-PDF via OpenAI GPT API
(need OpenAI API key)
Simple FastAPI/uvcorn microservice to process PDF and extract information using ChatGPT

## Getting Started

![alt text](/extras/chat.png)


First, copy config_sample.yaml to config.yaml and update your API key.
Inside you'll find settings, including:
- model names for Completion and Embeddings
- target chunk size, larger chunks may provide better context, but increase cost
- top_K is a value, use it to define how many chunks you need to extract to provide context for the question-answer pair. More chunks = more expensive each request
- request_delay_embeddings, in seconds. Accounts with signup bonus are limited to 60 requests per minute, thus 1 second delay between requests to meet rate limit

# Installation

Create virtual environment with python 3.10^ using virtualenv or conda
You can look up some additional guide like this: https://help.dreamhost.com/hc/en-us/articles/115000695551-Installing-and-using-virtualenv-with-Python-3

Install dependencies from requirements.txt
```
pip install -r requirements.txt
```
Then run service with 
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Access chat page at the root endpoint via browser, for example if you are using settings abuve, open URL http://0.0.0.0:8000/
You can also get list of endpoints to access them directly at http://0.0.0.0:8000/docs

