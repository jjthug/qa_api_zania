
# QA API


## Overview

Using OpenAI LLM we generate answers to requested questions based on a document file.

## Setup Instructions


### 1. Create a virtual environment named 'myenv' (Optional)
```bash
python3  -m  venv  myenv
```

### 2. Activate the virtual environment (Optional)
```bash
source  myenv/bin/activate
```
  
### 3. Install the dependencies (tested with python 3.12.7)
```bash
pip install -r requirements.txt
```
  

### 4. Setup the environment variables

Create a .env file and store the OpenAI Api key like this
```bash
OPENAI_API_KEY=sk-proj-********

```


### 5. Run the server

```bash
uvicorn  main:app  --reload
```

## Testing

Example data of questions file and input document file are in data directory

Refer to the example_curl_script file to call the api
