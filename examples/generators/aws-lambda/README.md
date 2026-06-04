# Gaussia Generators Lambda

AWS Lambda function for generating synthetic test datasets from context documents using LLMs.

## Description

This Lambda function uses the Gaussia `BaseGenerator` to create synthetic Q&A datasets from markdown content. It supports multiple LLM providers through a dynamic connector pattern.

## Invoke URL

```
https://0prcm0jwwg.execute-api.us-east-2.amazonaws.com/run
```

## Supported LLM Providers

| Provider | class_path |
|----------|-----------|
| Groq | `langchain_groq.chat_models.ChatGroq` |
| OpenAI | `langchain_openai.chat_models.ChatOpenAI` |
| Google Gemini | `langchain_google_genai.chat_models.ChatGoogleGenerativeAI` |
| Ollama | `langchain_ollama.chat_models.ChatOllama` |

## Test Example

```bash
curl -s -X POST "https://0prcm0jwwg.execute-api.us-east-2.amazonaws.com/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_groq.chat_models.ChatGroq",
      "params": {
        "model": "qwen/qwen3-32b",
        "api_key": "your-groq-api-key",
        "temperature": 0.7
      }
    },
    "context": "# Product Documentation\n\nOur product helps users manage their tasks efficiently.\n\n## Features\n\n- Task creation and management\n- Due date reminders\n- Team collaboration",
    "config": {
      "assistant_id": "docs-assistant",
      "num_queries": 3,
      "language": "english",
      "conversation_mode": false
    }
  }'
```

### Using OpenAI

```bash
curl -s -X POST "https://0prcm0jwwg.execute-api.us-east-2.amazonaws.com/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_openai.chat_models.ChatOpenAI",
      "params": {
        "model": "gpt-4o-mini",
        "api_key": "your-openai-api-key",
        "temperature": 0.7
      }
    },
    "context": "Your markdown content...",
    "config": {
      "assistant_id": "my-assistant",
      "num_queries": 3
    }
  }'
```

## Request Format

```json
{
  "connector": {
    "class_path": "langchain_groq.chat_models.ChatGroq",
    "params": {
      "model": "qwen/qwen3-32b",
      "api_key": "your-api-key",
      "temperature": 0.7
    }
  },
  "context": "# Your Markdown Content\n\nContent to generate questions from...",
  "config": {
    "assistant_id": "my-assistant",
    "num_queries": 3,
    "language": "english",
    "conversation_mode": false,
    "max_chunk_size": 2000,
    "min_chunk_size": 200,
    "seed_examples": ["Example question 1?", "Example question 2?"]
  }
}
```

## Response Format

```json
{
  "success": true,
  "datasets": [
    {
      "session_id": "uuid-generated",
      "assistant_id": "my-assistant",
      "language": "english",
      "context": "Combined chunk content...",
      "conversation": [
        {
          "qa_id": "chunk-1_q1",
          "query": "Generated question about the content?",
          "assistant": "",
          "ground_truth_assistant": ""
        }
      ]
    }
  ],
  "total_datasets": 1,
  "total_batches": 3
}
```

## View Logs

```bash
aws logs tail "/aws/lambda/gaussia-generators" --follow --region us-east-2
```

## Deployment Commands

```bash
# Deploy
./scripts/deploy.sh generators us-east-2

# Update (rebuild and redeploy)
./scripts/update.sh generators us-east-2

# Cleanup (remove all AWS resources)
./scripts/cleanup.sh generators us-east-2
```
