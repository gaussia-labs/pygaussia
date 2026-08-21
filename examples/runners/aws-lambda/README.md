# Gaussia Runners Lambda

AWS Lambda function for executing test datasets against AI systems.

## Description

This Lambda function executes test datasets against AI systems and collects responses for evaluation. It supports two modes:

1. **Alquimia Mode**: Execute tests against Alquimia AI agents
2. **LLM Mode**: Execute tests directly against any LangChain-compatible LLM

## Invoke URL

```
https://0js1qmtm6d.execute-api.us-east-2.amazonaws.com/run
```

## Supported LLM Providers (LLM Mode)

| Provider | class_path |
|----------|-----------|
| Groq | `langchain_groq.chat_models.ChatGroq` |
| OpenAI | `langchain_openai.chat_models.ChatOpenAI` |
| Google Gemini | `langchain_google_genai.chat_models.ChatGoogleGenerativeAI` |
| Ollama | `langchain_ollama.chat_models.ChatOllama` |

## Test Examples

### LLM Mode (Direct LLM Testing)

```bash
curl -s -X POST "https://0js1qmtm6d.execute-api.us-east-2.amazonaws.com/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_groq.chat_models.ChatGroq",
      "params": {
        "model": "qwen/qwen3-32b",
        "api_key": "your-groq-api-key"
      }
    },
    "datasets": [
      {
        "session_id": "test-session-1",
        "assistant_id": "groq-assistant",
        "language": "english",
        "context": "",
        "conversation": [
          {
            "qa_id": "q1",
            "query": "What is the capital of France?",
            "assistant": "",
            "ground_truth_assistant": "Paris"
          }
        ]
      }
    ]
  }'
```

### Using OpenAI

```bash
curl -s -X POST "https://0js1qmtm6d.execute-api.us-east-2.amazonaws.com/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_openai.chat_models.ChatOpenAI",
      "params": {
        "model": "gpt-4o-mini",
        "api_key": "your-openai-api-key"
      }
    },
    "datasets": [...]
  }'
```

### Alquimia Mode

```bash
curl -s -X POST "https://0js1qmtm6d.execute-api.us-east-2.amazonaws.com/run" \
  -H "Content-Type: application/json" \
  -d '{
    "datasets": [
      {
        "session_id": "test-session-1",
        "assistant_id": "target-assistant",
        "language": "english",
        "context": "",
        "conversation": [
          {
            "qa_id": "q1",
            "query": "What is the capital of France?",
            "assistant": "",
            "ground_truth_assistant": "Paris is the capital of France."
          }
        ]
      }
    ],
    "config": {
      "base_url": "https://api.alquimia.ai",
      "api_key": "your-alquimia-api-key",
      "agent_id": "your-agent-id",
      "channel_id": "your-channel-id"
    }
  }'
```

## Request Format

### LLM Mode

```json
{
  "connector": {
    "class_path": "langchain_groq.chat_models.ChatGroq",
    "params": {
      "model": "qwen/qwen3-32b",
      "api_key": "your-api-key"
    }
  },
  "datasets": [
    {
      "session_id": "unique-session-id",
      "assistant_id": "assistant-id",
      "language": "english",
      "context": "Optional context for all queries",
      "conversation": [
        {
          "qa_id": "unique-qa-id",
          "query": "User question to send to the LLM",
          "assistant": "",
          "ground_truth_assistant": "Expected answer (optional)"
        }
      ]
    }
  ]
}
```

### Alquimia Mode

```json
{
  "datasets": [...],
  "config": {
    "base_url": "https://api.alquimia.ai",
    "api_key": "your-alquimia-api-key",
    "agent_id": "your-agent-id",
    "channel_id": "your-channel-id"
  }
}
```

## Response Format

```json
{
  "success": true,
  "datasets": [
    {
      "session_id": "test-session-1",
      "assistant_id": "assistant-id",
      "language": "english",
      "context": "",
      "conversation": [
        {
          "qa_id": "q1",
          "query": "What is the capital of France?",
          "assistant": "The capital of France is Paris.",
          "ground_truth_assistant": "Paris"
        }
      ]
    }
  ],
  "summaries": [
    {
      "session_id": "test-session-1",
      "total_batches": 1,
      "successes": 1,
      "failures": 0,
      "total_execution_time_ms": 1234.5,
      "avg_batch_time_ms": 1234.5
    }
  ],
  "total_datasets": 1
}
```

## View Logs

```bash
aws logs tail "/aws/lambda/gaussia-runners" --follow --region us-east-2
```

## Deployment Commands

```bash
# Deploy
./scripts/deploy.sh runners us-east-2

# Update (rebuild and redeploy)
./scripts/update.sh runners us-east-2

# Cleanup (remove all AWS resources)
./scripts/cleanup.sh runners us-east-2
```
