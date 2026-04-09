# Gaussia BestOf Lambda

AWS Lambda function for tournament-style comparison of multiple AI assistants to find the best one.

## Description

This Lambda function uses the Gaussia `BestOf` metric to run a tournament-style elimination between multiple AI assistants. It compares their responses to the same questions and determines a winner based on specified evaluation criteria. The tournament runs in rounds, with winners advancing until a final winner is determined.

## Invoke URL

```
<DEPLOY_TO_GET_URL>
```

## Supported LLM Providers

| Provider | class_path |
|----------|-----------|
| Groq | `langchain_groq.chat_models.ChatGroq` |
| OpenAI | `langchain_openai.chat_models.ChatOpenAI` |
| Google Gemini | `langchain_google_genai.chat_models.ChatGoogleGenerativeAI` |
| Ollama | `langchain_ollama.chat_models.ChatOllama` |

## Test Example

### Using Groq

```bash
curl -s -X POST "<INVOKE_URL>/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_groq.chat_models.ChatGroq",
      "params": {
        "model": "qwen/qwen3-32b",
        "api_key": "your-groq-api-key",
        "temperature": 0.0
      }
    },
    "datasets": [
      {
        "session_id": "comparison_session",
        "assistant_id": "assistant_a",
        "language": "english",
        "context": "You are a helpful AI assistant.",
        "conversation": [
          {
            "qa_id": "q1",
            "query": "What are the benefits of renewable energy?",
            "assistant": "Renewable energy offers numerous benefits including reduced emissions and long-term cost savings.",
            "ground_truth_assistant": "Renewable energy provides clean power and reduces carbon emissions."
          }
        ]
      },
      {
        "session_id": "comparison_session",
        "assistant_id": "assistant_b",
        "language": "english",
        "context": "You are a helpful AI assistant.",
        "conversation": [
          {
            "qa_id": "q1",
            "query": "What are the benefits of renewable energy?",
            "assistant": "Clean energy good. Sun power help planet.",
            "ground_truth_assistant": "Renewable energy provides clean power and reduces carbon emissions."
          }
        ]
      }
    ],
    "config": {
      "criteria": "Overall response quality, clarity, and accuracy",
      "use_structured_output": true
    }
  }'
```

### Using OpenAI

```bash
curl -s -X POST "<INVOKE_URL>/run" \
  -H "Content-Type: application/json" \
  -d '{
    "connector": {
      "class_path": "langchain_openai.chat_models.ChatOpenAI",
      "params": {
        "model": "gpt-4o-mini",
        "api_key": "your-openai-api-key"
      }
    },
    "datasets": [...],
    "config": {
      "criteria": "Overall response quality"
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
      "temperature": 0.0
    }
  },
  "datasets": [
    {
      "session_id": "comparison_session",
      "assistant_id": "assistant_a",
      "language": "english",
      "context": "System context...",
      "conversation": [
        {
          "qa_id": "q1",
          "query": "User question",
          "assistant": "Assistant A response",
          "ground_truth_assistant": "Expected response (optional)"
        }
      ]
    },
    {
      "session_id": "comparison_session",
      "assistant_id": "assistant_b",
      "language": "english",
      "context": "System context...",
      "conversation": [
        {
          "qa_id": "q1",
          "query": "User question",
          "assistant": "Assistant B response",
          "ground_truth_assistant": "Expected response (optional)"
        }
      ]
    }
  ],
  "config": {
    "criteria": "Overall response quality",
    "use_structured_output": true,
    "verbose": false
  }
}
```

### Connector Configuration

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `connector.class_path` | string | Yes | Full class path of the LangChain chat model |
| `connector.params` | object | Yes | Parameters passed to the chat model constructor |
| `connector.params.model` | string | Yes | Model name/identifier |
| `connector.params.api_key` | string | Yes | API key for the LLM provider |
| `connector.params.temperature` | float | No | Sampling temperature (recommend 0.0 for consistency) |

### Module-Specific Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `datasets` | array | Yes | List of datasets, each representing one assistant's responses |
| `datasets[].session_id` | string | Yes | Session identifier (same across all assistants being compared) |
| `datasets[].assistant_id` | string | Yes | Unique identifier for each assistant |
| `datasets[].language` | string | No | Language of the conversation (default: "english") |
| `datasets[].context` | string | No | System context for the assistant |
| `datasets[].conversation` | array | Yes | List of Q&A interactions |
| `datasets[].conversation[].qa_id` | string | Yes | Question identifier (same across assistants for comparison) |
| `datasets[].conversation[].query` | string | Yes | User question |
| `datasets[].conversation[].assistant` | string | Yes | Assistant's response |
| `datasets[].conversation[].ground_truth_assistant` | string | No | Expected/ideal response |
| `config.criteria` | string | No | Evaluation criteria description (default: "Overall response quality") |
| `config.use_structured_output` | boolean | No | Use LangChain structured output (default: true) |
| `config.verbose` | boolean | No | Enable verbose logging (default: false) |

## Response Format

```json
{
  "success": true,
  "winner": "assistant_a",
  "contestants": ["assistant_a", "assistant_b", "assistant_c"],
  "total_rounds": 2,
  "contests": [
    {
      "round": 1,
      "left": "assistant_a",
      "right": "assistant_b",
      "winner": "assistant_a",
      "confidence": 0.85,
      "verdict": "Assistant A provides more comprehensive and accurate responses",
      "reasoning": "Detailed reasoning for the decision..."
    },
    {
      "round": 2,
      "left": "assistant_a",
      "right": "assistant_c",
      "winner": "assistant_a",
      "confidence": 0.92,
      "verdict": "Assistant A demonstrates superior clarity",
      "reasoning": "Detailed reasoning..."
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the evaluation completed successfully |
| `winner` | string | The assistant_id of the tournament winner |
| `contestants` | array | List of all assistant_ids that participated |
| `total_rounds` | integer | Number of rounds in the tournament |
| `contests` | array | Details of each head-to-head contest |
| `contests[].round` | integer | Round number |
| `contests[].left` | string | First contestant's assistant_id |
| `contests[].right` | string | Second contestant's assistant_id |
| `contests[].winner` | string | Winner's assistant_id (or "tie") |
| `contests[].confidence` | float | Judge's confidence in the decision (0-1) |
| `contests[].verdict` | string | Brief summary of the decision |
| `contests[].reasoning` | string | Detailed reasoning for the decision |

## Error Responses

```json
{
  "success": false,
  "error": "Error message describing what went wrong"
}
```

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `No connector configuration provided` | Missing `connector` field | Add connector with class_path and params |
| `connector.class_path is required` | Missing class_path in connector | Specify the LLM class path |
| `Failed to create LLM connector` | Invalid class_path or params | Check class_path spelling and required params |
| `No datasets provided` | Missing `datasets` array | Add datasets with assistant responses |
| `BestOf requires at least 2 datasets with different assistant_ids` | Only one dataset | Add datasets from multiple assistants |
| `BestOf requires datasets from at least 2 different assistants` | Same assistant_id | Use unique assistant_ids |

## View Logs

```bash
aws logs tail "/aws/lambda/gaussia-bestof" --follow --region us-east-2
```

## Deployment Commands

```bash
# Deploy
./scripts/deploy.sh bestof us-east-2

# Update (rebuild and redeploy)
./scripts/update.sh bestof us-east-2

# Cleanup (remove all AWS resources)
./scripts/cleanup.sh bestof us-east-2
```

## Environment Variables

The Lambda function supports the following environment variables:

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | Fallback API key if not provided in request |
