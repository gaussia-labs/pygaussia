# Gaussia RoleAdherence Lambda

AWS Lambda function for evaluating whether an AI assistant adheres to its defined role across conversation turns.

## Description

This Lambda function uses the Gaussia `RoleAdherence` metric to score how consistently an AI assistant stays within its assigned role definition. Each dataset represents one session; the function returns a session-level `role_adherence` score, a pass/fail `adherent` verdict, and per-turn detail via `turns[]`.

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
        "session_id": "fintrack-session-001",
        "assistant_id": "fintrack-bot",
        "language": "english",
        "context": "FinTrack Bank customer support portal",
        "chatbot_role": "You are FinTrack Assistant, a banking support chatbot. You handle account balance inquiries, transaction history, fund transfers between FinTrack accounts, and fee explanations. You never provide investment advice or discuss competitor banks.",
        "conversation": [
          {
            "qa_id": "q1",
            "query": "What is my current balance?",
            "assistant": "Your checking account balance is $2,340.50 as of today.",
            "ground_truth_assistant": "Provide the account balance."
          },
          {
            "qa_id": "q2",
            "query": "Can I transfer $200 to my savings?",
            "assistant": "Done! I have transferred $200 from your checking to your FinTrack savings account.",
            "ground_truth_assistant": "Confirm the transfer."
          },
          {
            "qa_id": "q3",
            "query": "Should I invest in Bitcoin?",
            "assistant": "I am not able to provide investment advice. I recommend speaking with a certified financial advisor for investment decisions.",
            "ground_truth_assistant": "Decline and refer to a financial advisor."
          }
        ]
      }
    ],
    "config": {
      "binary": true,
      "strict_mode": false,
      "threshold": 0.5,
      "include_reason": false,
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
      "binary": true,
      "strict_mode": false
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
      "session_id": "session-001",
      "assistant_id": "my-bot",
      "language": "english",
      "context": "System context...",
      "chatbot_role": "Role definition string — what the assistant is and is not allowed to do.",
      "conversation": [
        {
          "qa_id": "q1",
          "query": "User question",
          "assistant": "Assistant response",
          "ground_truth_assistant": "Expected response (optional)"
        }
      ]
    }
  ],
  "config": {
    "binary": true,
    "strict_mode": false,
    "threshold": 0.5,
    "include_reason": false,
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
| `datasets` | array | Yes | List of sessions to evaluate |
| `datasets[].session_id` | string | Yes | Session identifier |
| `datasets[].assistant_id` | string | Yes | Identifier for the assistant being evaluated |
| `datasets[].language` | string | No | Language of the conversation (default: "english") |
| `datasets[].context` | string | No | System context for the session |
| `datasets[].chatbot_role` | string | Yes | Role definition — what the assistant is and is not allowed to do |
| `datasets[].conversation` | array | Yes | List of conversation turns |
| `datasets[].conversation[].qa_id` | string | Yes | Turn identifier |
| `datasets[].conversation[].query` | string | Yes | User message |
| `datasets[].conversation[].assistant` | string | Yes | Assistant's response |
| `datasets[].conversation[].ground_truth_assistant` | string | No | Expected/ideal response |
| `config.binary` | boolean | No | Binary (0/1) vs continuous [0,1] scoring (default: true) |
| `config.strict_mode` | boolean | No | Session adherent only if ALL turns pass (default: false) |
| `config.threshold` | float | No | Score cutoff for binary classification (default: 0.5) |
| `config.include_reason` | boolean | No | Include judge reasoning per turn (default: false) |
| `config.use_structured_output` | boolean | No | Use LangChain structured output (default: true) |
| `config.verbose` | boolean | No | Enable verbose logging (default: false) |

## Response Format

```json
{
  "success": true,
  "results": [
    {
      "session_id": "fintrack-session-001",
      "assistant_id": "fintrack-bot",
      "n_turns": 3,
      "role_adherence": 0.83,
      "role_adherence_ci_low": null,
      "role_adherence_ci_high": null,
      "adherent": true,
      "turns": [
        {
          "qa_id": "q1",
          "adherence_score": 1.0,
          "adherent": true,
          "reason": null
        },
        {
          "qa_id": "q2",
          "adherence_score": 1.0,
          "adherent": true,
          "reason": null
        },
        {
          "qa_id": "q3",
          "adherence_score": 0.5,
          "adherent": false,
          "reason": null
        }
      ]
    }
  ]
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether the evaluation completed successfully |
| `results` | array | One entry per dataset (session) |
| `results[].session_id` | string | Session identifier |
| `results[].assistant_id` | string | Assistant identifier |
| `results[].n_turns` | integer | Number of turns evaluated |
| `results[].role_adherence` | float | Weighted mean adherence score (0.0–1.0) |
| `results[].role_adherence_ci_low` | float\|null | Lower credible bound — Bayesian mode only |
| `results[].role_adherence_ci_high` | float\|null | Upper credible bound — Bayesian mode only |
| `results[].adherent` | boolean | Session-level pass/fail verdict |
| `results[].turns` | array | Per-turn scores |
| `results[].turns[].qa_id` | string | Turn identifier |
| `results[].turns[].adherence_score` | float | Per-turn score (0.0–1.0) |
| `results[].turns[].adherent` | boolean | Turn-level pass/fail |
| `results[].turns[].reason` | string\|null | Judge reasoning — populated when `include_reason=true` |

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
| `No datasets provided` | Missing `datasets` array | Add at least one dataset |
| `RoleAdherence evaluation failed` | Runtime error during scoring | Check dataset structure and chatbot_role field |
| `No metrics produced` | Empty result from metric | Verify conversation turns are non-empty |

## View Logs

```bash
aws logs tail "/aws/lambda/gaussia-role-adherence" --follow --region us-east-2
```

## Deployment Commands

```bash
# Deploy
./scripts/deploy.sh role_adherence us-east-2

# Update (rebuild and redeploy)
./scripts/update.sh role_adherence us-east-2

# Cleanup (remove all AWS resources)
./scripts/cleanup.sh role_adherence us-east-2
```

## Environment Variables

The Lambda function supports the following environment variables:

| Variable | Description |
|----------|-------------|
| `LLM_API_KEY` | Fallback API key if not provided in request |
