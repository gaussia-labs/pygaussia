#!/usr/bin/env bash
#
# Deploy a Gaussia metric as an AWS Lambda container image with HTTP API
#
# Usage: ./deploy.sh <module-name> [region]
#
# Example: ./deploy.sh bestof us-east-1
#
# Prerequisites:
#   - AWS CLI v2 configured with valid credentials
#   - Docker running
#   - Run from examples/<module>/aws-lambda directory
#   - Gaussia repo at ../../.. (three levels up)
#
# Creates:
#   - ECR repository
#   - IAM role for Lambda
#   - Lambda function (container image)
#   - API Gateway HTTP API with POST /run route

set -euo pipefail

# --- Arguments ---
MODULE_NAME="${1:-}"
AWS_REGION="${2:-us-east-1}"

if [[ -z "$MODULE_NAME" ]]; then
    echo "Usage: $0 <module-name> [region]"
    echo "Example: $0 bestof us-east-1"
    echo ""
    echo "Module names: bestof, toxicity, bias, context, conversational, humanity"
    exit 1
fi

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMBDA_DIR="$(pwd)"
REPO_ROOT="$(cd "$LAMBDA_DIR/../../.." && pwd)"

# Verify we're in the right place
if [[ ! -f "$REPO_ROOT/pyproject.toml" ]]; then
    echo "Error: Cannot find gaussia repo root at $REPO_ROOT"
    echo "Run this script from examples/<module>/aws-lambda directory"
    exit 1
fi

# --- Derived names ---
PROJECT_NAME="gaussia-${MODULE_NAME}"
ECR_REPO="$PROJECT_NAME"
LAMBDA_NAME="$PROJECT_NAME"
LAMBDA_ROLE_NAME="${PROJECT_NAME}-lambda-role"
API_NAME="${PROJECT_NAME}-http-api"

echo "=== Deploying Gaussia Lambda: $PROJECT_NAME ==="
echo "Module: $MODULE_NAME"
echo "Region: $AWS_REGION"
echo "Repo root: $REPO_ROOT"
echo ""

# --- Get AWS Account ID ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: $ACCOUNT_ID"

IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"

# --- Step 1: Create ECR Repository ---
echo ""
echo "=== Step 1: Creating ECR repository ==="
if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" &>/dev/null; then
    echo "ECR repository '$ECR_REPO' already exists"
else
    aws ecr create-repository \
        --repository-name "$ECR_REPO" \
        --region "$AWS_REGION" \
        --output text
    echo "Created ECR repository: $ECR_REPO"
fi

# --- Step 2: Build Docker image from repo root ---
echo ""
echo "=== Step 2: Building Docker image ==="
echo "Building from: $REPO_ROOT"
echo "Using Dockerfile: $LAMBDA_DIR/Dockerfile"

# Build with the repo root as context, Dockerfile from lambda dir
docker build \
    -t "$PROJECT_NAME:latest" \
    -f "$LAMBDA_DIR/Dockerfile" \
    --build-arg MODULE_EXTRA="$MODULE_NAME" \
    "$REPO_ROOT"

# --- Step 3: Login to ECR and push ---
echo ""
echo "=== Step 3: Pushing image to ECR ==="
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker tag "$PROJECT_NAME:latest" "$IMAGE_URI"
docker push "$IMAGE_URI"
echo "Pushed: $IMAGE_URI"

# --- Step 4: Create IAM Role ---
echo ""
echo "=== Step 4: Creating IAM role ==="
TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "lambda.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}'

if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" &>/dev/null; then
    echo "IAM role '$LAMBDA_ROLE_NAME' already exists"
else
    aws iam create-role \
        --role-name "$LAMBDA_ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --output text

    aws iam attach-role-policy \
        --role-name "$LAMBDA_ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    echo "Created IAM role: $LAMBDA_ROLE_NAME"
    echo "Waiting for IAM propagation (10s)..."
    sleep 10
fi

ROLE_ARN=$(aws iam get-role --role-name "$LAMBDA_ROLE_NAME" --query "Role.Arn" --output text)

# --- Step 5: Create Lambda Function ---
echo ""
echo "=== Step 5: Creating Lambda function ==="
if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$AWS_REGION" &>/dev/null; then
    echo "Lambda function '$LAMBDA_NAME' already exists, updating..."
    aws lambda update-function-code \
        --function-name "$LAMBDA_NAME" \
        --image-uri "$IMAGE_URI" \
        --region "$AWS_REGION" \
        --output text
else
    aws lambda create-function \
        --function-name "$LAMBDA_NAME" \
        --package-type Image \
        --code ImageUri="$IMAGE_URI" \
        --role "$ROLE_ARN" \
        --timeout 300 \
        --memory-size 2048 \
        --region "$AWS_REGION" \
        --output text

    echo "Waiting for Lambda to become active..."
    aws lambda wait function-active --function-name "$LAMBDA_NAME" --region "$AWS_REGION"
fi

LAMBDA_ARN=$(aws lambda get-function \
    --function-name "$LAMBDA_NAME" \
    --query "Configuration.FunctionArn" \
    --output text \
    --region "$AWS_REGION")

echo "Lambda ARN: $LAMBDA_ARN"

# --- Step 6: Create API Gateway HTTP API ---
echo ""
echo "=== Step 6: Creating API Gateway HTTP API ==="

# Check if API already exists
EXISTING_API_ID=$(aws apigatewayv2 get-apis \
    --query "Items[?Name=='$API_NAME'].ApiId | [0]" \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || echo "None")

if [[ "$EXISTING_API_ID" != "None" && -n "$EXISTING_API_ID" ]]; then
    echo "API '$API_NAME' already exists (ID: $EXISTING_API_ID)"
    API_ID="$EXISTING_API_ID"
else
    API_ID=$(aws apigatewayv2 create-api \
        --name "$API_NAME" \
        --protocol-type HTTP \
        --query "ApiId" \
        --output text \
        --region "$AWS_REGION")
    echo "Created API: $API_ID"

    # Create Lambda integration
    INTEGRATION_ID=$(aws apigatewayv2 create-integration \
        --api-id "$API_ID" \
        --integration-type AWS_PROXY \
        --integration-uri "$LAMBDA_ARN" \
        --payload-format-version "2.0" \
        --query "IntegrationId" \
        --output text \
        --region "$AWS_REGION")

    # Create route POST /run
    aws apigatewayv2 create-route \
        --api-id "$API_ID" \
        --route-key "POST /run" \
        --target "integrations/$INTEGRATION_ID" \
        --region "$AWS_REGION" \
        --output text

    # Create $default stage with auto-deploy
    aws apigatewayv2 create-stage \
        --api-id "$API_ID" \
        --stage-name "\$default" \
        --auto-deploy \
        --region "$AWS_REGION" \
        --output text

    # Grant API Gateway permission to invoke Lambda
    aws lambda add-permission \
        --function-name "$LAMBDA_NAME" \
        --statement-id "apigw-invoke-$API_ID" \
        --action "lambda:InvokeFunction" \
        --principal apigateway.amazonaws.com \
        --source-arn "arn:aws:execute-api:$AWS_REGION:$ACCOUNT_ID:$API_ID/*/*/*" \
        --region "$AWS_REGION" \
        --output text 2>/dev/null || true
fi

INVOKE_URL=$(aws apigatewayv2 get-api \
    --api-id "$API_ID" \
    --query "ApiEndpoint" \
    --output text \
    --region "$AWS_REGION")

# --- Done ---
echo ""
echo "=========================================="
echo "Deployment complete!"
echo "=========================================="
echo ""
echo "Invoke URL: $INVOKE_URL/run"
echo ""
echo "Test with:"
echo "  curl -s -X POST \"$INVOKE_URL/run\" \\"
echo "    -H \"Content-Type: application/json\" \\"
echo "    -d '{\"datasets\": [...], \"config\": {\"api_key\": \"your-key\"}}'"
echo ""
echo "View logs:"
echo "  aws logs tail \"/aws/lambda/$LAMBDA_NAME\" --follow --region $AWS_REGION"
