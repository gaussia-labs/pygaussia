#!/usr/bin/env bash
#
# Update an existing Gaussia Lambda function with a new container image
#
# Usage: ./update.sh <module-name> [region]
#
# Example: ./update.sh bestof us-east-1
#
# Rebuilds the Docker image from gaussia repo, pushes to ECR, and updates Lambda.

set -euo pipefail

# --- Arguments ---
MODULE_NAME="${1:-}"
AWS_REGION="${2:-us-east-1}"

if [[ -z "$MODULE_NAME" ]]; then
    echo "Usage: $0 <module-name> [region]"
    echo "Example: $0 bestof us-east-1"
    exit 1
fi

# --- Paths ---
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

echo "=== Updating Gaussia Lambda: $PROJECT_NAME ==="
echo "Region: $AWS_REGION"
echo ""

# --- Get AWS Account ID ---
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
IMAGE_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest"

# --- Step 1: Rebuild Docker image ---
echo "=== Step 1: Building Docker image ==="
docker build \
    -t "$PROJECT_NAME:latest" \
    -f "$LAMBDA_DIR/Dockerfile" \
    --build-arg MODULE_EXTRA="$MODULE_NAME" \
    "$REPO_ROOT"

# --- Step 2: Push to ECR ---
echo ""
echo "=== Step 2: Pushing to ECR ==="
aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

docker tag "$PROJECT_NAME:latest" "$IMAGE_URI"
docker push "$IMAGE_URI"

# --- Step 3: Update Lambda ---
echo ""
echo "=== Step 3: Updating Lambda function ==="
aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$AWS_REGION" \
    --output text

echo ""
echo "Waiting for Lambda update to complete..."
aws lambda wait function-updated --function-name "$LAMBDA_NAME" --region "$AWS_REGION"

echo ""
echo "=========================================="
echo "Update complete!"
echo "=========================================="
echo ""
echo "The API Gateway endpoint remains unchanged."
echo ""
echo "View logs:"
echo "  aws logs tail \"/aws/lambda/$LAMBDA_NAME\" --follow --region $AWS_REGION"
