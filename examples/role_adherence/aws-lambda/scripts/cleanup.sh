#!/usr/bin/env bash
#
# Clean up all AWS resources created by deploy.sh
#
# Usage: ./cleanup.sh <module-name> [region]
#
# Example: ./cleanup.sh bestof us-east-1
#
# Deletes:
#   - API Gateway HTTP API
#   - Lambda function
#   - IAM role (detaches policies first)
#   - ECR repository (including all images)
#   - CloudWatch log group

set -euo pipefail

# --- Arguments ---
MODULE_NAME="${1:-}"
AWS_REGION="${2:-us-east-1}"

if [[ -z "$MODULE_NAME" ]]; then
    echo "Usage: $0 <module-name> [region]"
    echo "Example: $0 bestof us-east-1"
    exit 1
fi

# --- Derived names ---
PROJECT_NAME="gaussia-${MODULE_NAME}"
ECR_REPO="$PROJECT_NAME"
LAMBDA_NAME="$PROJECT_NAME"
LAMBDA_ROLE_NAME="${PROJECT_NAME}-lambda-role"
API_NAME="${PROJECT_NAME}-http-api"
LOG_GROUP="/aws/lambda/$LAMBDA_NAME"

echo "=== Cleaning up: $PROJECT_NAME ==="
echo "Region: $AWS_REGION"
echo ""
echo "This will delete:"
echo "  - API Gateway: $API_NAME"
echo "  - Lambda: $LAMBDA_NAME"
echo "  - IAM Role: $LAMBDA_ROLE_NAME"
echo "  - ECR Repository: $ECR_REPO (including all images)"
echo "  - CloudWatch Log Group: $LOG_GROUP"
echo ""
read -p "Continue? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""

# --- Delete API Gateway ---
echo "=== Deleting API Gateway ==="
API_ID=$(aws apigatewayv2 get-apis \
    --query "Items[?Name=='$API_NAME'].ApiId | [0]" \
    --output text \
    --region "$AWS_REGION" 2>/dev/null || echo "None")

if [[ "$API_ID" != "None" && -n "$API_ID" ]]; then
    aws apigatewayv2 delete-api \
        --api-id "$API_ID" \
        --region "$AWS_REGION"
    echo "Deleted API Gateway: $API_NAME ($API_ID)"
else
    echo "API Gateway not found: $API_NAME"
fi

# --- Delete Lambda function ---
echo ""
echo "=== Deleting Lambda function ==="
if aws lambda get-function --function-name "$LAMBDA_NAME" --region "$AWS_REGION" &>/dev/null; then
    aws lambda delete-function \
        --function-name "$LAMBDA_NAME" \
        --region "$AWS_REGION"
    echo "Deleted Lambda: $LAMBDA_NAME"
else
    echo "Lambda not found: $LAMBDA_NAME"
fi

# --- Delete IAM Role ---
echo ""
echo "=== Deleting IAM role ==="
if aws iam get-role --role-name "$LAMBDA_ROLE_NAME" &>/dev/null; then
    # Detach all managed policies
    POLICIES=$(aws iam list-attached-role-policies \
        --role-name "$LAMBDA_ROLE_NAME" \
        --query "AttachedPolicies[].PolicyArn" \
        --output text)

    for policy in $POLICIES; do
        aws iam detach-role-policy \
            --role-name "$LAMBDA_ROLE_NAME" \
            --policy-arn "$policy"
        echo "Detached policy: $policy"
    done

    # Delete inline policies
    INLINE_POLICIES=$(aws iam list-role-policies \
        --role-name "$LAMBDA_ROLE_NAME" \
        --query "PolicyNames" \
        --output text)

    for policy in $INLINE_POLICIES; do
        aws iam delete-role-policy \
            --role-name "$LAMBDA_ROLE_NAME" \
            --policy-name "$policy"
        echo "Deleted inline policy: $policy"
    done

    # Delete the role
    aws iam delete-role --role-name "$LAMBDA_ROLE_NAME"
    echo "Deleted IAM role: $LAMBDA_ROLE_NAME"
else
    echo "IAM role not found: $LAMBDA_ROLE_NAME"
fi

# --- Delete ECR Repository ---
echo ""
echo "=== Deleting ECR repository ==="
if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$AWS_REGION" &>/dev/null; then
    aws ecr delete-repository \
        --repository-name "$ECR_REPO" \
        --force \
        --region "$AWS_REGION" \
        --output text
    echo "Deleted ECR repository: $ECR_REPO"
else
    echo "ECR repository not found: $ECR_REPO"
fi

# --- Delete CloudWatch Log Group ---
echo ""
echo "=== Deleting CloudWatch log group ==="
if aws logs describe-log-groups \
    --log-group-name-prefix "$LOG_GROUP" \
    --query "logGroups[?logGroupName=='$LOG_GROUP'].logGroupName" \
    --output text \
    --region "$AWS_REGION" | grep -q "$LOG_GROUP"; then
    aws logs delete-log-group \
        --log-group-name "$LOG_GROUP" \
        --region "$AWS_REGION"
    echo "Deleted log group: $LOG_GROUP"
else
    echo "Log group not found: $LOG_GROUP"
fi

echo ""
echo "=========================================="
echo "Cleanup complete!"
echo "=========================================="
