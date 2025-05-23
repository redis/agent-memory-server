#!/bin/bash

# Redis Memory Server - AWS Fargate Deployment Script
set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
ECR_REPOSITORY=${ECR_REPOSITORY:-redis-memory-server}
ECS_CLUSTER=${ECS_CLUSTER:-redis-memory-server-cluster}
VPC_ID=${VPC_ID}
SUBNET_IDS=${SUBNET_IDS}  # Comma-separated list
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "🚀 Deploying Redis Memory Server to AWS Fargate"
echo "Region: $AWS_REGION"
echo "Account: $ACCOUNT_ID"

# Build and push Docker image to ECR
echo "📦 Building and pushing Docker image..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION 2>/dev/null || \
aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION

# Build and tag image
docker build -t $ECR_REPOSITORY .
docker tag $ECR_REPOSITORY:latest $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

# Push to ECR
docker push $ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest

echo "✅ Docker image pushed to ECR"

# Create ECS cluster if it doesn't exist
aws ecs describe-clusters --clusters $ECS_CLUSTER --region $AWS_REGION 2>/dev/null || \
aws ecs create-cluster --cluster-name $ECS_CLUSTER --capacity-providers FARGATE --region $AWS_REGION

echo "✅ ECS cluster ready"

# Create security group for the application
SECURITY_GROUP_ID=$(aws ec2 create-security-group \
  --group-name redis-memory-server-sg \
  --description "Security group for Redis Memory Server" \
  --vpc-id $VPC_ID \
  --region $AWS_REGION \
  --query 'GroupId' \
  --output text 2>/dev/null || \
  aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=redis-memory-server-sg" "Name=vpc-id,Values=$VPC_ID" \
  --query 'SecurityGroups[0].GroupId' \
  --output text)

# Add ingress rules
aws ec2 authorize-security-group-ingress \
  --group-id $SECURITY_GROUP_ID \
  --protocol tcp \
  --port 8000 \
  --cidr 0.0.0.0/0 \
  --region $AWS_REGION 2>/dev/null || true

echo "✅ Security group configured: $SECURITY_GROUP_ID"

# Update task definitions with correct values
sed -i.bak "s/ACCOUNT_ID/$ACCOUNT_ID/g" task-definition-api.json
sed -i.bak "s/YOUR_ECR_REPOSITORY_URI/$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com\/$ECR_REPOSITORY/g" task-definition-api.json
sed -i.bak "s/us-east-1/$AWS_REGION/g" task-definition-api.json

sed -i.bak "s/ACCOUNT_ID/$ACCOUNT_ID/g" task-definition-worker.json
sed -i.bak "s/YOUR_ECR_REPOSITORY_URI/$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com\/$ECR_REPOSITORY/g" task-definition-worker.json
sed -i.bak "s/us-east-1/$AWS_REGION/g" task-definition-worker.json

# Register task definitions
echo "📋 Registering ECS task definitions..."
aws ecs register-task-definition --cli-input-json file://task-definition-api.json --region $AWS_REGION
aws ecs register-task-definition --cli-input-json file://task-definition-worker.json --region $AWS_REGION

# Create or update ECS services
echo "🚀 Creating/updating ECS services..."

# API Service
aws ecs create-service \
  --cluster $ECS_CLUSTER \
  --service-name redis-memory-server-api \
  --task-definition redis-memory-server-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
  --region $AWS_REGION 2>/dev/null || \
aws ecs update-service \
  --cluster $ECS_CLUSTER \
  --service redis-memory-server-api \
  --task-definition redis-memory-server-api \
  --region $AWS_REGION

# Worker Service
aws ecs create-service \
  --cluster $ECS_CLUSTER \
  --service-name redis-memory-server-worker \
  --task-definition redis-memory-server-worker \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SECURITY_GROUP_ID],assignPublicIp=ENABLED}" \
  --region $AWS_REGION 2>/dev/null || \
aws ecs update-service \
  --cluster $ECS_CLUSTER \
  --service redis-memory-server-worker \
  --task-definition redis-memory-server-worker \
  --region $AWS_REGION

echo "✅ Deployment complete!"
echo "🔗 Your API will be available at the public IPs of the running tasks"
echo "   Use 'aws ecs list-tasks --cluster $ECS_CLUSTER --region $AWS_REGION' to find task ARNs"
echo "   Then 'aws ecs describe-tasks' to get public IPs"
