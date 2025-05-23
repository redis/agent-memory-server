# Redis Memory Server Deployment Guide

This guide covers deploying the Redis Memory Server to production environments using either AWS Fargate or Kubernetes.

## 🏗️ Architecture Overview

The Redis Memory Server consists of three main components:

1. **API Server** - REST API and MCP server for client interactions
2. **Background Workers** - Process long-term memory tasks (summarization, deduplication, etc.)
3. **Redis** - Primary data store with RedisVL for vector search

## 📋 Prerequisites

### Common Prerequisites
- Docker installed and configured
- API keys for OpenAI and/or Anthropic
- Basic understanding of containerized deployments

### AWS Fargate Prerequisites
- AWS CLI installed and configured
- IAM permissions for ECS, ECR, VPC, and Secrets Manager
- Existing VPC with subnets (or willingness to create them)

### Kubernetes Prerequisites
- `kubectl` installed and configured
- Access to a Kubernetes cluster (EKS, GKE, AKS, or self-managed)
- Docker registry for storing images

## 🚀 AWS Fargate Deployment

AWS Fargate provides serverless containers with automatic scaling and no server management.

### Step 1: Prepare Infrastructure

```bash
# Set required environment variables
export AWS_REGION=us-east-1
export VPC_ID=vpc-xxxxxxxxx           # Your VPC ID
export SUBNET_IDS=subnet-xxx,subnet-yyy  # Comma-separated subnet IDs
```

### Step 2: Store Secrets in AWS Secrets Manager

```bash
# Create secrets for API keys and Redis URL
aws secretsmanager create-secret \
  --name redis-memory-server/openai-api-key \
  --secret-string "your-openai-api-key" \
  --region $AWS_REGION

aws secretsmanager create-secret \
  --name redis-memory-server/anthropic-api-key \
  --secret-string "your-anthropic-api-key" \
  --region $AWS_REGION

# For managed Redis (ElastiCache), use your connection string:
aws secretsmanager create-secret \
  --name redis-memory-server/redis-url \
  --secret-string "redis://your-elasticache-endpoint:6379" \
  --region $AWS_REGION
```

### Step 3: Deploy with Fargate

```bash
cd deployment/fargate
chmod +x deploy.sh
./deploy.sh
```

### Step 4: Set Up Load Balancer (Optional)

For production traffic, create an Application Load Balancer:

```bash
# Create ALB targeting your ECS tasks
aws elbv2 create-load-balancer \
  --name redis-memory-server-alb \
  --subnets $SUBNET_IDS \
  --security-groups $SECURITY_GROUP_ID
```

## ☸️ Kubernetes Deployment

Kubernetes provides more control and is suitable for multi-cloud or on-premises deployments.

### Step 1: Prepare Your Image Registry

```bash
# Example for Docker Hub
export REGISTRY=your-dockerhub-username
export IMAGE_NAME=redis-memory-server

# Example for AWS ECR
export REGISTRY=123456789012.dkr.ecr.us-east-1.amazonaws.com
export IMAGE_NAME=redis-memory-server

# Example for Google Container Registry
export REGISTRY=gcr.io/your-project-id
export IMAGE_NAME=redis-memory-server
```

### Step 2: Configure Secrets

Edit `deployment/k8s/secrets.yaml` and add your base64-encoded secrets:

```bash
# Encode your secrets
echo -n "redis://redis:6379" | base64
echo -n "your-openai-api-key" | base64
echo -n "your-anthropic-api-key" | base64
```

Or create secrets directly with kubectl:

```bash
kubectl create secret generic redis-memory-server-secrets \
  --namespace=redis-memory-server \
  --from-literal=REDIS_URL="redis://redis:6379" \
  --from-literal=OPENAI_API_KEY="your-openai-key" \
  --from-literal=ANTHROPIC_API_KEY="your-anthropic-key"
```

### Step 3: Deploy to Kubernetes

```bash
cd deployment/k8s
chmod +x deploy.sh

# Set your registry
export REGISTRY=your-registry.com
./deploy.sh
```

### Step 4: Configure Ingress (Production)

Update `deployment/k8s/api-deployment.yaml` with your domain:

```yaml
spec:
  rules:
  - host: api.your-domain.com  # Replace with your domain
```

## 🔧 Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `LONG_TERM_MEMORY` | `True` | Enable long-term memory features |
| `WINDOW_SIZE` | `20` | Number of recent messages to keep in context |
| `GENERATION_MODEL` | `gpt-4o-mini` | Model for text generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embeddings |
| `ENABLE_TOPIC_EXTRACTION` | `True` | Enable topic modeling |
| `ENABLE_NER` | `True` | Enable named entity recognition |

### Resource Requirements

#### Minimum Resources (Development)
- **API Server**: 512MB RAM, 0.2 CPU cores
- **Worker**: 512MB RAM, 0.2 CPU cores
- **Redis**: 256MB RAM, 0.1 CPU cores

#### Recommended Resources (Production)
- **API Server**: 1GB RAM, 0.5 CPU cores (2+ replicas)
- **Worker**: 1GB RAM, 0.5 CPU cores (1+ replicas)
- **Redis**: 2GB RAM, 0.5 CPU cores + persistent storage

## 📊 Monitoring and Observability

### Health Checks

The API server provides a health endpoint at `/health`:

```bash
curl http://your-api-endpoint/health
# Response: {"now": 1234567890}
```

### Logging

Logs are structured JSON and include:
- Request IDs for tracing
- Error details and stack traces
- Performance metrics
- Background task status

### Metrics

Key metrics to monitor:
- API response times
- Redis connection pool health
- Memory usage for embeddings
- Background task queue length
- LLM API rate limits and costs

## 🔒 Security Considerations

### API Keys Management
- Store API keys in secure secret management (AWS Secrets Manager, K8s Secrets)
- Rotate keys regularly
- Use separate keys for different environments

### Network Security
- Use private subnets for worker tasks
- Implement proper security groups/network policies
- Consider VPC endpoints for AWS services

### Redis Security
- Use Redis AUTH when possible
- Enable Redis SSL/TLS in production
- Consider Redis sentinel for high availability

### Access Control
- Implement authentication/authorization for your API
- Use network-level restrictions
- Consider API rate limiting

## 🚨 Troubleshooting

### Common Issues

#### API Server Won't Start
```bash
# Check logs
kubectl logs deployment/redis-memory-server-api -n redis-memory-server

# Common causes:
# - Missing API keys
# - Redis connection issues
# - Insufficient resources
```

#### Worker Tasks Not Processing
```bash
# Check worker logs
kubectl logs deployment/redis-memory-server-worker -n redis-memory-server

# Verify task queue
redis-cli -u $REDIS_URL LLEN agent_memory_server_docket_tasks
```

#### High Memory Usage
```bash
# Check Redis memory usage
redis-cli -u $REDIS_URL INFO memory

# Consider:
# - Reducing WINDOW_SIZE
# - Running memory compaction tasks
# - Scaling up resources
```

## 📈 Scaling Guidelines

### Horizontal Scaling
- **API Servers**: Scale based on request volume (HPA configured for 70% CPU)
- **Workers**: Scale based on task queue length and processing time
- **Redis**: Consider Redis Cluster for very large datasets

### Vertical Scaling
- Increase memory for better embedding performance
- Increase CPU for faster text processing
- Monitor and adjust based on actual usage patterns

## 🔄 Updates and Maintenance

### Rolling Updates
Both deployment methods support rolling updates:

```bash
# Kubernetes
kubectl set image deployment/redis-memory-server-api api=your-registry/redis-memory-server:new-tag -n redis-memory-server

# AWS Fargate - update task definition and service
aws ecs update-service --cluster redis-memory-server-cluster --service redis-memory-server-api --force-new-deployment
```

### Backup and Restore
- Redis data should be backed up regularly
- Consider Redis persistence configuration
- Test restore procedures

## 📞 Support

For deployment issues:
1. Check the troubleshooting section above
2. Review application logs
3. Verify configuration and secrets
4. Check resource availability
5. Open an issue with detailed logs and configuration
