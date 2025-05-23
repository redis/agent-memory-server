#!/bin/bash

# Redis Memory Server - Kubernetes Deployment Script
set -e

# Configuration
REGISTRY=${REGISTRY:-your-registry.com}
IMAGE_NAME=${IMAGE_NAME:-redis-memory-server}
TAG=${TAG:-latest}
NAMESPACE=redis-memory-server

echo "🚀 Deploying Redis Memory Server to Kubernetes"
echo "Registry: $REGISTRY"
echo "Image: $IMAGE_NAME:$TAG"
echo "Namespace: $NAMESPACE"

# Build and push Docker image
echo "📦 Building and pushing Docker image..."
docker build -t $REGISTRY/$IMAGE_NAME:$TAG ../../
docker push $REGISTRY/$IMAGE_NAME:$TAG

echo "✅ Docker image pushed to registry"

# Update image references in manifests
echo "📝 Updating image references in manifests..."
sed -i.bak "s|YOUR_REGISTRY/redis-memory-server:latest|$REGISTRY/$IMAGE_NAME:$TAG|g" api-deployment.yaml
sed -i.bak "s|YOUR_REGISTRY/redis-memory-server:latest|$REGISTRY/$IMAGE_NAME:$TAG|g" worker-deployment.yaml

# Apply Kubernetes manifests
echo "🔧 Applying Kubernetes manifests..."

# Create namespace
kubectl apply -f namespace.yaml

# Create secrets (you'll need to update secrets.yaml with actual values first)
echo "⚠️  Make sure to update secrets.yaml with your actual API keys before deploying!"
read -p "Have you updated the secrets? (y/N): " confirm
if [[ $confirm != [yY] ]]; then
    echo "❌ Please update secrets.yaml and run the script again"
    exit 1
fi

# Apply configuration
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# Deploy Redis
kubectl apply -f redis.yaml

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/redis -n $NAMESPACE

# Deploy API and Worker
kubectl apply -f api-deployment.yaml
kubectl apply -f worker-deployment.yaml

# Wait for deployments to be ready
echo "⏳ Waiting for API deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/redis-memory-server-api -n $NAMESPACE

echo "⏳ Waiting for Worker deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/redis-memory-server-worker -n $NAMESPACE

echo "✅ Deployment complete!"

# Show deployment status
echo "📊 Deployment Status:"
kubectl get pods -n $NAMESPACE
kubectl get services -n $NAMESPACE
kubectl get ingress -n $NAMESPACE

# Instructions for accessing the service
echo ""
echo "🔗 Access Instructions:"
echo "  Local access: kubectl port-forward svc/redis-memory-server-api 8000:8000 -n $NAMESPACE"
echo "  Then visit: http://localhost:8000/health"
echo ""
echo "  For production access, configure your ingress domain in api-deployment.yaml"

# Clean up backup files
rm -f *.yaml.bak
