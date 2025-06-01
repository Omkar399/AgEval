#!/bin/bash
# Production build script for AgEval

set -e

echo "🏗️  Building AgEval for Production"
echo "================================="

# Build React frontend
echo "📦 Building React frontend..."
cd frontend/ageval-dashboard
npm run build
cd ../..

# Create production directory
echo "📁 Creating production directory..."
mkdir -p production
cp -r frontend/ageval-dashboard/build production/frontend

# Copy backend files
echo "📂 Copying backend files..."
cp -r backend production/
cp -r src production/
cp -r config production/
cp requirements.txt production/

# Create production start script
echo "📝 Creating production start script..."
cat > production/start_production.sh << 'EOF'
#!/bin/bash
# Production start script

set -e

echo "🚀 Starting AgEval in Production Mode"

# Install dependencies
pip install -r requirements.txt

# Start backend with production settings
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
EOF

chmod +x production/start_production.sh

echo ""
echo "✅ Production build completed!"
echo ""
echo "Production files are in the 'production' directory."
echo "To start in production mode:"
echo "  cd production"
echo "  ./start_production.sh"
echo ""
echo "Frontend will be served by FastAPI at http://localhost:8000"