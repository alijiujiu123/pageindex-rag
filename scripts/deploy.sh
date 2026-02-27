#!/bin/bash
# 部署脚本: 将 pageindex-rag 部署到 dongjingTest 服务器
# 用法: ./scripts/deploy.sh

set -e

# 服务器配置
SERVER="root@43.167.189.165"
PROJECT_DIR="/root/pageindex-rag"
SERVICE_NAME="pageindex-rag"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}PageIndex RAG 部署脚本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查本地是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo -e "${YELLOW}警告: 有未提交的更改，是否继续？(y/N)${NC}"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "部署取消"
        exit 1
    fi
fi

# 1. 同步代码到服务器
echo -e "${GREEN}[1/6] 同步代码到服务器...${NC}"
rsync -avz --delete \
    --exclude='.git' \
    --exclude='.worktrees' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache' \
    --exclude='chroma_data' \
    --exclude='results' \
    --exclude='financebench_pdfs' \
    --exclude='.idea' \
    --exclude='uv.lock' \
    ./ "$SERVER:$PROJECT_DIR/"

# 2. 在服务器上创建 .env 文件（如果不存在）
echo -e "${GREEN}[2/6] 配置环境变量...${NC}"
ssh "$SERVER" << 'ENDSSH'
cd /root/pageindex-rag
if [ ! -f .env ]; then
    echo "创建 .env 文件..."
    cat > .env << 'EOF'
CHATGPT_API_KEY=your_openai_key_here
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/pageindex_rag
CHROMA_PERSIST_DIR=./chroma_data
EOF
    echo "请编辑 .env 文件并填入正确的 API Key"
fi
ENDSSH

# 3. 停止旧容器
echo -e "${GREEN}[3/6] 停止旧容器...${NC}"
ssh "$SERVER" << 'ENDSSH'
cd /root/pageindex-rag
docker-compose down || true
ENDSSH

# 4. 构建新镜像
echo -e "${GREEN}[4/6] 构建 Docker 镜像...${NC}"
ssh "$SERVER" << 'ENDSSH'
cd /root/pageindex-rag
docker-compose build --no-cache
ENDSSH

# 5. 启动新容器
echo -e "${GREEN}[5/6] 启动容器...${NC}"
ssh "$SERVER" << 'ENDSSH'
cd /root/pageindex-rag
docker-compose up -d
ENDSSH

# 6. 等待健康检查
echo -e "${GREEN}[6/6] 等待服务启动...${NC}"
sleep 10

# 测试健康检查端点
echo -e "${GREEN}测试健康检查端点...${NC}"
if ssh "$SERVER" "curl -f -s http://localhost:8000/health > /dev/null"; then
    echo -e "${GREEN}✓ 健康检查通过${NC}"
else
    echo -e "${RED}✗ 健康检查失败${NC}"
    exit 1
fi

# 测试 QA 端点
echo -e "${GREEN}测试 QA 端点...${NC}
HEALTH_CHECK=$(curl -s -X POST http://43.167.189.165/qa \
    -H "Content-Type: application/json" \
    -d '{"query": "test"}' \
    -w "%{http_code}" -o /dev/null)

if [ "$HEALTH_CHECK" = "200" ] || [ "$HEALTH_CHECK" = "400" ]; then
    echo -e "${GREEN}✓ QA 端点可用${NC}"
else
    echo -e "${YELLOW}⚠ QA 端点返回: $HEALTH_CHECK${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}部署完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "服务器: $SERVER"
echo "项目目录: $PROJECT_DIR"
echo ""
echo "常用命令:"
echo "  查看日志: ssh $SERVER 'cd $PROJECT_DIR && docker-compose logs -f api'"
echo "  重启服务: ssh $SERVER 'cd $PROJECT_DIR && docker-compose restart api'"
echo "  停止服务: ssh $SERVER 'cd $PROJECT_DIR && docker-compose down'"
echo "  进入容器: ssh $SERVER 'docker exec -it pageindex-api bash'"
echo ""
echo "API 端点:"
echo "  http://43.167.189.165/health"
echo "  http://43.167.189.165/documents"
echo "  http://43.167.189.165/search"
echo "  http://43.167.189.165/qa"
