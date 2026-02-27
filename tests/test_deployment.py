"""Tests for deployment configuration (Issue #18)."""

import os
import pytest
from pathlib import Path


def test_dockerfile_exists():
    """测试 Dockerfile 存在。"""
    dockerfile = Path("Dockerfile")
    assert dockerfile.exists(), "Dockerfile 不存在"


def test_dockerfile_contains_required_directives():
    """测试 Dockerfile 包含必要的指令。"""
    dockerfile = Path("Dockerfile")
    content = dockerfile.read_text()

    required_directives = [
        "FROM python:",
        "WORKDIR /app",
        "EXPOSE 8000",
        "HEALTHCHECK",
        'CMD ["uv", "run", "uvicorn"',
    ]

    for directive in required_directives:
        assert directive in content, f"Dockerfile 缺少指令: {directive}"


def test_docker_compose_exists():
    """测试 docker-compose.yml 存在。"""
    compose = Path("docker-compose.yml")
    assert compose.exists(), "docker-compose.yml 不存在"


def test_docker_compose_contains_required_services():
    """测试 docker-compose.yml 包含必要的服务。"""
    compose = Path("docker-compose.yml")
    content = compose.read_text()

    required_services = ["postgres", "api"]

    for service in required_services:
        assert f"{service}:" in content, f"docker-compose.yml 缺少服务: {service}"


def test_docker_compose_contains_required_volumes():
    """测试 docker-compose.yml 包含必要的 volumes。"""
    compose = Path("docker-compose.yml")
    content = compose.read_text()

    required_volume_patterns = ["postgres_data:", "chroma_data:"]

    for pattern in required_volume_patterns:
        assert pattern in content, f"docker-compose.yml 缺少 volume: {pattern}"


def test_nginx_conf_exists():
    """测试 nginx.conf 存在。"""
    nginx = Path("nginx.conf")
    assert nginx.exists(), "nginx.conf 不存在"


def test_nginx_conf_contains_required_directives():
    """测试 nginx.conf 包含必要的配置。"""
    nginx = Path("nginx.conf")
    content = nginx.read_text()

    required_directives = [
        "listen 80",
        "proxy_pass http://127.0.0.1:8000",
        "client_max_body_size",
        "location /health",
    ]

    for directive in required_directives:
        assert directive in content, f"nginx.conf 缺少配置: {directive}"


def test_deploy_script_exists():
    """测试 deploy.sh 存在且可执行。"""
    deploy = Path("scripts/deploy.sh")
    assert deploy.exists(), "scripts/deploy.sh 不存在"
    assert os.access(deploy, os.X_OK), "scripts/deploy.sh 不可执行"


def test_deploy_script_contains_required_commands():
    """测试 deploy.sh 包含必要的命令。"""
    deploy = Path("scripts/deploy.sh")
    content = deploy.read_text()

    required_commands = [
        "rsync",
        "docker-compose",
        "curl",
        "health",
    ]

    for command in required_commands:
        assert command in content, f"deploy.sh 缺少命令: {command}"


def test_deploy_script_has_correct_server():
    """测试 deploy.sh 包含正确的服务器地址。"""
    deploy = Path("scripts/deploy.sh")
    content = deploy.read_text()

    assert "43.167.189.165" in content, "deploy.sh 缺少正确的服务器地址"


def test_env_example_exists():
    """测试 .env.example 存在（作为 .env 模板）。"""
    env_example = Path(".env.example")
    assert env_example.exists(), ".env.example 不存在"


def test_env_example_contains_required_vars():
    """测试 .env.example 包含必要的环境变量。"""
    env_example = Path(".env.example")
    if env_example.exists():
        content = env_example.read_text()

        required_vars = [
            "CHATGPT_API_KEY",
            "DATABASE_URL",
            "CHROMA_PERSIST_DIR",
        ]

        for var in required_vars:
            assert var in content, f".env.example 缺少变量: {var}"


def test_dockerignore_exists():
    """测试 .dockerignore 存在（优化构建）。"""
    dockerignore = Path(".dockerignore")
    # 可选文件，但建议存在
    if dockerignore.exists():
        content = dockerignore.read_text()
        # 应该排除不必要的文件
        excluded_patterns = [".git", "__pycache__", "*.pyc", ".pytest_cache"]
        for pattern in excluded_patterns:
            assert pattern in content or any(line.strip().startswith(pattern) for line in content.splitlines()), \
                f".dockerignore 应该排除: {pattern}"
