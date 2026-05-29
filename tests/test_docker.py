import pytest
from pathlib import Path
import yaml

def test_dockerfile_contents():
    """Verify that Dockerfile exists and contains required layers."""
    dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
    assert dockerfile_path.exists()
    
    content = dockerfile_path.read_text()
    assert "FROM python" in content
    assert "WORKDIR /app" in content
    assert "COPY requirements.txt" in content
    assert "EXPOSE 8000" in content
    assert "EXPOSE 8501" in content


def test_docker_compose_syntax():
    """Verify that docker-compose.yml is valid YAML and defines both app services."""
    compose_path = Path(__file__).parent.parent / "docker-compose.yml"
    assert compose_path.exists()
    
    with open(compose_path, "r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid YAML syntax in docker-compose.yml: {e}")
            
    assert "services" in data
    services = data["services"]
    assert "api" in services
    assert "dashboard" in services
    
    # Assert network configuration
    assert "networks" in data
