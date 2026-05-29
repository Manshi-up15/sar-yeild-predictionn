import pytest
from pathlib import Path

def test_dashboard_syntax():
    """Verify that the Streamlit dashboard script is syntactically correct and can compile."""
    dashboard_path = Path(__file__).parent.parent / "src" / "crop_yield" / "app" / "dashboard.py"
    assert dashboard_path.exists(), f"Dashboard path not found: {dashboard_path}"
    
    with open(dashboard_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    try:
        # Compile code block to confirm syntax correctness
        compile(content, str(dashboard_path), "exec")
    except SyntaxError as e:
        pytest.fail(f"Syntax error in dashboard.py: {e}")
