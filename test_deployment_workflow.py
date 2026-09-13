from pathlib import Path


WORKFLOW = Path(__file__).parent / ".github" / "workflows" / "deploy.yml"


def test_deployment_workflow_serializes_runs_and_rolls_back_on_termination():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "cancel-in-progress: false" in workflow
    assert "trap 'rollback $?' ERR" in workflow
    assert "trap 'rollback 129' HUP" in workflow
    assert "trap 'rollback 130' INT" in workflow
    assert "trap 'rollback 143' TERM" in workflow
    assert "trap - ERR HUP INT TERM" in workflow
