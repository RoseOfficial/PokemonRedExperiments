"""Tests for the W&B logger wrapper."""
from pokemon_planner.world_model.wandb_logger import WandbLogger


def test_wandb_logger_noop_when_unset(monkeypatch):
    """If WANDB_API_KEY is unset, the logger should be a silent no-op."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    logger = WandbLogger(project="test", run_name="r", config={})
    assert logger.enabled is False
    # Should not raise
    logger.log({"loss/total": 0.5}, step=1)
    logger.finish()


def test_wandb_logger_enabled_when_set(monkeypatch, mocker):
    """If WANDB_API_KEY is set, the logger should call wandb.init."""
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    mock_wandb = mocker.patch("wandb.init")
    mock_wandb.return_value = mocker.MagicMock()
    logger = WandbLogger(project="test", run_name="r", config={})
    assert logger.enabled is True
    mock_wandb.assert_called_once()
    logger.finish()


def test_wandb_logger_log_passes_through(monkeypatch, mocker):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    mock_run = mocker.MagicMock()
    mocker.patch("wandb.init", return_value=mock_run)
    logger = WandbLogger(project="test", run_name="r", config={})
    logger.log({"loss/total": 0.5}, step=1)
    mock_run.log.assert_called_once_with({"loss/total": 0.5}, step=1)


def test_wandb_logger_resume_passes_run_id(monkeypatch, mocker):
    monkeypatch.setenv("WANDB_API_KEY", "fake-key")
    mock_init = mocker.patch("wandb.init")
    mock_init.return_value = mocker.MagicMock()
    logger = WandbLogger(project="test", run_name="r", config={}, run_id="abc123")
    call_kwargs = mock_init.call_args.kwargs
    assert call_kwargs["id"] == "abc123"
    assert call_kwargs["resume"] == "allow"
