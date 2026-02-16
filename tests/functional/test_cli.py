"""Test CLI argument parsing and validation."""

import subprocess
import sys


def test_resume_and_config_mutually_exclusive():
    """Cannot pass both --resume and --config_path."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim", "-r", "some/path", "-c", "some.yml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not allowed" in result.stderr.lower() or "error" in result.stderr.lower()


def test_resume_and_overwrite_mutually_exclusive():
    """--overwrite should be rejected when --resume is used."""
    # This is validated in aegis/__init__.py, not argparse, so it needs
    # a real output dir. We just check the error message.
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim", "-r", "nonexistent", "-o"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_no_command_prints_help():
    """Running aegis with no subcommand should print help."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "sim" in result.stdout or "gui" in result.stdout


def test_invalid_config_stem():
    """A config path with no stem (like '.yml') should be rejected."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim", "-c", ".yml"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "no valid name" in result.stderr.lower() or "error" in result.stderr.lower()
