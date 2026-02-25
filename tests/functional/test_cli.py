"""Test CLI argument parsing and validation."""

import subprocess
import sys


def test_config_required():
    """sim subcommand requires -c."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "required" in result.stderr.lower() or "error" in result.stderr.lower()


def test_overwrite_and_resume_mutually_exclusive():
    """Cannot pass both -o and -r."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim", "-c", "some.yml", "-o", "-r"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not allowed" in result.stderr.lower() or "error" in result.stderr.lower()


def test_pickle_and_resume_mutually_exclusive():
    """Cannot pass both -p and -r."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim", "-c", "some.yml", "-p", "some.pickle", "-r"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not allowed" in result.stderr.lower() or "error" in result.stderr.lower()


def test_pickle_and_overwrite_mutually_exclusive():
    """Cannot pass both -p and -o."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim", "-c", "some.yml", "-p", "some.pickle", "-o"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not allowed" in result.stderr.lower() or "error" in result.stderr.lower()


def test_extend_requires_resume():
    """--extend without -r should fail."""
    result = subprocess.run(
        [sys.executable, "-m", "aegis", "sim", "-c", "some.yml", "--extend", "500"],
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
