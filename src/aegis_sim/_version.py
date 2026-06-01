"""Runtime version information for AEGIS.

Exposes three identifiers a sysadmin can check from the GUI footer:
  - version      package version from setup.py (e.g. "2.3.1")
  - commit       short git commit SHA captured at container build time
                 (falls back to "unknown" if not provided)
  - build_date   ISO-8601 date captured at container build time
                 (falls back to "unknown" if not provided)

The container build is expected to inject commit and build_date via
environment variables (AEGIS_GIT_COMMIT, AEGIS_BUILD_DATE) so the
resulting image is identifiable without inspecting the .git directory
inside it. See the deploy notes in docs/deploy.md.
"""

import os
from importlib import metadata


def _package_version() -> str:
    # Prefer setuptools-discovered metadata (works from pip install -e .)
    try:
        return metadata.version("aegis-sim")
    except metadata.PackageNotFoundError:
        pass
    # Fallback to setup.py's hard-coded literal so this still resolves
    # when the package was checked out but never installed.
    try:
        from pathlib import Path
        import re

        setup_py = Path(__file__).resolve().parents[2] / "setup.py"
        text = setup_py.read_text()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "unknown"


version = _package_version()
commit = os.environ.get("AEGIS_GIT_COMMIT", "unknown")
build_date = os.environ.get("AEGIS_BUILD_DATE", "unknown")


def version_string() -> str:
    """Human-readable one-liner suitable for GUI footers / logs."""
    parts = [f"v{version}"]
    if commit != "unknown":
        parts.append(f"commit {commit}")
    if build_date != "unknown":
        parts.append(f"built {build_date}")
    return " · ".join(parts)
