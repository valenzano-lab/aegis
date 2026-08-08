# AEGIS deployment notes

This file documents how the AEGIS GUI is deployed at FLI (and how to
deploy it elsewhere if needed). It complements the operational manual
(`src/aegis/documentation/manual.md`) and is intended for sysadmins,
not end users.

## Current FLI deployment

The public webserver at https://genome.leibniz-fli.de/aegis/ runs as a
rootless **podman** container on the `merlin` HPC host, with the FLI
the FLI infrastructure team (Bernd Senf) managing the host and deploys.

The container is rebuilt and restarted by the sysadmin on demand. There
is no auto-deploy from GitHub: a deliberate security decision given
merlin's role inside the HPC, agreed with the FLI infrastructure team.

## What the GUI footer shows

Every page renders a small footer at the bottom showing the running
revision. Example:

    v2.3.1 · commit abc1234 · built 2026-06-01

The three pieces come from `aegis_sim/_version.py`:

| Piece        | Source                                | When unset           |
|--------------|---------------------------------------|----------------------|
| `version`    | `importlib.metadata.version("aegis-sim")`, falls back to parsing `setup.py` | `"unknown"` |
| `commit`     | env var `AEGIS_GIT_COMMIT`            | `"unknown"`          |
| `build_date` | env var `AEGIS_BUILD_DATE`            | `"unknown"`          |

So the version always works out of the box (read from package metadata).
The commit and build date need to be supplied at container build time.

## Dockerfile snippet to inject commit + build date

Add three lines to the Dockerfile, before the `pip install -e
git+...#egg=aegis-sim` line that already exists:

```dockerfile
# Capture the git revision and build date so the GUI footer shows
# exactly which AEGIS revision is running. Both are read by the
# aegis_sim._version module at app startup.
ARG AEGIS_GIT_COMMIT=unknown
ARG AEGIS_BUILD_DATE=unknown
ENV AEGIS_GIT_COMMIT=$AEGIS_GIT_COMMIT
ENV AEGIS_BUILD_DATE=$AEGIS_BUILD_DATE
```

Then pass the build args when building the image:

```bash
podman build \
  --build-arg AEGIS_GIT_COMMIT="$(git ls-remote https://github.com/valenzano-lab/aegis v2 | cut -c1-7)" \
  --build-arg AEGIS_BUILD_DATE="$(date -u +%Y-%m-%d)" \
  -t aegis-gui:latest .
```

The `git ls-remote` line resolves the *current* HEAD of the `v2` branch
without needing a local clone — useful because the container build line
already uses `pip install -e git+...` which doesn't leave a `.git`
directory inside the image.

If the build script is shell-scripted, that same pair of values is
worth logging at deploy time too, so deploy history has a verifiable
record of which AEGIS revision went out and when.

## Pinning the install line to v2 explicitly

The pip install line in the current Dockerfile is:

    RUN python3 -m pip install -e git+https://github.com/valenzano-lab/aegis.git#egg=aegis-sim

Without an explicit branch, pip resolves to the repo's *default branch*,
which is `v2`. So the line is functionally correct but the dependency
is implicit. For clarity (and to make future branch changes obvious in
the Dockerfile diff), pin it explicitly:

    RUN python3 -m pip install -e git+https://github.com/valenzano-lab/aegis.git@v2#egg=aegis-sim

No functional change today.

## What a deploy round looks like

1. Developer pushes changes to the `v2` branch on GitHub. CI runs `pytest`.
2. Developer notifies sysadmin (Bernd Senf) that a deploy is ready, with a
   one-line summary of what's in it.
3. Sysadmin rebuilds the container (10 minutes), which pulls fresh
   `v2`, runs the install, captures commit + build date.
4. Sysadmin restarts the running container.
5. Verify: load https://genome.leibniz-fli.de/aegis/ and check the
   footer at the bottom of any page — the commit short hash should
   match the head of `v2` at deploy time.

## Local development

`aegis_sim._version` reads from package metadata, which is populated
when `pip install -e ".[dev]"` runs. For local dev the footer will
show the version string but `commit unknown · built unknown`. That's
expected — only the production container build sets those env vars.
