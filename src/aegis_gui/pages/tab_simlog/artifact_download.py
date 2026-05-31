"""Per-artifact download buttons (fasta/, vcf/, lineage/, selection/).

Adds one download button per v3 output subdirectory that actually contains
data for this sim. Each button zips just that subdir and serves it — much
smaller than the "Download data" zip-everything button in zipp.py, which
is overkill when a student only needs the FASTA files.

Buttons are pattern-matched by ("type": "artifact-download", "index":
"<sim_name>::<artifact>") so the callback can resolve both the sim and
the subdirectory server-side from the triggered id alone (no client-
supplied paths trusted).
"""

import io
import logging
import pathlib
import zipfile

from dash import html, dcc, Input, Output, MATCH, callback, ctx, no_update
import dash_bootstrap_components as dbc

from aegis_gui.utilities import utilities


# (subdir_name, button_label, bootstrap-icon)
KNOWN_ARTIFACTS = (
    ("fasta", "Download FASTA", "bi-file-earmark-text-fill"),
    ("vcf", "Download VCF", "bi-table"),
    ("lineage", "Download lineage CSVs", "bi-diagram-3-fill"),
    ("selection", "Download selection log", "bi-graph-up-arrow"),
    ("pickles", "Download pickles", "bi-box-seam-fill"),
)

_SEPARATOR = "::"


def make_buttons(sim_name: str, sim_path: pathlib.Path):
    """Return a list of (button, dcc.Download) tuples for every artifact
    subdir that exists and contains at least one file. Empty list if none.
    """
    elements = []
    for subdir, label, icon in KNOWN_ARTIFACTS:
        artifact_dir = sim_path / subdir
        if not artifact_dir.is_dir():
            continue
        if not any(artifact_dir.iterdir()):
            continue
        index = f"{sim_name}{_SEPARATOR}{subdir}"
        elements.append(
            html.Div(
                children=[
                    dbc.Button(
                        [html.I(className=f"bi {icon}"), label],
                        id={"type": "artifact-download-button", "index": index},
                        color="secondary",
                        outline=True,
                        className="me-2 mb-2",
                        size="sm",
                    ),
                    dcc.Download(id={"type": "artifact-download-dcc", "index": index}),
                ],
                style={"display": "inline-block"},
            )
        )
    return elements


@callback(
    Output({"type": "artifact-download-dcc", "index": MATCH}, "data"),
    Input({"type": "artifact-download-button", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def download_artifact(n_clicks):
    if n_clicks is None:
        return no_update
    triggered = ctx.triggered_id
    if not isinstance(triggered, dict) or "index" not in triggered:
        return no_update

    index = triggered["index"]
    if _SEPARATOR not in index:
        logging.warning("Rejected artifact download (bad index): %r", index)
        return no_update
    sim_name, _, subdir = index.partition(_SEPARATOR)

    # Whitelist: subdir must be one we know about. No path traversal possible
    # because the subdir name itself is checked against a hard-coded list.
    known_subdirs = {entry[0] for entry in KNOWN_ARTIFACTS}
    if subdir not in known_subdirs:
        logging.warning("Rejected artifact download (unknown subdir): %r", subdir)
        return no_update

    try:
        sim_path = utilities.safe_sim_path(sim_name)
    except utilities.UnsafeSimNameError as exc:
        logging.warning("Rejected artifact download: %s", exc)
        return no_update

    artifact_dir = sim_path / subdir
    if not artifact_dir.is_dir():
        logging.warning("Artifact dir %s vanished between render and click", artifact_dir)
        return no_update

    zip_buffer = _zip_folder(artifact_dir)
    return dcc.send_bytes(zip_buffer.getvalue(), f"{sim_name}-{subdir}.zip")


def _zip_folder(folder_path: pathlib.Path) -> io.BytesIO:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in folder_path.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(folder_path))
    buf.seek(0)
    return buf
