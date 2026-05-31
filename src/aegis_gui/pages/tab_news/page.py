"""News tab — announcements, releases, publications.

The headline item is the PLOS Computational Biology paper. As v3 features
ship and as the lab publishes more, add new dbc.Card entries at the top
(newest first). Each card is a self-contained announcement; no shared
state, no callbacks needed.
"""

import dash
from dash import html
import dash_bootstrap_components as dbc

dash.register_page(__name__, name="AEGIS | News", path="/news")


PAPER_DOI = "10.1371/journal.pcbi.1014109"
PAPER_URL = f"https://doi.org/{PAPER_DOI}"


def _news_item(date, title, body, link=None, link_label=None, color="primary"):
    children = [
        html.Div(
            [
                dbc.Badge(date, color=color, className="me-2"),
                html.Strong(title),
            ],
            className="mb-2",
        ),
        body,
    ]
    if link:
        children.append(
            html.Div(
                dbc.Button(
                    link_label or "Read more",
                    href=link,
                    target="_blank",
                    color=color,
                    outline=True,
                    size="sm",
                    className="mt-2",
                ),
            )
        )
    return dbc.Card(dbc.CardBody(children), className="mb-3")


def layout():
    return dbc.Container(
        [
            html.H1("News"),
            html.P(
                "Announcements, software releases, and publications related to AEGIS.",
                className="text-muted mb-4",
            ),
            _news_item(
                date="2026",
                title="AEGIS paper published in PLOS Computational Biology",
                body=html.Div(
                    [
                        html.P(
                            "The methods paper describing AEGIS — model design, ODD protocol, "
                            "validation experiments — is published in PLOS Computational Biology.",
                            className="mb-2",
                        ),
                        html.Blockquote(
                            html.Em(
                                "Bagic M, Valenzano DR, et al. AEGIS: Aging of Evolving Genomes In Silico. "
                                "PLOS Computational Biology, 2026."
                            ),
                            className="ms-3",
                        ),
                        html.P(
                            [html.Strong("DOI: "), html.A(PAPER_DOI, href=PAPER_URL, target="_blank")],
                            className="mb-0",
                        ),
                    ]
                ),
                link=PAPER_URL,
                link_label="Read the paper",
                color="success",
            ),
            _news_item(
                date="2026 v2.3",
                title="v3 features land on v2: FASTA / VCF export, lineage tracing, selection coefficient, per-trait dominance",
                body=html.Div(
                    [
                        html.P(
                            "Several new outputs are now available from the simulation engine and accessible from the GUI:",
                            className="mb-2",
                        ),
                        html.Ul(
                            [
                                html.Li("FASTA export of agent genomes — feeds long-read simulators like Badread."),
                                html.Li("VCF export — direct genotype access for PLINK / vcftools / ADMIXTOOLS."),
                                html.Li(
                                    "Lineage tracing (asexual reproduction) — birth/death logs and Muller plots of clonal dynamics."
                                ),
                                html.Li(
                                    "Selection coefficient s — controlled allele injection at a configurable (trait, age, bit) locus + trajectory tracking + s estimation."
                                ),
                                html.Li("Per-trait dominance coefficient h — recessive / codominant / dominant per trait."),
                                html.Li(
                                    "Trait phenotype ranges — G_<trait>_lo / G_<trait>_hi now actually scale phenotypes (were previously dead parameters)."
                                ),
                            ]
                        ),
                        html.P(
                            "See the Wiki tab (Operational manual) for the how-to.",
                            className="mb-0",
                        ),
                    ]
                ),
                color="info",
            ),
        ],
    )
