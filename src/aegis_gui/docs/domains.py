import dash
import re
from aegis_sim.parameterization.default_parameters import DEFAULT_PARAMETERS

from aegis_sim.recording.recordingmanager import RecordingManager
from aegis_sim.submodels.reproduction.reproduction import Reproducer
from aegis_sim.submodels import predation, infection, abiotic
from aegis_sim.submodels.resources import starvation
from aegis_sim.submodels.genetics.composite.architecture import CompositeArchitecture
from aegis_sim.submodels.genetics.modifying.architecture import ModifyingArchitecture
from aegis_sim.submodels.genetics.envdrift import Envdrift


def extract_gui_from_docstring(class_):
    docstring = class_.__doc__
    gui_section = docstring.split("GUI")[1]
    # text = replace_params_in_brackets_with_span(gui_section)
    parsed = parse_gui_docstrings(gui_section)
    return list(parsed)


def parse_gui_docstrings(text):
    pattern = r"(\[\[|\]\])"
    parts = re.split(pattern, text)
    is_parameter = False
    for part in parts:
        if part == "[[":
            is_parameter = True
        elif part == "]]":
            is_parameter = False
        else:
            if is_parameter:
                yield get_parameter_span(part)
            else:
                yield dash.html.Span(part)


def highlight_terms_in_markdown(class_):
    docstring = class_.__doc__
    gui_section = docstring.split("GUI")[1]
    text = gui_section
    if not isinstance(text, str):
        text = str(text)

    def replace_highlight(match):
        return f"**{match.group(1)}**"

    pattern = r"\[\[(.*?)\]\]"
    highlighted_text = re.sub(pattern, replace_highlight, text)
    highlighted_text = highlighted_text.replace("\n\n    ", "\n    ")
    highlighted_text = highlighted_text.replace("\n    ", "\n")
    return highlighted_text


def get_parameter_span(name):
    reformatted_name = name.replace("_", " ").lower()
    param = DEFAULT_PARAMETERS[name]
    info = param.info
    return dash.html.Span(
        reformatted_name,
        title=info,
        # style={"font-style": "italic"},
    )


genetics_text = """
GUI
Every individual carries their own genome. In AEGIS, those are bit strings (arrays of 0's and 1's), passed on from parent to offspring, and mutated in the process.
The submodel genetics transforms genomes into phenotypes; more specifically – into intrinsic phenotypes – biological potentials to exhibit a certain trait (e.g. probability to reproduce).
These potentials are either realized or not realized, depending on the environment (e.g. availability of resources), interaction with other individuals (e.g. availability of mates) and interaction with other traits (e.g. survival).

In AEGIS, genetics is simplified in comparison to the biological reality – it references no real genes and it simulates no molecular interactions; thus, it cannot be used to answer questions about specific genes, metabolic pathways or molecular mechanisms.
However, in AEGIS, in comparison to empirical datasets, genes are fully functionally characterized (in terms of their impact on the phenotype), and are to be studied as functional, heritable genetic elements – in particular, their evolutionary dynamics.

The configuration of genetics – the genetic architecture – is highly flexible. This includes specifying which traits are evolvable number of genetic elements (i.e. size of genome)...
AEGIS offers two genetic architectures – composite and modifying. They are mutually exclusive and are described in detail below...
"""


def get_texts(func):
    return {
        "starvation": func(starvation.Starvation),
        "predation": func(predation.Predation),
        "infection": func(infection.Infection),
        "abiotic": func(abiotic.Abiotic),
        "reproduction": func(Reproducer),
        "recording": func(RecordingManager),
        "genetics": genetics_text,
        "composite genetic architecture": func(CompositeArchitecture),
        "modifying genetic architecture": func(ModifyingArchitecture),
        "environmental drift": func(Envdrift),
        "technical": "",  # TODO add
        "other": "",  # TODO add
    }


TEXTS_DOMAIN = get_texts(extract_gui_from_docstring)
TEXTS_DOMAIN_MD = get_texts(highlight_terms_in_markdown)
