import argparse
from .args_validation import validate_args


class AegisArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('description', "Aging of Evolving Genomes in Silico")
        super().__init__(*args, **kwargs)
        self._setup_subparsers()
    
    def parse_args(self, args=None, namespace=None):
        return super().parse_args(args, namespace)
    
    def parse_and_validate(self, args=None, namespace=None):
        """Parse and validate arguments in one step."""
        parsed_args = self.parse_args(args, namespace)
        validate_args(parsed_args, self.error)
        return parsed_args
    
    def validate(self, parsed_args):
        """Validate parsed arguments. Call this after parse_args()."""
        validate_args(parsed_args, self.error)

    def _setup_subparsers(self):
        subparsers = self.add_subparsers(dest="command", help="", parser_class=argparse.ArgumentParser)
        self._setup_sim_parser(subparsers)
        self._setup_gui_parser(subparsers)

    def _setup_sim_parser(self, subparsers):
        parser = subparsers.add_parser("sim", help="run a simulation")
        parser.add_argument(
            "-c",
            "--config_path",
            help="path to config file"
        )
        parser.add_argument(
            "-p",
            "--pickle_path",
            "--pickle_paths",
            nargs="*",
            default=[],
            help="path(s) to pre-evolved population (pickle) file(s)"
        )
        parser.add_argument(
            "--pickle_weights",
            type=float,
            nargs="*",
            default=[],
            help="proportion of pre-evolved population to sample"
        )
        parser.add_argument(
            "-o",
            "--overwrite",
            action="store_true",
            help="overwrite old data with new simulation"
        )

    def _setup_gui_parser(self, subparsers):
        parser = subparsers.add_parser("gui", help="run GUI")
        parser.add_argument(
            "-s",
            "--server",
            action="store_true",
            help="run gui – the interactive GUI – in server mode"
        )
        parser.add_argument(
            "-d",
            "--debug",
            action="store_true",
            help="activate the debugger if running locally"
        )
