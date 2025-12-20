import pathlib


def validate_args(args, error_callback):
    """Validate parsed arguments and call error_callback if validation fails."""
    if args.command == "sim":
        _validate_sim_args(args, error_callback)


def _validate_sim_args(args, error_callback):
    """Validate simulation-specific arguments."""
    # Validate config exists
    if args.config_path and not pathlib.Path(args.config_path).exists():
        error_callback(f"Config file not found: {args.config_path}")

    # Validate pickle paths exist
    for path in args.pickle_path:
        if not pathlib.Path(path).exists():
            error_callback(f"Pickle file not found: {path}")

    # Validate weights
    if args.pickle_weights:
        if len(args.pickle_weights) != len(args.pickle_path):
            error_callback(f"Number of pickle weights ({len(args.pickle_weights)}) must match number of pickle paths ({len(args.pickle_path)})")
        if any(w <= 0 or w > 1 for w in args.pickle_weights):
            error_callback("Pickle weights must be in range (0, 1]")