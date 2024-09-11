import logging


def set_logging(level):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(module)s: %(message)s",
        datefmt="%d/%m/%Y %I:%M:%S",
        level=level,  # logging.DEBUG, logging.INFO, ...
    )
