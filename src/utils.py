import logging
import logging.config
import coloredlogs


def setLogging():
    """
    [summary]

    Returns:
        [type]: [description]
    """
    logging.config.fileConfig(fname='../configs/logging.conf')
    logger = logging.getLogger(__name__)
    coloredlogs.install(level='DEBUG', logger=logger)
    return logger
