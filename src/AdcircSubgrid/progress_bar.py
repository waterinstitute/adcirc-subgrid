import logging
import os

from tqdm import tqdm


class ProgressBar:
    """
    A progress bar which uses a logger for output
    """

    def __init__(self, total: int, interval: float, logger: logging.Logger) -> None:
        """
        Initialize the progress bar

        Args:
            total: The total number of iterations
            interval: The percent interval to update the progress bar
            logger: The logger to use for updating the progress
        """
        self.__progress = tqdm(
            total=total,
            ncols=50,
            file=open(os.devnull, "w"),  # noqa:SIM115
        )
        self.__interval = interval / 100.0
        self.__last_update = -1
        self.__logger = logger

    def increment(self, value: int = 1) -> None:
        """
        Add to the progress bar

        Args:
            value: The value to add to the progress bar
        """
        self.__progress.update(value)
        if (
            self.__progress.n
            >= self.__last_update + self.__interval * self.__progress.total
            or self.__progress.n == self.__progress.total
            or self.__last_update == -1
        ):
            self.__last_update = self.__progress.n
            self.__logger.info(self.__progress)
