"""Season tracker

Enables non-overlapping generations by tracking seasons.
"""

countdown = None
STAGES_PER_SEASON = None


def init(self, STAGES_PER_SEASON):
    self.countdown = float("inf") if STAGES_PER_SEASON == 0 else STAGES_PER_SEASON
    self.STAGES_PER_SEASON = STAGES_PER_SEASON


def restart():
    """Reset the season length tracker."""
    global countdown
    countdown += STAGES_PER_SEASON


def tick():
    global countdown
    countdown -= 1


def is_over():
    return countdown == 0


def not_applicable():
    return countdown == float("inf")
