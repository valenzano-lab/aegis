class Season:
    """Season tracker
    
    Enables non-overlapping generations by tracking seasons.
    """

    def __init__(self, STAGES_PER_SEASON):
        self.countdown = float("inf") if STAGES_PER_SEASON == 0 else STAGES_PER_SEASON
        self.STAGES_PER_SEASON = STAGES_PER_SEASON

    def start_new(self):
        """Reset the season length tracker."""
        self.countdown += self.STAGES_PER_SEASON
