class Frailty:
    def __init__(self, FRAILTY_MODIFIER, AGE_LIMIT):
        self.FRAILTY_MODIFIER = FRAILTY_MODIFIER
        self.AGE_LIMIT = AGE_LIMIT

    def modify(self, hazard, ages):
        amount = ages / self.AGE_LIMIT
        return hazard * (1 + amount * self.FRAILTY_MODIFIER)
