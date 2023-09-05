from math import inf


class EarlyStopping:
    """
    The early stopping it used to avoid the over-fitting.
    """

    def __init__(self, patience: int):
        self.patience: int = patience
        self.curr_pat: int = patience + 1
        self.current_vl: float = -inf
        self.earlyStop = False

    def update(self, vl_loss: float):
        if self.current_vl <= vl_loss:
            self.curr_pat -= 1
        else:
            self.curr_pat = self.patience
        self.current_vl = vl_loss
        if self.curr_pat == 0:
            self.earlyStop = True

