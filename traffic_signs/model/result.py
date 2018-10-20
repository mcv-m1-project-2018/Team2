class Result:

    """
    We define and initialize the different variables for the evaluation
    of the results
    """
    tp: float
    fp_w: float
    tn: float
    fn: float
    time: float
    tp_w: float
    fp_w: float
    fn_w: float

    def __init__(self, tp=0, fp=0, tn=0, fn=0, time=0, tp_w=0, fn_w=0, fp_w=0):
        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn
        self.time = time
        self.tp_w = tp_w
        self.fn_w = fn_w
        self.fp_w = fp_w

    def get_precision(self):
        return float(self.tp) / float(self.tp + self.fp)

    def get_accuracy(self):
        return float(self.tp + self.tn) / float(self.tp + self.fp + self.fn + self.tn)

    def get_specificity(self):
        return float(self.tn) / float(self.tn + self.fp)

    def get_recall(self):
        return float(self.tp) / float(self.tp + self.fn)

    def get_f1(self):
        return (2 * self.get_precision() * self.get_recall()) / (self.get_precision() + self.get_recall())

    def get_precision_w(self):
        return float(self.tp_w) / float(self.tp_w + self.fp_w)

    def get_accuracy_w(self):
        return float(self.tp_w) / float(self.tp_w + self.fp_w + self.fn_w)

    def get_recall_w(self):
        return float(self.tp_w) / float(self.tp_w + self.fn_w)

    def get_f1_w(self):
        return (2 * self.get_precision_w() * self.get_recall_w()) / (self.get_precision_w() + self.get_recall_w())