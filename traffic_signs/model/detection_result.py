class Result:
    pixel_precision: float
    pixel_accuracy: float
    pixel_specificity: float
    pixel_sensitivity: float
    window_precision: float
    window_accuracy: float

    def __init__(self, pixel_precision=0, pixel_accuracy=0, pixel_specificity=0, pixel_sensitivity=0,
                 window_precision=0, window_accuracy=0):
        self.pixel_precision = pixel_precision
        self.pixel_accuracy = pixel_accuracy
        self.pixel_specificity = pixel_specificity
        self.pixel_sensitivity = pixel_sensitivity
        self.window_precision = window_precision
        self.window_accuracy = window_accuracy
