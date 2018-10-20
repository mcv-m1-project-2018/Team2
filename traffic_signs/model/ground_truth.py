from model.rectangle import Rectangle


class GroundTruth(Rectangle):
    type: str

    def __init__(self, type='A', top_left=(0, 0), width=0, height=0):
        super().__init__(top_left, width, height)
        self.type = type
