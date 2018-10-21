from model.rectangle import Rectangle


class GroundTruth(Rectangle):
    type: str

    def __init__(self, sign_type='A', top_left=(0, 0), width=0, height=0):
        super().__init__(top_left, width, height)
        self.type = sign_type

    def to_csv(self):
        return super().to_csv() + ' ' + self.type
