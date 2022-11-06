from ..base_model import BaseModel


class BaseEmbedder(BaseModel):
    """Base class for classifiers"""

    def __init__(self):
        super(BaseEmbedder, self).__init__()
