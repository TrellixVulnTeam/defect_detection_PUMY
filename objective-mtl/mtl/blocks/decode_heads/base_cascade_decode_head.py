from abc import ABCMeta, abstractmethod

from .base_decode_head import BaseDecodeHead


class BaseCascadeDecodeHead(BaseDecodeHead, metaclass=ABCMeta):
    """Base class for cascade decode head used in
    :class:`CascadeEncoderDecoder."""

    def __init__(self, *args, **kwargs):
        super(BaseCascadeDecodeHead, self).__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs, prev_output):
        """Placeholder of forward function."""
        pass
