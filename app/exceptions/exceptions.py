"""Module for organizing all application errors"""


class UnsupportedContentTypeError(Exception):
    """
    Exception raised when an unsupported content type is sent in the request.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message='Content type is not supported!'):
        self.message = message
        super().__init__(self.message)


class UserNotFoundError(Exception):
    """
    Exception raised when the requested user is not found in the system.

    Attributes:
        message (str): Explanation of the error.
    """

    def __init__(self, message='User not found'):
        self.message = message
        super().__init__(self.message)
