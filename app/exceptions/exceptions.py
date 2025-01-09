"""Module for organizing all application errors"""


class UnsupportedContentTypeError(Exception):
    """Exception class for when an unknown content type is sent"""


class UserNotFoundError(Exception):
    """Exception class for when API is unable to find user"""
