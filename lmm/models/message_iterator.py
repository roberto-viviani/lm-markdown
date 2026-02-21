"""
Message Iterator Module

This module provides functionality to create iterators that generate
sequential messages. Used to feed message through a fake language model.
"""

from collections.abc import Iterator


class MessageIterator:
    """
    An iterator that generates sequential messages with a customizable prefix.

    The iterator is infinite and will continue generating messages
    indefinitely. Messages follow the pattern: "{prefix} {counter}" where
    counter starts at 1.
    """

    def __init__(self, prefix: str = "Message") -> None:
        """
        Initialize the MessageIterator.

        Args:
            prefix: The prefix to use for generated messages. Defaults to
                "Message".
        """
        self.prefix = prefix
        self.counter = 1

    def __iter__(self) -> Iterator[str]:
        """Return the iterator object itself."""
        return self

    def __next__(self) -> str:
        """
        Generate the next message in the sequence.

        Returns:
            A string in the format "{prefix} {counter}"
        """
        message = f"{self.prefix} {self.counter}"
        self.counter += 1
        return message


def yield_message(prefix: str = "Message") -> MessageIterator:
    """
    Create and return a MessageIterator instance.

    This function creates an iterator that generates sequential messages
    with the specified prefix. The iterator is infinite and will continue
    generating messages indefinitely.

    Args:
        prefix: The prefix to use for generated messages. Defaults to "Message"

    Returns:
        A MessageIterator instance that generates messages like:
        "Message 1", "Message 2", "Message 3", etc.

    Example:
        >>> iterator = yield_message()
        >>> next(iterator)
        'Message 1'
        >>> next(iterator)
        'Message 2'

        >>> custom_iterator = yield_message("Alert")
        >>> next(custom_iterator)
        'Alert 1'
        >>> next(custom_iterator)
        'Alert 2'
    """
    return MessageIterator(prefix)


class ConstantMessageIterator:
    """
    An iterator that generates the same message with which is was intialized.

    The iterator is infinite and will continue generating messages
    indefinitely.
    """

    def __init__(self, message: str = "Message") -> None:
        """
        Initialize the MessageIterator.

        Args:
            message: The message to return in generated messages. Defaults to
                "Message".
        """
        self.message = message
        self.counter = 0

    def __iter__(self) -> Iterator[str]:
        """Return the iterator object itself."""
        return self

    def __next__(self) -> str:
        """
        Generate the next message in the sequence.

        Returns:
            The string message"
        """
        self.counter += 1
        return self.message


def yield_constant_message(
    message: str = "Message",
) -> ConstantMessageIterator:
    """
    Create and return a ConstantMessageIterator instance.

    This function creates an iterator that generates the same
    message repeatedly. The iterator is infinite and will continue
    generating the message indefinitely.

    Args:
        message: The message to generate. Defaults to "Message".

    Returns:
        A ConstantMessageIterator instance.

    Example:
        >>> iterator = constant_message()
        >>> next(iterator)
        'Message'
        >>> next(iterator)
        'Message'

        >>> custom_iterator = constant_message("Alert")
        >>> next(custom_iterator)
        'Alert'
        >>> next(custom_iterator)
        'Alert'
    """
    return ConstantMessageIterator(message)
