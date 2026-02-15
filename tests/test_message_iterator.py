"""
Test script for the message iterator module.

This script demonstrates and tests the yield_message function and 
MessageIterator class.
"""

from lmm.models.message_iterator import (
    yield_message,
    yield_constant_message,
)


def test_basic_functionality():
    """Test basic message generation functionality."""
    print("=== Testing Basic Functionality ===")

    # Test default prefix
    iterator = yield_message()
    print("Default prefix iterator:")
    for i in range(5):  # type: ignore
        message = next(iterator)
        print(f"  {message}")

    print()

    # Test custom prefix
    custom_iterator = yield_message("Alert")
    print("Custom prefix iterator:")
    for i in range(5):  # type: ignore
        message = next(custom_iterator)
        print(f"  {message}")


def test_multiple_iterators():
    """Test that multiple iterators work independently."""
    print("\n=== Testing Multiple Independent Iterators ===")

    iter1 = yield_message("Task")
    iter2 = yield_message("Event")

    print("Alternating between two iterators:")
    for i in range(3):  # type: ignore
        msg1 = next(iter1)
        msg2 = next(iter2)
        print(f"  Iterator 1: {msg1}")
        print(f"  Iterator 2: {msg2}")


def test_iterator_protocol():
    """Test that the iterator protocol is properly implemented."""
    print("\n=== Testing Iterator Protocol ===")

    iterator = yield_message("Test")

    # Test that the iterator returns itself
    assert (
        iterator.__iter__() is iterator
    ), "Iterator should return itself"
    print("✓ Iterator returns itself from __iter__()")

    # Test that it works with Python's built-in iteration
    messages = []
    for i, message in enumerate(iterator):
        messages.append(message)  # type: ignore
        if i >= 4:  # Stop after 5 messages to avoid infinite loop
            break

    expected = ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5"]
    assert (
        messages == expected
    ), f"Expected {expected}, got {messages}"
    print("✓ Iterator works with built-in iteration")
    print(f"  Generated: {messages}")


def test_constant_message():
    """Test the constant message iterator"""

    iterator = yield_constant_message("Alert")
    msg = next(iterator)
    assert msg == "Alert"
    msg = next(iterator)
    assert msg == "Alert"


if __name__ == "__main__":
    test_basic_functionality()
    test_multiple_iterators()
    test_iterator_protocol()
    print("\n=== All Tests Passed! ===")
