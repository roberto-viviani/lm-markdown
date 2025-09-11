"""Hash from text utility"""

import hashlib
import base64


def base_hash(input_string: str) -> str:
    """
    Generate human-readable hash to check changes in strings.
    Args: an input string
    Returns: a hash string
    """
    if not input_string:
        return ""

    # Encode the input string to bytes
    encoded_string = input_string.encode('utf-8')

    # Using MD5 for performance, crypto quality not required
    md5_hasher = hashlib.md5()
    md5_hasher.update(encoded_string)

    # Convert to human-readable
    byte_digest_md5 = md5_hasher.digest()
    base64_digest_md5 = base64.b64encode(byte_digest_md5).decode(
        'utf-8'
    )

    return base64_digest_md5[:-2]


import uuid


def generate_uuid(
    text_input: str, namespace_uuid: uuid.UUID = uuid.NAMESPACE_URL
) -> str:
    """
    Generates a UUID Version 5 from a given text string using a
    specified namespace.

    UUID v5 is based on SHA-1 hashing, ensuring that the same text
    input with the same namespace will always produce the same UUID.

    Args:
        text_input (str): The string from which to generate the UUID.
        namespace_uuid (UUID object, optional): The namespace UUID.
                                Defaults to uuid.NAMESPACE_URL.
                                You can use other predefined
                                namespaces (e.g., uuid.NAMESPACE_DNS)
                                or define your own.

    Returns:
        str: The generated UUID v5 as a hyphenated string.
    """
    # We do need to raise exception here, as type check is static
    if not isinstance(text_input, str):  # type: ignore[reportUnnecessaryInstance]
        raise TypeError("Input 'text_input' must be a string.")

    generated_uuid = uuid.uuid5(namespace_uuid, text_input)
    return str(generated_uuid)
