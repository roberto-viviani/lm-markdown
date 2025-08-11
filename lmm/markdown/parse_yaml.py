"""
Interface to the pyyaml package.

Layer of functions to work with the output of safe_load to handle
list of dictionaries and cover edge cases. YAML can contain a lot of
content kinds that are not compatible for use with a vector database,
and are not relevant in their use to interact with a language model.
The aim here is to isolate an object that is represented in python as
a dictionary with string keys. This dictionary will be used to
exchange messages with the language model.

Conformant YAML objects consist of dictionaries, or list of
dictionaries of type dict[str, elementary_type] where elementary type
is one of int, float, bool, str.

This module defines types MetadataDict and MetadataValue, which are
union types defining the set of dictionaries and dictionary values
that deemed conformant with the use of LM markdown.

The YAML object contained in a metadata block is decomposed into two,
'part', and 'whole'. The 'part' component is the one that may be used
in the rest of the application, containing a conformant dictionary.
The whole part is kept aside and recomposed with the part when the
whole YAML object is reconstituted.

YAML objects consisting of literals only will raise an exception,
since it is conceivable that the user intended something different.
Byte/imaginary literals are put in whole.
"""

# note: unknown types introduced from pyyaml
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false

from typing import Any, cast, Mapping
import yaml
import re

# TODO: move function using these to scan
# from lmm.scan.scan_keys import QUERY_KEY, MESSAGE_KEY, EDIT_KEY

# The parsed yaml object. ParsedYaml is a tuple with the first
# member being a dictionary with which we can work, and the
# rest a list of things we cannot. The MetadataValue restricts the
# values in the dictionary. We allow only one level of recursion in
# the type definition, because , but defining recursive types is a
# challenge with Pydantic and python versions
MetadataValue = (
    str
    | int
    | bool
    | float
    | list[str | int | bool | float]
    | dict[
        str, str | int | bool | float | list[str | int | bool | float]
    ]
)
MetadataDict = dict[str, MetadataValue]
ParsedYaml = tuple[dict[str, MetadataValue], list[object]]


def _is_metadata_type(value: object) -> bool:
    """Alas, required to match on MetadataValue. This is completely
    recursive, unlike the MetadataValue type. Also lists end up
    being nestable."""
    match value:
        case str() | int() | bool() | float():
            return True
        case list():
            return all([_is_metadata_type(x) for x in value])
        case dict():
            return is_metadata_dict(value)
        case _:
            return False


def _is_primitive_type(value: object) -> bool:
    return isinstance(value, (int, float, str, bool, complex, bytes))


def _is_string_dict(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    if not all([isinstance(k, str) for k in data.keys()]):
        return False
    return True


def _split_metadata_dict(
    values: dict[str, object],
) -> tuple[MetadataDict, list[object]]:
    """Eliminate all values in yaml header that are not in
    the conformant value set"""
    newdict: MetadataDict = {}
    buff = {}
    for v in values.keys():
        if _is_metadata_type(values[v]):
            newdict[v] = values[v]  # type: ignore
        else:
            buff[v] = values[v]
    return (newdict, [buff]) if buff else (newdict, [])


def is_metadata_dict(data: object) -> bool:
    if not _is_string_dict(data):
        return False
    # data is now of type dict[str, ...]
    data = cast(dict[str, object], data)
    return all([_is_metadata_type(value) for value in data.values()])


def split_yaml_parse(
    yamldata: object | None,
    mapped_keys: Mapping[str, str] | None = None,
) -> ParsedYaml:
    """
    Constrain output of parsed yaml objects to a tuple that
    represents a conformant ParsedYaml type, and the original
    object

    Args:
        yamldata: the output of yaml.safe_load()
        mapped_keys: a dict-type to replace keys in the parsed
            yaml object

    Returns:
        a tuple. In the first member of the tuple a conformant
        dictionary with strings as keys and values of conformant
        types. The second member of the tuple is a list of yaml
        data that could not be parsed.
    """

    part: MetadataDict = {}
    whole: list[object] = []
    match yamldata:
        case None | [] | [None]:
            pass
        case list() as value if value == [{}]:
            pass
        case list() as value if value == [[]]:
            pass
        case list() if is_metadata_dict(yamldata[0]):
            # set reference to chosen element of the list
            part = yamldata[0]
            if len(yamldata) > 1:
                whole = yamldata[1:]
        case list() if _is_string_dict(yamldata[0]):
            # heterogeneous dict in first position
            part, buff = _split_metadata_dict(yamldata[0])
            whole = (buff + yamldata[1:]) if len(yamldata) > 1 else buff
        case list():
            # invalid dictionary in first element or list of non-dict
            whole = yamldata
        case dict() if is_metadata_dict(yamldata):
            # we keep whole to empty, as there is no list
            part = yamldata
        case dict() if _is_string_dict(yamldata):
            # heterogeneous dict
            part, whole = _split_metadata_dict(yamldata)
        case dict():
            # invalid dict, keep empty dictionary in part
            whole = [yamldata]
        case _ as lit if _is_metadata_type(lit):
            # someone is specifying data as a literal
            raise ValueError(
                "Data in markdown header must follow a property.\n"
                + "Specify the data like this:\n"
                + f"property_name: {lit}"
            )
        case _ as prim if _is_primitive_type(prim):
            whole = [prim]
        case _:
            # non-dictionary
            raise ValueError(
                "Invalid YAML object type for markdown header (not"
                + " a dict or list)"
            )

    if mapped_keys is not None and bool(part):
        # need a copy of the keys
        keys: list[str] = [k for k in part.keys()]  # type: ignore
        for key in keys:
            for new_key in mapped_keys:
                if key == new_key:
                    # since part is a MetadataDict, its values are
                    # MetadataValue's. pyright does not agree,
                    # however. It is difficult to see how this could
                    # fail, given that we use the same value as before
                    # but with a new key.
                    val: MetadataValue = part.pop(key)  # type: ignore
                    part[mapped_keys[new_key]] = val
                    break

    return part, whole


def desplit_yaml_parse(
    split_parse: (
        tuple[Mapping[str, MetadataValue], list[object]] | None
    ),
) -> Any:
    """
    Reconstitute the original yaml object from the tuple
    constructed by yaml_parse. Dictionaries that were split
    as some values were not elementary remain split.
    """
    if split_parse is None:
        return None
    part, whole = split_parse
    if part == {} and whole == []:
        return None
    if not whole:
        return part
    if part == {}:
        if len(whole) == 1:
            return whole[0]
        else:
            return whole
    else:
        return [part] + whole


def serialize_yaml_parse(
    split_parse: (
        tuple[Mapping[str, MetadataValue], list[object]] | None
    ),
) -> str:
    """
    Reconstitute a yaml string from the tuple
    constructed by yaml_parse. Dictionaries that were split
    as some values were not elementary remain split.
    """
    yamldata = desplit_yaml_parse(split_parse)
    return dump_yaml(yamldata)


def dump_yaml(x: Any) -> str:
    if x is None:
        return ""

    y: str = (
        yaml.safe_dump(
            x,
            default_flow_style=False,
            width=float("Inf"),
            encoding="utf-8",
            allow_unicode=True,
            indent=1,
        )
        .decode("utf-8")
        .replace("'''", "'")
        .replace("__NEWLINE__", "\n")
    )
    return re.sub(r"\n\.\.\.\n$", "", y)
