"""
Interface to the pyyaml package.

Layer of functions to work with the output of safe_load to handle
list of dictionaries and cover edge cases. YAML can contain a lot of
content kinds that are not compatible for use with a vector database,
and are not relevant in their use to interact with a language model.
The aim here is to isolate an object that is represented in python as
a dictionary with string keys. This dictionary will be used to
exchange messages with the language model.

The YAML object contained in a metadata block is decomposed into two,
'part', and 'whole'. The 'part' component is the one that may be used
in the rest of the application. The whole part is kept aside and 
recomposed with the part when the whole YAML object is reconstituted.

Conformant YAML objects consist of dictionaries, or list of 
dictionaries of type dict[str, elementary_type] where elementary type
is one of int, float, bool, str. All other objects will be put in
whole.

YAML objects consisting of literals only will raise an exception, 
since it is conceivable that the user inteded something different.
Byte/imaginary literals are put in whole.
"""

# note: unknown types introduced from pyyaml

# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingTypeArgument=false

from typing import Any
import yaml
import re

# TODO: move function using these to scan
# from lmm.scan.scan_keys import QUERY_KEY, MESSAGE_KEY, EDIT_KEY

# Conformant input
MetadataValue = str | int | bool | float
MetadataDict = dict[str, MetadataValue]
ConformantYaml = MetadataDict | list[MetadataDict]
ParsedYaml = tuple[MetadataDict, list[dict]]


def _is_metadata_type(value: object) -> bool:
    """Alas, required to match on MetadataValue"""
    match value:
        case str() | int() | bool() | float():
            return True
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


def _is_metadata_dict(data: object) -> bool:
    if not _is_string_dict(data):
        return False
    # data is now of type dict[str, ...]
    return all(
        [_is_metadata_type(value) 
         for value 
         in data.values()]  # type: ignore
    )


def _split_metadata_dict(
    values: dict[str, Any],
) -> tuple[MetadataDict, list]:
    """Eliminate all values in yaml header that are not in
    the conformant value set"""
    newdict = {}
    bin = {}
    for v in values.keys():
        if _is_metadata_type(values[v]):
            newdict[v] = values[v]
        else:
            bin[v] = values[v]
    return (newdict, [bin]) if bin else (newdict, [])


def split_yaml_parse(yamldata: Any | None) -> ParsedYaml:
    """
    Constrain output of parsed yaml objects to a tuple that
    represents a conformant ParsedYaml type, and the original
    object

    Args:
        the output of yaml.safe_load()

    Returns:
        a tuple. In the first member of the tuple a conformant
        dictionary with strings as keys and values of primitive
        types. The second member of the tuple is a list of 
        yaml data.
    """

    part: MetadataDict = {}
    whole: list[dict] = []
    match yamldata:
        case None | [] | [None]:
            pass
        case list() as value if value == [{}]:
            pass
        case list() as value if value == [[]]:
            pass
        case list() if _is_metadata_dict(yamldata[0]):
            # set reference to chosen element of the list
            part = yamldata[0]
            if len(yamldata) > 1:
                whole = yamldata[1:]
        case list() if _is_string_dict(yamldata[0]):
            # heterogeneous dict in first position
            part, bin = _split_metadata_dict(yamldata[0])
            whole = bin + yamldata[1:] if len(yamldata) > 1 else bin
        case list():
            # invalid dictionary in first element or list of non-dict
            whole = yamldata
        case dict() if _is_metadata_dict(yamldata):
            # we keep whole to empty, as there is no list
            part = yamldata
        case dict() if _is_string_dict(yamldata):
            # heterogeneous dict
            part, whole = _split_metadata_dict(yamldata)
        case dict():
            # invalid dict, keep empty dictionary in part
            whole = [yamldata]
        case _ as value if _is_metadata_type(value):
            # someone is specifying data as a literal
            raise ValueError(
                "Data in markdown header must "
                + "follow a property.\n"
                + "Specify the data like this:\n"
                + f"property_name: {value}"
            )
        case _ as value if _is_primitive_type(value):
            whole = [value]
        case _:
            # non-dictionary
            raise ValueError(
                "Invalid YAML object for markdown header"
            )

    # # replace shortcuts for language model interactions
    # if isinstance(part, dict):
    #     keys = [k for k in part.keys()]  # copy
    #     for key in keys:
    #         if key == "?":
    #             part[QUERY_KEY] = part.pop(key)
    #         elif key == "+":
    #             part[MESSAGE_KEY] = part.pop(key)
    #         elif key == "=":
    #             part[EDIT_KEY] = part.pop(key)
    #         else:
    #             pass

    return part, whole


def desplit_yaml_parse(
    split_parse: ParsedYaml | None,
) -> ConformantYaml:
    """
    Reconstitute the original yaml object from the tuple
    constructed by yaml_parse. Dictionaries that were splitted
    as some values were not elementary remain splitted.
    """
    if split_parse is None:
        return {}
    part, whole = split_parse
    if whole == []:
        return part
    if part == {}:
        if len(whole) == 1:
            return whole[0]
        else:
            return whole
    else:
        return [part] + whole


def dump_yaml(x: Any) -> str:
    x = yaml.safe_dump(
        x,
        default_flow_style=False,
        width=float("Inf"),
        encoding="utf-8",
        allow_unicode=True,
        indent=1,
    )
    x = x.decode("utf-8")
    x = x.replace("'''", "'").replace("__NEWLINE__", "\n")
    return re.sub(r"\n\.\.\.\n$", "", x)
