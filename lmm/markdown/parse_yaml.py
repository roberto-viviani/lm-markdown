"""
Process output of yaml.safe_load.

Layer of function to interface with the output of safe_load to handle
list of dictionaries and cover edge cases. YAML can contain a lot of
content kinds that are not compatible for use with a vector database,
and are not relevant in their use to interact with a language model.
The target is to isolate an object that is represented in python as
a dictionary with string keys. This dictionary will be used to
exchange messages with the language model.

The YAML object contained in a metadata block is decomposed into two,
'part', and 'whole'. The 'part' component is the one that may be used
in the rest of the application.

This means:
- YAML objects consisting of literals only will return a None part
- YAML objects consisting of a list will be required list jasonable
    objects, with the first being used for the interaction
- the jsonale objects will be objects with strings as keys, otherwise
    the part will be none.
"""
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportMissingTypeArgument=false

# note: unknown types introduced from pyyaml

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


def _validate_dict(values: dict[str, Any]) -> tuple[MetadataDict, list]:
    """Eliminate all values in yaml header that are not in
    the conformant value set"""
    newdict = {}
    bin = {}
    for v in values.keys():
        if isinstance(v, str) and isinstance(values[v], MetadataValue):
            newdict[v] = values[v]
        else:
            bin[v] = values[v]
    return (newdict, [bin]) if bin else (newdict, [])


def dump_yaml(x: Any) -> str:
    x = yaml.safe_dump(
        x,
        default_flow_style=False,
        width=float('Inf'),
        encoding='utf-8',
        allow_unicode=True,
        indent=1,
    )
    x = x.decode("utf-8")
    x = x.replace("'''", "'").replace("__NEWLINE__", "\n")
    return re.sub(r"\n\.\.\.\n$", "", x)


def split_yaml_parse(yamldata: Any | None) -> ParsedYaml:
    """
    Constrain output of parsed yaml objects to a tuple that
    represents a conformant ParsedYaml type, and the original
    object

    Args:
        the output of yaml.safe_load()

    Returns:
        a tuple. In the first member of the tuple a dictionary
        with strings as keys, or empty if the parsed yaml object
        is not a dictionary with these properties. When the
        parsed yaml object is a list, returns the first element
        of the list in the first tuple element if this element is
        a dictionary with these characterisics.
        The second element of the tuple is dictionaries with
        other keys or the rest of dictionary lists.
        Non-dictionaries are purged.

    Guarantees: pure function
    """

    part: MetadataDict = {}
    whole: list[dict] = []
    match yamldata:
        # case list() if all(
        #     [not isinstance(x, dict) for x in yamldata]
        # ):
        #     # a list of literal s, put in whole
        #     whole = yamldata
        case list() if isinstance(yamldata[0], MetadataDict):
            # set reference to chosen element of the list
            part = yamldata[0]
            if len(yamldata) > 1:
                whole = yamldata[1:]
        case list() if isinstance(yamldata[0], dict):
            if all([isinstance(k, str) for k in yamldata[0].keys()]):
                # heterogeneous dict
                part, bin = _validate_dict(yamldata)
                whole = bin + yamldata[1:] if len(yamldata) > 1 else bin
            else:
                whole = yamldata
        case list():
            # invalid dictionary in first element or list of non-dict
            whole = yamldata
        case MetadataDict():
            # we keep whole to empty, as there is no list
            part = yamldata
        case dict() if all([isinstance(k, str) for k in yamldata.keys()]):
            # heterogeneous dict
            part, whole = _validate_dict(yamldata)
        case dict():
            # invalid dict, keep empty dictionary in part
            whole = [yamldata]
        case MetadataValue() as value:
            # someone is specifying data as a literal
            raise ValueError("Data in markdown header must " +
                             "follow a property.\n" +
                             "Specify the data like this:\n" +
                             f"property_name: {value}")
        case _ if bool(yamldata) == False:
            # empty
            pass
        case _: 
            # non-dictionary (probably literal)
            raise ValueError("Invalid YAML object for markdown header")

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
    Reconstitute the original object constructed by yaml_parse, with
    some clean-up.
    """
    empty_dict: bool = split_parse[0] == {} if split_parse else False
    match split_parse:
        case (part, [] | [{}]):
            return part
        # case ({}, whole) if (
        #     empty_dict
        #     and len(whole) == 1
        #     and not all([isinstance(x, str) for x in whole[0].keys()])
        # ):
        #     # If whole has only one string-keyed dict element and
        #     # part is empty, return the single element
        #     return whole[0]
        # case ({}, whole) if empty_dict:  # empty_dict seems be needed
        #     return whole
        case ({}, whole) if len(whole) == 1:
            return whole[0]
        case ({}, whole):
            return whole
        case (part, whole):
            # When part is non-empty, add it to whole
            return [part] + whole
        case _:
            raise RuntimeError("Unreachable code reached due to" + 
                               " unrecognized part/whole objects")
