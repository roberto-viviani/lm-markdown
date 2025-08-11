# flake8: noqa
# pyright: basic

import unittest
import yaml  # PyYAML library

# Attempt to import the functions to be tested
from lmm.markdown.parse_yaml import split_yaml_parse
from lmm.markdown.parse_yaml import desplit_yaml_parse
from lmm.markdown.parse_yaml import serialize_yaml_parse


class TestMarkdownMetadata(unittest.TestCase):
    """Class testing behaviour when library provided with
    yaml objects. Note: in the markdown application, yaml objects
    will be usually provided after parsing a string with
    yaml.safe_load, which will not provide this level of variety"""

    # --- Tests for split_yaml_parse ---

    def test_parse_none_input(self):
        self.assertEqual(split_yaml_parse(None), ({}, []))

    def test_parse_none_inputlist(self):
        self.assertEqual(split_yaml_parse([None]), ({}, []))

    def test_parse_empty_list(self):
        self.assertEqual(split_yaml_parse([]), ({}, []))

    def test_parse_empty_dict(self):
        self.assertEqual(split_yaml_parse({}), ({}, []))

    def test_parse_empty_listeddict(self):
        self.assertEqual(split_yaml_parse([{}]), ({}, []))

    def test_parse_conformant_dict(self):
        data = {"key1": "value1", "key2": 2}
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_conformant_dict_list(self):
        data = {"key1": "value1", "key2": ["first", "second"]}
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_conformant_dict_dict(self):
        data = {"key1": "value1", "key2": {"first": 1, "second": 2}}
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_conformant_dict_dict_list(self):
        data = {
            "key1": "value1",
            "key2": {"first": 1, "second": ["One", "Two"]},
            "key3": ["One", "Two", 3],
        }
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_conformant_dict_dict_dict(self):
        data = {
            "key1": "value1",
            "key2": {"first": 1, "second": {"One": [0, 1, 2]}},
            "key3": ["One", "Two", 3],
        }
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_partially_conformant_dict(self):
        data = {
            "key1": "value1",
            "key2": bytes("data", encoding="utf-8"),
        }
        # expected behaviour: the conformant values are in p1,
        # the others in p2
        expected_p1 = {"key1": "value1"}
        expected_p2 = [{"key2": bytes("data", encoding="utf-8")}]
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)),
            [expected_p1] + expected_p2,
        )

    def test_parse_empty_dict_is_conformant(self):
        data = {}
        self.assertEqual(split_yaml_parse(data), (data, []))

    def test_parse_non_conformant_dict_int_key(self):
        data = {1: "value1", "key2": "value2"}
        self.assertEqual(split_yaml_parse(data), ({}, [data]))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_non_conformant_dict_tuple_key(self):
        data = {("a", "b"): "value1", "key2": "value2"}
        self.assertEqual(split_yaml_parse(data), ({}, [data]))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_list_first_conformant(self):
        data = [{"key1": "val1"}, {"key2": "val2"}, {3: "val3"}]
        expected_p1 = {"key1": "val1"}
        expected_p2 = [{"key2": "val2"}, {3: "val3"}]
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_list_first_partially_conformant(self):
        data = [
            {"key1": "val1", "key2": ["first", "second"]},
            {3: "val3"},
        ]
        # expected behavior: first dict split between p1 and p2
        expected_p1 = {"key1": "val1", "key2": ["first", "second"]}
        expected_p2 = [{3: "val3"}]
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)),
            [expected_p1] + expected_p2,
        )

    def test_parse_list_first_non_conformant(self):
        data = [{1: "val1"}, {"key2": "val2"}]
        expected_p1 = {}
        expected_p2 = data
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_list_single_conformant_item(self):
        data = [{"key1": "val1"}]
        # this is not reconstructed differently from {"key1": "val1"}
        expected_p1 = {"key1": "val1"}
        expected_p2 = []
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), expected_p1
        )

    def test_parse_list_single_non_conformant_item(self):
        data = [{1: "val1"}]
        expected_p1 = {}
        expected_p2 = data
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data[0]
        )

    def test_parse_list_with_non_dict_items(self):
        data = [
            {"key1": "val1"},
            "a string",
            {"key2": "val2"},
            None,
            {3: "val3"},
        ]
        # Expected behavior: the rest of the list is taken over in p2
        expected_p1 = {"key1": "val1"}
        expected_p2 = data[1:]
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_list_with_non_dict_items_first_dict_non_conformant(
        self,
    ):
        data = ["a string", {1: "non_conf"}, {"key2": "val2"}]
        # Expected behavior: everything in p2
        expected_p1 = {}
        expected_p2 = data
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_list_only_non_dict_items(self):
        data = ["a string", 123, None]
        # Expected behavior: everything in p2 list
        self.assertEqual(split_yaml_parse(data), ({}, data))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_scalar_string_input(self):
        data = "Just a string"
        self.assertRaises(ValueError, split_yaml_parse, data)

    def test_parse_scalar_bytes_input(self):
        data = bytes("Encoded text", encoding="utf-8")
        self.assertEqual(split_yaml_parse(data), ({}, [data]))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )

    def test_parse_query_prompt(self):
        data = yaml.safe_load("?: Please summarize")
        self.assertEqual(
            split_yaml_parse(data, {"?": "query"}),
            ({"query": "Please summarize"}, []),
        )

    def test_parse_chat_prompt(self):
        data = yaml.safe_load("+: Please summarize")
        self.assertEqual(
            split_yaml_parse(data, {"+": "message"}),
            ({"message": "Please summarize"}, []),
        )

    def test_parse_edit_prompt(self):
        data = yaml.safe_load("=: Please summarize")
        self.assertEqual(
            split_yaml_parse(data, {"=": "edit"}),
            ({"edit": "Please summarize"}, []),
        )

    def test_mapped_key(self):
        text = "?: Why is the sky blue?\nresp: Rayleigh phenomenon"
        yamldata = yaml.safe_load(text)
        part, whole = split_yaml_parse(yamldata, {"?": "query"})
        self.assertDictEqual(
            part,
            {
                "query": "Why is the sky blue?",
                "resp": "Rayleigh phenomenon",
            },
        )
        self.assertEqual(
            serialize_yaml_parse((part, whole)).strip(),
            "query: Why is the sky blue?\nresp: Rayleigh phenomenon",
        )

    def test_mapped_keys(self):
        text = "?: Why is the sky blue?\n+: Rayleigh phenomenon"
        yamldata = yaml.safe_load(text)
        part, whole = split_yaml_parse(
            yamldata, {"?": "query", "+": "resp"}
        )
        self.assertDictEqual(
            part,
            {
                "query": "Why is the sky blue?",
                "resp": "Rayleigh phenomenon",
            },
        )
        self.assertEqual(
            serialize_yaml_parse((part, whole)).strip(),
            "query: Why is the sky blue?\nresp: Rayleigh phenomenon",
        )

    # --- Tests for desplit_yaml_parse ---

    def test_desplit_part_nowhole(self):
        part = {"This": 1}
        whole = []
        data = part, whole
        self.assertEqual(desplit_yaml_parse(data), part)

    def test_desplit_nopart_complexwhole(self):
        part = {}
        whole: list[object] = [{(0, 1): 1}]
        data = part, whole
        self.assertEqual(desplit_yaml_parse(data), whole[0])

    def test_desplit_part_whole(self):
        part = {"First": 1}
        whole: list[object] = [{"This": 0}]
        data = part, whole
        outcome = [{"First": 1}, {"This": 0}]
        self.assertEqual(desplit_yaml_parse(data), outcome)

    # --- Tests for serialize_yaml_parse (and implicit split_yaml_parse) ---
    def _assert_serialization_logic(
        self,
        original_data,
        expected_reloaded_object,  # type: ignore
    ):
        """Helper to test the parse -> serialize -> reload cycle."""
        parsed_p1, parsed_p2 = split_yaml_parse(original_data)
        yaml_output = serialize_yaml_parse((parsed_p1, parsed_p2))
        reloaded_data = yaml.safe_load(yaml_output)
        self.assertEqual(reloaded_data, expected_reloaded_object)

    def test_dump_yaml_none(self):
        # split_yaml_parse(None) -> ({}, [])
        # serialize_yaml_parse(({}, [])) dumps `None`
        self._assert_serialization_logic(None, None)

    def test_serialize_from_none_input(self):
        # split_yaml_parse(None) -> ({}, [])
        # serialize_yaml_parse(({}, [])) dumps `None``
        self._assert_serialization_logic(None, None)

    def test_serialize_from_conformant_dict(self):
        original = {"key1": "value1", "key2": 2}
        # split_yaml_parse(original) -> (original, [])
        # serialize_yaml_parse((original, [])) dumps `original` (p1)
        self._assert_serialization_logic(original, original)

    def test_serialize_from_empty_dict(self):
        original = {}
        # split_yaml_parse(original) -> (original, [])
        # serialize_yaml_parse((original, [])) dumps `original` as None
        self._assert_serialization_logic(original, None)

    def test_serialize_from_non_conformant_dict(self):
        original = {1: "value1", "key2": "value2"}
        # split_yaml_parse(original) -> ({}, [original])
        # serialize_yaml_parse(({}, [original])) dumps `original` (p2)
        self._assert_serialization_logic(original, original)

    def test_serialize_from_empty_list(self):
        original = []
        # split_yaml_parse(original) -> ({}, [])
        # serialize_yaml_parse(({}, [])) dumps `[]` as None
        self._assert_serialization_logic(original, None)

    def test_serialize_from_list_first_conformant(self):
        original = [{"k1": "v1"}, {"k2": "v2"}]
        # split_yaml_parse(original) -> ({"k1":"v1"}, [{"k2":"v2"}])
        # serialize_yaml_parse(...) dumps `[p1] + p2` which is `original`
        self._assert_serialization_logic(original, original)

    def test_serialize_from_list_single_conformant_item(self):
        original = [{"k1": "v1"}]
        # split_yaml_parse(original) -> ({"k1":"v1"}, [])
        # serialize_yaml_parse(...) dumps `the input
        self._assert_serialization_logic(original, original[0])

    def test_serialize_from_list_first_non_conformant(self):
        original = [{1: "v1"}, {"k2": "v2"}]
        # split_yaml_parse(original) -> ({}, original)
        # serialize_yaml_parse(...) dumps `p2` which is `original`
        self._assert_serialization_logic(original, original)

    def test_serialize_from_list_single_non_conformant_item(self):
        original = [{1: "v1"}]
        # split_yaml_parse(original) -> ({}, original)
        # serialize_yaml_parse(...) dumps `p2` which is `original[0]`
        self._assert_serialization_logic(original, original[0])

    def test_serialize_list_with_non_dict_items_filter_out(self):
        original = [{"key1": "val1"}, "a string", {"key2": "val2"}]
        # non conformant elements of list, but first is conformant
        expected_reloaded = original
        self._assert_serialization_logic(original, expected_reloaded)

    def test_serialize_list_with_non_dict_items_first_dict_non_conformant(
        self,
    ):
        original = ["str", {1: "non_conf"}, {"k2": "v2"}]
        # split_yaml_parse puts everything in whole
        expected_reloaded = ["str", {1: "non_conf"}, {"k2": "v2"}]
        self._assert_serialization_logic(original, expected_reloaded)

    def test_serialize_list_only_non_dict_items(self):
        original = ["a string", 123, None]
        expected_reloaded = original
        self._assert_serialization_logic(original, expected_reloaded)


class TestMarkdownMetadataText(unittest.TestCase):
    """this tests beahviour from strings parsed by yaml.safe_load.
    This function tends to treat many things as strings, so some
    behaviour of the previous tests cannot be reproduced."""

    def test_empty(self):
        yamldata = yaml.safe_load("")
        part, whole = split_yaml_parse(yamldata)
        self.assertEqual(serialize_yaml_parse((part, whole)), "")

    def test_parse_conformant_dict(self):
        text = "key1: value1\nkey2: 2"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        self.assertEqual(
            serialize_yaml_parse(split_yaml_parse(data)).strip(), text
        )

    def test_parse_conformant_dict_list(self):
        text = "key1: value1\nkey2: [first, second]"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertDictEqual(yaml.safe_load(newtext), data)

    def test_parse_conformant_dict_dict(self):
        text = "key1: value1\nkey2: {first: 1, second: 2}"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertDictEqual(yaml.safe_load(newtext), data)

    def test_parse_conformant_dict_dict_list(self):
        text = "key1: value1\nkey2: {first: 1, second: [One, Two]}\nkey3: [One, Two, 3]"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertDictEqual(yaml.safe_load(newtext), data)

    def test_parse_conformant_dict_dict_dict(self):
        text = "key1: value1\nkey2: {first: 1, second: {One: [0, 1, 2]}}\nkey3: [One, Two, 3]"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertDictEqual(yaml.safe_load(newtext), data)

    def test_parse_non_conformant_dict_int_key(self):
        # Note: this DOES not end up in the conformant dict
        text = "1: value1\nkey2: value2"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), ({}, [data]))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertDictEqual(yaml.safe_load(newtext), data)

    def test_parse_non_conformant_dict_tuple_key(self):
        # Note: this is accepted as the tuple is parsed as a string
        text = "('a', 'b'): value1\nkey2: value2"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertDictEqual(yaml.safe_load(newtext), data)

    def test_parse_non_conformant_dict_tuple_intkey(self):
        # Note: this is accepted as the tuple is parsed as a string
        text = "(1, 2): value1\nkey2: value2"
        data = yaml.safe_load(text)
        self.assertEqual(split_yaml_parse(data), (data, []))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertDictEqual(yaml.safe_load(newtext), data)

    def test_parse_list_first_conformant(self):
        # data = [{"key1": "val1"}, {"key2": "val2"}, {3: "val3"}]
        text = "- key1: val1\n- key2: val2\n- 3: val3"
        data = yaml.safe_load(text)
        expected_p1 = {"key1": "val1"}
        expected_p2 = [{"key2": "val2"}, {3: "val3"}]
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertListEqual(yaml.safe_load(newtext), data)

    def test_parse_list_first_partially_conformant(self):
        # data = [
        #     {"key1": "val1", "key2": ["first", "second"]},
        #     {3: "val3"},
        # ]
        text = "- key1: val1\n  key2: [first, second]\n- 3: val3"
        data = yaml.safe_load(text)
        # expected behavior: first dict split between p1 and p2
        expected_p1 = {"key1": "val1", "key2": ["first", "second"]}
        expected_p2 = [{3: "val3"}]
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)),
            [expected_p1] + expected_p2,
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertListEqual(yaml.safe_load(newtext), data)

    def test_parse_list_first_non_conformant(self):
        # data = [{1: "val1"}, {"key2": "val2"}]
        text = "- 1: val1\n- key2: val2"
        data = yaml.safe_load(text)
        expected_p1 = {}
        expected_p2 = data
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertListEqual(yaml.safe_load(newtext), data)

    def test_parse_list_single_conformant_item(self):
        # data = [{"key1": "val1"}]
        # this is not reconstructed differently from {"key1": "val1"}
        text = "- key1: val1"
        data = yaml.safe_load(text)
        expected_p1 = {"key1": "val1"}
        expected_p2 = []
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), expected_p1
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        # note the list is shed here
        self.assertDictEqual(yaml.safe_load(newtext), data[0])

    def test_parse_list_single_non_conformant_item(self):
        # data = [{1: "val1"}]
        text = "- 1: val1"
        data = yaml.safe_load(text)
        expected_p1 = {}
        expected_p2 = data
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data[0]
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        # note the list is shed here
        self.assertDictEqual(yaml.safe_load(newtext), data[0])

    def test_parse_list_with_non_dict_items(self):
        # data = [
        #     {"key1": "val1"},
        #     "a string",
        #     {"key2": "val2"},
        #     None,
        #     {3: "val3"},
        # ]
        text = "- key1: val1\n- a string\n- key2_ val2\n- None\n- 3: val3"
        data = yaml.safe_load(text)
        # Expected behavior: the rest of the list is taken over in p2
        expected_p1 = {"key1": "val1"}
        expected_p2 = data[1:]
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertListEqual(yaml.safe_load(newtext), data)

    def test_parse_list_with_non_dict_items_first_dict_non_conformant(
        self,
    ):
        # data = ["a string", {1: "non_conf"}, {"key2": "val2"}]
        text = "- a string\n- 1: non_conf\n- key2: val2"
        data = yaml.safe_load(text)
        # Expected behavior: everything in p2
        expected_p1 = {}
        expected_p2 = data
        self.assertEqual(
            split_yaml_parse(data), (expected_p1, expected_p2)
        )
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertListEqual(yaml.safe_load(newtext), data)

    def test_parse_list_only_non_dict_items(self):
        # data = ["a string", 123, None]
        text = "- a string\n- 123\n- None"
        data = yaml.safe_load(text)
        # Expected behavior: everything in p2 list
        self.assertEqual(split_yaml_parse(data), ({}, data))
        self.assertEqual(
            desplit_yaml_parse(split_yaml_parse(data)), data
        )
        newtext = serialize_yaml_parse(split_yaml_parse(data))
        self.assertListEqual(yaml.safe_load(newtext), data)

    def test_parse_scalar_string_input(self):
        # data = "Just a string"
        text = "Just a string"
        data = yaml.safe_load(text)
        self.assertRaises(ValueError, split_yaml_parse, data)


if __name__ == "__main__":
    unittest.main(
        argv=["first-arg-is-ignored"], exit=False
    )  # Using exit=False for environments like notebooks
