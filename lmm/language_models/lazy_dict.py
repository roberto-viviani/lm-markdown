"""
A utility class to store memoized language model class objects, or
indeed objects of any class.
"""

from typing import Callable, TypeVar

# We define a TypeVar for value type the dictionary is storing.
ValueT = TypeVar('ValueT')


class LazyLoadingDict(dict[object, ValueT]):
    """A lazy dictionary class with memoized object of type ValueT.
    To restrict the keys used, use a StrEnum key value (see example
    below). Any object type may be used as key, depending on how the
    dictionary is used.

    Example:
    ```python
    # We define here permissible keys in the lazy dict, with base StrEnum
    class LMSource(StrEnum):
        Anthropic = 'Anthropic'
        Gemini = 'Gemini'
        OpenAI = 'OpenAI'


    # A factory function that creates a model class using the permissible key
    # as an info for its creation. ModelClass is the concrete type stored in
    # the dictionary.
    def create_model_instance(model_name: LMSource) -> ModelClass:
        print(f"Created instance of {model_name}")
        return ModelClass(model_name=model_name)


    # A dictionary with factory functions. Keys are of StrEnum type.
    model_creators: dict[StrEnum, Callable[[], ModelClass]] = {
        LMSource.OpenAI: lambda: create_model_instance(LMSource.OpenAI),
        LMSource.Anthropic: lambda: create_model_instance(
            LMSource.Anthropic
        ),
        LMSource.Gemini: lambda: create_model_instance(LMSource.Gemini),
    }

    # The lazy dictionary is created thus
    lazy_dict = LazyLoadingDict(model_creators)

    # The objects are retrieved so:
    openai_model = lazy_dict[LMSource("OpenAI")]

    # This will throw a ValueError:
    model = lazy_dict[LMSource("OpenX")]

    ```
    """

    def __init__(
        self,
        key_creator_map: dict[
            object,
            Callable[[], ValueT],
        ],
    ):
        super().__init__()
        self._key_creator_map = key_creator_map

    def __getitem__(self, key: object) -> ValueT:
        # Check if the value is already cached
        if key in self:
            return super().__getitem__(key)

        # Lazy-load the data, cache it, and return
        if key in self._key_creator_map:
            value: ValueT = self._key_creator_map[key]()
            self[key] = value
            return value

        # This part should not be reached if StrEnum used
        raise ValueError(
            f"Key '{key}' not among allowed language models."
        )


KeyT = TypeVar('KeyT')


class LazyLoadingDictFunc(dict[KeyT, ValueT]):
    """A lazy dictionary class with memoized object of type ValueT.
    To restrict the keys used, use a StrEnum key value (see example
    below). Any object type may be used as key, depending on how the
    dictionary is used.

    Example:
    ```python
    # We define here permissible keys in the lazy dict, with base StrEnum
    class LMSource(StrEnum):
        Anthropic = 'Anthropic'
        Gemini = 'Gemini'
        OpenAI = 'OpenAI'


    # A factory function that creates a model class using the permissible key
    # as an info for its creation. ModelClass is the concrete type stored in
    # the dictionary.
    def create_model_instance(model_name: LMSource) -> ModelClass:
        print(f"Created instance of {model_name}")
        return ModelClass(model_name=model_name)


    # The lazy dictionary is created thus
    lazy_dict = LazyLoadingDictFunc(create_model_instance)

    # The objects are retrieved so:
    openai_model = lazy_dict[LMSource("OpenAI")]

    # This will throw a ValueError:
    model = lazy_dict[LMSource("OpenX")]

    ```
    """

    def __init__(
        self,
        key_creator_func: Callable[[KeyT], ValueT],
    ):
        super().__init__()
        self._key_creator_func = key_creator_func

    def __getitem__(self, key: KeyT) -> ValueT:
        # Check if the value is already cached
        if key in self:
            return super().__getitem__(key)

        # Lazy-load the data, cache it, and return
        value: ValueT = self._key_creator_func(key)
        self[key] = value
        return value
