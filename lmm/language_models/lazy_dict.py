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

    This is a more elaborate example, where a whole specification is
    used to create objects and memoize them:

    ```python
    # This defines the supported model sources
    class LanguageModelSource(StrEnum):
        Anthropic = 'Anthropic'
        Gemini = 'Gemini'
        Mistral = 'Mistral'
        OpenAI = 'OpenAI'


    # This defines source + model
    class LanguageModelSpecification(BaseModel):
        source_name: LanguageModelSource
        model_name: str

        # This required to make it hashable
        class Config:
            frozen = True


    # Langchain model type specified here.
    def _create_model_instance(
        model: LanguageModelSpecification,
    ) -> BaseLM[BaseMsg]:
        # Factory function to create Langchain models while checking permissible
        # sources.

        match model.source_name:
            case LanguageModelSource.OpenAI:
                from langchain_openai.chat_models import ChatOpenAI

                return ChatOpenAI(
                    model=model.model_name,
                    temperature=0.1,
                    max_retries=2,
                    use_responses_api=False,
                )
        ... (rest of code not shown)

    # The memoized dictionary
    langchain_factory = LazyLoadingDictFunc(_create_model_instance)

    # Example of use
    model_spec = {'source_name': "OpenAI", 'model_name': "gpt-4o"}
    model = langchain_factory[
        LanguageModelSpecification(**model_spec)
    ]
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
