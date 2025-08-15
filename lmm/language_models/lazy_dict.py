"""
A utility class to store memoized language model class objects, or
indeed objects of any class, produced by a factory function.

This class has three main uses that may be combined.

- the first is to create objects based on a definition using a
dictionary interface. The key of the dictionary is the definition that
provides the object instance; different instances may be created
based on the definition

- the second is to memoize the objects created by the definition

 the third is to enable runtime errors when an invalid definition is
given.

The class is instantiated by providing the factory function in the
constructor. To trigger runtime errors, provide keys of a EnumStr
of BaseModel-derived types.
"""

from typing import Callable, TypeVar

# We define a TypeVar for value type the dictionary is storing.
ValueT = TypeVar('ValueT')
KeyT = TypeVar('KeyT')


class LazyLoadingDict(dict[KeyT, ValueT]):
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
    # the dictionary (code not included):
    def create_model_instance(model_name: LMSource) -> ModelClass:
        print(f"Created instance of {model_name}")
        return ModelClass(model_name=model_name)


    # The lazy dictionary is created thus
    lazy_dict = LazyLoadingDict(create_model_instance)

    # The objects are retrieved so:
    openai_model = lazy_dict[LMSource("OpenAI")]

    # This will throw a ValueError:
    model = lazy_dict[LMSource("OpenX")]
    ```

    This is a more elaborate example, where a whole specification is
    used to create objects and memoize them:

    ```python
    # This defines the supported model sources. Runtime errors
    # provided by BaseModel below
    LanguageModelSource = Literal[
            'Anthropic',
            'Gemini',
            'Mistral',
            'OpenAI'
        ]

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
    langchain_factory = LazyLoadingDict(_create_model_instance)

    # Example of use
    model_spec = {'source_name': "OpenAI", 'model_name': "gpt-4o"}
    model = langchain_factory[
        LanguageModelSpecification(**model_spec)
    ]
    ```

    In the following example, the runtime error is generated in the
    factory function, because literals do not give rise to runtime
    errors in themselves.

    ModelSource = Literal["OpenAI", "Cohere"]

    def _model_factory(src: ModelSource) -> ModelClass:
        match src:
            case "OpenAI"
                return ModelClass("OpenAI") # code not shown
            case "Cohere"
                return ModelClass("Cohere") # code not shown
            case _:
                # required to raise error
                raise ValueError(f"Invalid model source: {src}")

    model_factory = LazyLoadingDict(_model_factory)
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
