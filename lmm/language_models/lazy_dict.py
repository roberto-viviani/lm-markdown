"""
The utility class `LazyLoadingDict` stores memoized language model
class objects, or indeed objects of any class, produced by a factory
function.

The `LazyLoadingDict` class has three main uses that may be combined.

- the first is to create objects based on a definition using a
dictionary interface. The key of the dictionary is the definition that
provides the object instance; different instances may be created
based on the definition

- the second is to memoize the objects created by the definition

- the third is to enable runtime errors when an invalid definition is
given.

The class is instantiated by providing the factory function in the
constructor. The factory function takes one argument of the type of
the dictionary key, and returns a type that determined the type of
the values in the dictionary. To trigger runtime errors when invalid
definitions are provided, provide keys of EnumStr of BaseModel-derived
types (for example, see the documentation of the class).
"""

from collections.abc import Callable
from typing import TypeVar

# ValueT is the parameter for the stored valued, KeyT for the keys.
ValueT = TypeVar('ValueT')
KeyT = TypeVar('KeyT')


class LazyLoadingDict(dict[KeyT, ValueT]):
    """A lazy dictionary class with memoized object of type ValueT.
    To restrict the keys used, use a StrEnum key value (see example
    below). Any object type may be used as key, depending on how the
    dictionary is used.

    Example:
    ```python
    # We define here permissible keys by inheriting from StrEnum
    class LMSource(StrEnum):
        Anthropic = 'Anthropic'
        Gemini = 'Gemini'
        OpenAI = 'OpenAI'

    # We then define a factory function that creates a model object
    # designated by the key, i.e. a function that maps the possible
    # keys to instances that are memoized. In the example, ModelClass
    # objects are stored in the dictionary (code not included):
    def create_model_instance(model_name: LMSource) -> ModelClass:
        print(f"Created instance of {model_name}")
        return ModelClass(model_name=model_name)

    # The lazy dictionary is created by giving the factory function
    # in the constructor.
    lazy_dict = LazyLoadingDict(create_model_instance)

    # The objects are created or retrieved as the value of the key:
    openai_model = lazy_dict['OpenAI']

    # If the argument of the factory is derived from StrEnum, calling
    # the dictionary with an invalid key will throw a ValueError:
    model = lazy_dict[LMSource('OpenX')]
    ```

    This is a more elaborate example, where a whole specification is
    used to create objects and memoize them:

    ```python
    # This defines the supported model sources. Runtime errors
    # provided by BaseModel below
    from typing import Literal
    from pydantic import BaseModel, ConfigDict

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

        # This required to make instances hashable, so that they can
        # be used as keys in the dictionary
        model_config = ConfigDict(frozen=True)


    # Langchain model type specified here.
    def _create_model_instance(
        model: LanguageModelSpecification,
    ) -> BaseLM[BaseMsg]:
        # Factory function to create Langchain models while checking
        # permissible sources, provided as key values:

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

    # The memoized dictionary. langchain_factory is parametrized like
    # a dict[LanguageModelSpecification, BaseLM[BaseMSg]]
    langchain_factory = LazyLoadingDict(_create_model_instance)

    # Example of use
    model_spec = {'source_name': "OpenAI", 'model_name': "gpt-4o"}
    model = langchain_factory[
        LanguageModelSpecification(**model_spec)
    ]
    ```

    A Pydantic model class may also be used to create a more flexible
    dictionary. In the previous example, only the models specified in
    LanguageModel source can be specified without raising exceptions.
    However, a Pydantic model class may be used to constrain the
    objects saved in the dictionary without limiting them to a finite
    sets, i.e. by a validation that does not constrain the instances
    to that set. Thus, if source_name was a str in the above example,
    then any LanguageModelSpecification constructed with any string
    will be accepted.

    In the following example, the runtime error is generated in the
    factory function, because literals do not give rise to runtime
    errors in themselves.

    ```python
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

    It is also possible to assign to the dictionary directly, thus
    bypassing the factory function. In this case, the only checks
    are those that are possibly computed by Pydantic when the object
    is assigned.

    Expected behaviour: may raise ValidationError and ValueErrors.
    """

    def __init__(
        self,
        key_creator_func: Callable[[KeyT], ValueT],
        destructor_func: Callable[[ValueT], None] | None = None,
    ):
        super().__init__()
        self._key_creator_func = key_creator_func
        self._destructor_func = destructor_func

    def _destroy_value(self, value: ValueT) -> None:
        """Helper to destroy a value using the configured strategy."""
        if self._destructor_func:
            self._destructor_func(value)
        elif hasattr(value, "close") and callable(value.close): # type: ignore (self-reflection)
            value.close()  # type: ignore (checked)
        elif hasattr(value, "dispose") and callable(value.dispose): # type: ignore (self-reflection)
            value.dispose() # type: ignore (checked)

    def __getitem__(self, key: KeyT) -> ValueT:
        # Check if the value is already cached
        if key in self:
            return super().__getitem__(key)

        # Lazy-load the data, cache it, and return
        value: ValueT = self._key_creator_func(key)
        super().__setitem__(key, value)
        return value

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        """Allow direct setting of key/value pairs.

        This bypasses the factory function for the given key.
        Once set directly, the factory function will not be called
        for this key unless the key is deleted first.
        
        Raises:
            ValueError: If the key already exists in the dictionary.
        """
        if key in self:
            raise ValueError(f"Key '{key}' already exists. Delete it first to overwrite.")
        super().__setitem__(key, value)

    def __delitem__(self, key: KeyT) -> None:
        if key in self:
            value: ValueT = super().__getitem__(key)
            self._destroy_value(value)
        super().__delitem__(key)

    def clear(self) -> None:
        for value in list(self.values()):
            self._destroy_value(value)
        super().clear()

    def __del__(self) -> None:
        # We need to be careful here during interpreter shutdown
        try:
            self.clear()
        except Exception:
            # Suppress errors during destruction to avoid noise
            pass
