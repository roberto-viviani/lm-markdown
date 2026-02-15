# Language model kernels

This package implements the interface to the language models. It consists of three layers:

- the basic layer handles the language models themselves, abstracting from the details of individual vendors. This is implemented through the Langchain framework in the module `.langchain.models`. The primary function of this layer is to liaise between creation of Langchain objects encapsulating the exchange with the language model API and a standard specification that may be read from a config.toml file, automating user-led selection of models.

- an intermediate layer formulates and uses specifications of language kernels. The kernels are specific uses of the language models. For example, to implement a chat, one will specify a system and a human prompt to obtain a chat kernel. These specifications are handled by the `tools` module. The aim of this module is to provide a library through which simple configurations of language models may be created, supporting variations without the need to specify them at the level of the Langchain framework.

- an implementation layer creates the kernel themselves. This is implemented again through the langchain framework in the `.langchain.runnables` module, which uses `BaseChatModel` as a key element to produce objects callable via `.invoke`. `BaseChatModel` supports tool calling and structured output, thus supporting further customization. The Langchain kernels are called 'runnables' in the implementation.

The boundary between the layers is not rigid. For example, it is possible to interact with the language model directly using basic models. Importantly, Langchain provides its own approach to building complex kernels, through "LCEL objects" built on top of `Runnable`. Therefore, kernels may also be implemented on top of Langchain's runnables.

The `tools` module contains a number of prespecified tools, enabling the creation of a Langchain kernel (a runnable) on the fly.

```python
from lmm.language_models.langchain.runnables import create_runnable
summarizer = create_runnable("summarizer")
```

Here, `summarizer` includes a language model as specified in config.toml and a prompt to summarize text:

```python
try:
    summary = summarizer.invoke(
        {'text': "This is some long text, reduced to a few lines here for the sake of the example"}
    )
except Exception:
    print("Could not retrieve response from model")
```


### Basic models (Langchain)
::: lmm.models.langchain.models

### Primpts
::: lmm.models.prompts

### Simple packaged models (Langchain runnables)
::: lmm.models.langchain.runnables


