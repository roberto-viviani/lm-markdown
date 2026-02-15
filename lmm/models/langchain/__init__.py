""" LangChain/LangGraph API interface to language models 

This package has the purpose of connecting specifications written in
a toml file to the implementation of language model applications 
using the LangChain/LangGraph API. This allows non-programmer end
users to configure the application layer without writing or modifying
the API interface directly. The connection takes place through objects
containing configuration parameters, based on the Pydantic Settings
library, that may be created in code or left empty to be loaded from
config.toml.

The LangChain/LangGraph API interface has three layers of abstractions:

- model objects: they wrap calling and receiving messages to and from
    the language model.
- objects implementing the runnable interface (RunnableSerializable):
    these objects can be chained together to form larger entities. The
    runnable as well as the chains offer a unified calling interface
    with the .invoke/.ainvoke member function.
- graphs that define fixed or modifiable flow through nodes. The nodes
    can be regular Python functions, chains, or agents.

"""
