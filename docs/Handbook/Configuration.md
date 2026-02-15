# Configuration

The application may be configured by editing the settings in `Config.toml`. The main settings are:

- embeddings: the model to create embeddings.
    - dense_model:  the model to create dense embeddings. For example, OpenAI embedding are set by "OpenAI/text-embedding-3-small"
    - sparse_model: the model for sparse embeddings. The sparse embeddings supported by Qdrant are used, i.e. "Qdrant/bm25"

- major: the main language model used, for example, for chatting.
    - model: the provider and the model name, e.g. "OpenAI/gpt-4.1-mini"
    - temperature: temperature (defaults to 0.1)
    - max_retries: number of connection attempts when using model (default to 2)

- major.provider_params: other parameters set for the model (model-dependent)

- minor: the accessory language model used, for example, for creating questions and for creating summaries.
    - model: the provider and the model name, e.g. "OpenAI/gpt-4.1-nano"
    - temperature: temperature (defaults to 0.1)
    - max_retries: number of connection attempts when using model (default to 2)

- major.provider_params: other parameters set for the model (model-dependent)

- aux: the auxiliary language model used, for example to classify text. This model should have low latency.
    - model: the provier and the model name, e.g. "Mistral/mistral-small-latest"
    - temperature: temperature (defaults to 0.7)
    - max_retries: number of connection attempts when using model (default to 2)

- aux.provider_params: other parameters set for the model (model-dependent)
