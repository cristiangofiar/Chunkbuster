"""Public exception hierarchy."""


class ChunkbusterError(Exception):
    """Base error raised by chunkbuster."""


class ConfigurationError(ChunkbusterError):
    """The serialized pipeline configuration is invalid."""


class BuildError(ChunkbusterError):
    """A pipeline cannot be built from otherwise valid inputs."""


class InvalidTaxonomyError(BuildError):
    """A taxonomy violates the strict-forest contract."""


class ExecutionError(ChunkbusterError):
    """A built pipeline failed while processing one query."""


class PreprocessingError(ExecutionError):
    """A query or stable document representation is invalid."""


class RetrievalError(ExecutionError):
    """A retriever failed or returned an invalid ranking."""


class InvalidModelOutputError(ExecutionError):
    """An external component returned identities outside its input."""

