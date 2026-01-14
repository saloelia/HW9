"""
Base classes for DeepFake detection tools.

This module defines the abstract base class and common interfaces
for all detection tools, implementing the Building Blocks design pattern.

Building Block Design:
    Input Data: Parameters passed to the execute() method
    Output Data: ToolResult containing analysis results
    Setup Data: Configuration passed during initialization

Classes:
    ToolResult: Standard result container for all tools
    BaseTool: Abstract base class for detection tools
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


@dataclass
class ToolResult(Generic[T]):
    """
    Standard result container for tool executions.

    This class provides a consistent interface for tool outputs,
    following the Building Blocks output data pattern.

    Attributes:
        success: Whether the tool execution was successful
        data: The actual result data (generic type)
        error: Error message if execution failed
        metadata: Additional metadata about the execution

    Example:
        >>> result = ToolResult(success=True, data={"score": 0.85})
        >>> if result.success:
        ...     print(result.data["score"])
        0.85
    """

    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result state after initialization."""
        if not self.success and self.error is None:
            self.error = "Unknown error occurred"

    @classmethod
    def success_result(
        cls, data: T, metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult[T]":
        """
        Create a successful result.

        Args:
            data: The result data
            metadata: Optional metadata

        Returns:
            ToolResult instance with success=True
        """
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def error_result(
        cls, error: str, metadata: Optional[Dict[str, Any]] = None
    ) -> "ToolResult[T]":
        """
        Create an error result.

        Args:
            error: Error message
            metadata: Optional metadata

        Returns:
            ToolResult instance with success=False
        """
        return cls(success=False, error=error, metadata=metadata or {})


class ToolConfig(BaseModel):
    """
    Base configuration for tools.

    Setup Data for tool initialization following Building Blocks pattern.
    """

    enabled: bool = True
    timeout: float = 60.0
    verbose: bool = False


class BaseTool(ABC):
    """
    Abstract base class for all DeepFake detection tools.

    This class defines the interface that all tools must implement,
    following the Building Blocks design pattern:

    Setup Data:
        - name: Tool identifier
        - description: Tool description for agent context
        - config: Tool-specific configuration

    Input Data:
        - Defined by each tool's execute() method

    Output Data:
        - ToolResult containing analysis data

    Subclasses must implement:
        - execute(): Perform the tool's analysis
        - get_schema(): Return the tool's input schema for the agent

    Example:
        >>> class MyTool(BaseTool):
        ...     def execute(self, input_data: Dict) -> ToolResult:
        ...         # Perform analysis
        ...         return ToolResult.success_result({"score": 0.85})
        ...
        ...     def get_schema(self) -> Dict:
        ...         return {"type": "object", "properties": {...}}
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: Optional[ToolConfig] = None,
    ) -> None:
        """
        Initialize the base tool.

        Args:
            name: Unique identifier for the tool
            description: Human-readable description for the agent
            config: Tool configuration (Setup Data)
        """
        self.name = name
        self.description = description
        self.config = config or ToolConfig()
        self._execution_count = 0
        self._total_execution_time = 0.0

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """
        Execute the tool's analysis.

        This is the main entry point for the tool. Subclasses must
        implement this method to perform their specific analysis.

        Args:
            **kwargs: Tool-specific input parameters (Input Data)

        Returns:
            ToolResult containing the analysis results (Output Data)

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement execute()")

    @abstractmethod
    def get_schema(self) -> Dict[str, Any]:
        """
        Get the JSON schema for tool inputs.

        This schema is used by the AI agent to understand what
        parameters the tool accepts.

        Returns:
            JSON schema dictionary describing input parameters

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement get_schema()")

    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get the complete tool definition for the AI agent.

        This method returns a structured definition that includes
        the tool's name, description, and input schema.

        Returns:
            Dictionary with tool definition for agent context
        """
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.get_schema(),
        }

    def validate_input(self, **kwargs: Any) -> List[str]:
        """
        Validate input parameters against the schema.

        Args:
            **kwargs: Input parameters to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        schema = self.get_schema()

        # Check required parameters
        required = schema.get("required", [])
        for param in required:
            if param not in kwargs:
                errors.append(f"Missing required parameter: {param}")

        return errors

    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics for this tool.

        Returns:
            Dictionary with execution count and timing stats
        """
        avg_time = (
            self._total_execution_time / self._execution_count
            if self._execution_count > 0
            else 0.0
        )
        return {
            "name": self.name,
            "execution_count": self._execution_count,
            "total_time": round(self._total_execution_time, 3),
            "average_time": round(avg_time, 3),
        }

    def __repr__(self) -> str:
        """String representation of the tool."""
        return f"{self.__class__.__name__}(name='{self.name}')"
