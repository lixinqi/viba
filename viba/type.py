# type.py
# Auto-generated VIBA AST type definitions
# Generated from VIBA self-description

from dataclasses import dataclass
from typing import Union, List, Optional, Any, Dict
from enum import Enum


# ----------------------------------------------------------------------
# Base class for all AST nodes
# ----------------------------------------------------------------------


class Type:
    """Base class for all VIBA type AST nodes."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Type":
        """
        Factory method: create appropriate Type subclass from dict.
        Uses the "node" field to determine which concrete type to instantiate.
        """
        node_type = data.get("node")
        if node_type == "Definition":
            return DefinitionType.from_dict(data)
        elif node_type == "SumType":
            return SumType.from_dict(data)
        elif node_type == "ProductType":
            return ProductType.from_dict(data)
        elif node_type == "ExponentType":
            return ExponentType.from_dict(data)
        elif node_type == "SumChain":
            return SumChainType.from_dict(data)
        elif node_type == "ProductChain":
            return ProductChainType.from_dict(data)
        elif node_type == "ExponentChain":
            return ExponentChainType.from_dict(data)
        elif node_type == "ExponentChainType":
            return ExponentChainType.from_dict(data)
        elif node_type == "TaggedType":
            return TaggedType.from_dict(data)
        elif node_type == "TypeApp":
            return TypeAppType.from_dict(data)
        elif node_type == "TypeRef":
            return TypeRefType.from_dict(data)
        elif node_type == "Identity":
            return IdentityType.from_dict(data)
        elif node_type == "Ellipsis":
            return EllipsisType.from_dict(data)
        elif node_type == "PureTag":
            return PureTagType.from_dict(data)
        elif node_type == "Literal":
            return LiteralType.from_dict(data)
        else:
            raise ValueError(f"Unknown node type: {node_type}")


# ----------------------------------------------------------------------
# Literal value representation
# ----------------------------------------------------------------------


@dataclass
class LiteralValue:
    """Container for literal values with type information."""

    value: Union[int, float, str, bool]
    type_name: str  # "int", "float", "str", "bool"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiteralValue":
        """Create LiteralValue from dict with val and val_type fields."""
        return cls(value=data["val"], type_name=data["val_type"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by parser."""
        return {"val": self.value, "val_type": self.type_name}


# ----------------------------------------------------------------------
# Concrete type definitions (all inherit from Type)
# ----------------------------------------------------------------------


@dataclass
class DefinitionType(Type):
    """Type definition: name := body"""

    node: str
    name: str
    generic_params: List[str]
    body: "Type"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefinitionType":
        """Create DefinitionType from dict, recursively parsing body."""
        return cls(
            node=data.get("node", "Definition"),
            name=data["name"],
            generic_params=data.get("generic_params", []),
            body=(
                Type.from_dict(data["body"])
                if isinstance(data["body"], dict)
                else data["body"]
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict format expected by parser."""
        return {
            "node": self.node,
            "name": self.name,
            "generic_params": self.generic_params,
            "body": self.body.to_dict() if hasattr(self.body, "to_dict") else self.body,
        }


@dataclass
class SumType(Type):
    """Sum type: left | right"""

    node: str
    left: Type
    right: Type

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SumType":
        return cls(
            node=data.get("node", "SumType"),
            left=Type.from_dict(data["left"]),
            right=Type.from_dict(data["right"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass
class ProductType(Type):
    """Product type: left * right"""

    node: str
    left: Type
    right: Type

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProductType":
        return cls(
            node=data.get("node", "ProductType"),
            left=Type.from_dict(data["left"]),
            right=Type.from_dict(data["right"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "left": self.left.to_dict(),
            "right": self.right.to_dict(),
        }


@dataclass
class ExponentType(Type):
    """Exponent type: result <- argument (function type)"""

    node: str
    result: Type
    argument: Type

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExponentType":
        return cls(
            node=data.get("node", "ExponentType"),
            result=Type.from_dict(data["result"]),
            argument=Type.from_dict(data["argument"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "result": self.result.to_dict(),
            "argument": self.argument.to_dict(),
        }


@dataclass
class TaggedType(Type):
    """Tagged type: $tag type"""

    node: str
    tag: str
    type: Type

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaggedType":
        return cls(
            node=data.get("node", "TaggedType"),
            tag=data["tag"],
            type=Type.from_dict(data["type"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"node": self.node, "tag": self.tag, "type": self.type.to_dict()}


@dataclass
class TypeAppType(Type):
    """Type application: constructor[args]"""

    node: str
    constructor: str
    args: List[Type]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypeAppType":
        args = data.get("args", [])
        parsed_args = [
            Type.from_dict(arg) if isinstance(arg, dict) else arg for arg in args
        ]
        return cls(
            node=data.get("node", "TypeApp"),
            constructor=data["constructor"],
            args=parsed_args,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node": self.node,
            "constructor": self.constructor,
            "args": [
                arg.to_dict() if hasattr(arg, "to_dict") else arg for arg in self.args
            ],
        }


@dataclass
class TypeRefType(Type):
    """Type reference: name"""

    node: str
    name: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TypeRefType":
        return cls(node=data.get("node", "TypeRef"), name=data["name"])

    def to_dict(self) -> Dict[str, Any]:
        return {"node": self.node, "name": self.name}


@dataclass
class IdentityType(Type):
    """Identity element: void (product) or never (sum), with optional alias for ()"""

    node: str
    type: str  # "ProductIdentity" or "SumIdentity"
    alias: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IdentityType":
        return cls(
            node=data.get("node", "Identity"),
            type=data["type"],
            alias=data.get("alias"),
        )

    def to_dict(self) -> Dict[str, Any]:
        result = {"node": self.node, "type": self.type}
        if self.alias is not None:
            result["alias"] = self.alias
        return result


@dataclass
class EllipsisType(Type):
    """Ellipsis: ... (open sum type)"""

    node: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EllipsisType":
        return cls(node=data.get("node", "Ellipsis"))

    def to_dict(self) -> Dict[str, Any]:
        return {"node": self.node}


@dataclass
class PureTagType(Type):
    """Pure tag: $a.b.c (standalone semantic tag)"""

    node: str
    name: str  # full tag, e.g. "$a.b.c"
    path: List[str]  # components split by "."

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PureTagType":
        return cls(
            node=data.get("node", "PureTag"),
            name=data["name"],
            path=data.get("path", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"node": self.node, "name": self.name, "path": self.path}


@dataclass
class LiteralType(Type):
    """Literal value node"""

    node: str
    val: Union[int, float, str, bool]
    val_type: str  # "int", "float", "str", "bool"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiteralType":
        return cls(
            node=data.get("node", "Literal"), val=data["val"], val_type=data["val_type"]
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"node": self.node, "val": self.val, "val_type": self.val_type}


# ----------------------------------------------------------------------
# Chain type definitions
# ----------------------------------------------------------------------


class SumChainType(Type):
    def __init__(self, *elements: Type):
        self.node = "SumChain"
        self.elements = list(elements)

    def __repr__(self) -> str:
        return f"SumChainType({', '.join(repr(e) for e in self.elements)})"

    @classmethod
    def from_dict(cls, data: dict) -> "SumChainType":
        elements = [Type.from_dict(e) for e in data.get("elements", [])]
        return cls(*elements)

    def to_dict(self) -> dict:
        return {"node": self.node, "elements": [e.to_dict() for e in self.elements]}


class ProductChainType(Type):
    def __init__(self, *elements: Type):
        self.node = "ProductChain"
        self.elements = list(elements)

    def __repr__(self) -> str:
        return f"ProductChainType({', '.join(repr(e) for e in self.elements)})"

    @classmethod
    def from_dict(cls, data: dict) -> "ProductChainType":
        elements = [Type.from_dict(e) for e in data.get("elements", [])]
        return cls(*elements)

    def to_dict(self) -> dict:
        return {"node": self.node, "elements": [e.to_dict() for e in self.elements]}


class ExponentChainType(Type):
    def __init__(self, result: Type, *args: Type):
        self.node = "ExponentChain"
        self.result = result
        self.args = list(args)

    def __repr__(self) -> str:
        args_repr = ", ".join(repr(a) for a in self.args)
        return f"ExponentChainType({repr(self.result)}{', ' + args_repr if args_repr else ''})"

    @classmethod
    def from_dict(cls, data: dict) -> "ExponentChainType":
        result = Type.from_dict(data["result"])
        args = [Type.from_dict(a) for a in data.get("args", [])]
        return cls(result, *args)

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "result": self.result.to_dict(),
            "args": [a.to_dict() for a in self.args],
        }


# ----------------------------------------------------------------------
# Main parsing function
# ----------------------------------------------------------------------


def parse(data_list: List[Dict[str, Any]]) -> List[Type]:
    """
    Parse a list of dictionaries (from JSON output of parser.py) into a list of Type AST nodes.

    Args:
        data_list: List of dictionaries, each containing a parsed VIBA definition,
                  typically from parser.py's output (which returns a list of definitions).

    Returns:
        A list of Type instances representing the AST for each definition.
    """
    result = []
    for data in data_list:
        if isinstance(data, dict):
            result.append(Type.from_dict(data))
        else:
            raise ValueError(f"Expected dict, got {type(data)}")
    return result


# ----------------------------------------------------------------------
# Example usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    # Example: parse a list of definitions
    sample_list = [
        {
            "node": "Definition",
            "name": "Option[T]",
            "generic_params": ["T"],
            "body": {
                "node": "SumType",
                "left": {
                    "node": "TaggedType",
                    "tag": "$some",
                    "type": {"node": "TypeRef", "name": "T"},
                },
                "right": {"node": "Identity", "type": "ProductIdentity", "alias": "()"},
            },
        },
        {
            "node": "Definition",
            "name": "Result[T, E]",
            "generic_params": ["T", "E"],
            "body": {
                "node": "SumType",
                "left": {
                    "node": "TaggedType",
                    "tag": "$ok",
                    "type": {"node": "TypeRef", "name": "T"},
                },
                "right": {
                    "node": "TaggedType",
                    "tag": "$err",
                    "type": {"node": "TypeRef", "name": "E"},
                },
            },
        },
    ]

    ast_list = parse(sample_list)
    print(f"Parsed {len(ast_list)} definitions")
    for i, ast in enumerate(ast_list):
        print(f"\nDefinition {i+1}: {type(ast).__name__}")
        print(f"Name: {ast.name if hasattr(ast, 'name') else 'N/A'}")
        print(f"Back to dict: {ast.to_dict()}")
