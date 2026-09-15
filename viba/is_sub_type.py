"""Subtype judgment over the Type model: is_sub_type(sub, sup) -> bool.

Contract: consumes Type values only (viba.type). Nothing here knows
about rules, results or compliance semantics.

Semantics (nominal, per design):
- Leaves compare by family: literal(v) <: base iff same family;
  literal <: literal iff equal values; never <: T; T <: never iff
  T is never (a Rule's never branch forbids any other resident).
- AstNodeType wrapping a *definition* is nominal: same container
  module (object identity) and same definition name -> equal;
  anything else -> not a subtype. Bodies of differently-named
  definitions are never unfolded.
- AstNodeType wrapping inline structure (sums, products, tuples,
  exponents, type applications) is compared structurally, recursing
  at the Type level whenever a side lifts to a Type (TypeRef
  resolution, constants).
- TypeRef resolves through module_get_type in its own container
  module (lexical scoping); unresolvable names become OpaqueType
  nominal atoms keyed by (module identity, name) — exactly the
  semantics of free generic parameters.
- AssertionViolated (PoisonType): on sub -> False; on sup -> lint
  error (RuleContainsPoisonError).
- `...` on the sup side absorbs anything.
- The memo table maps (sub_key, sup_key) to the result in flight;
  under nominal semantics no cycle can require assuming a pair, so
  it is a pure memoization / shared-subgraph guard.
"""

from viba import ast as viba_ast
from viba.type import (
    AstNodeType,
    BoolLiteralType,
    BoolType,
    BuiltinGenericType,
    BuiltinModuleType,
    CustomModuleType,
    Err,
    FloatLiteralType,
    FloatType,
    IntLiteralType,
    IntType,
    ModuleType,
    NeverType,
    OpaqueType,
    PoisonType,
    RuleContainsPoisonError,
    StrLiteralType,
    StrType,
    Type,
    UnitType,
    UnresolvedTypeError,
    module_get_type,
)

_BASE_OF_LITERAL = {
    BoolLiteralType: BoolType,
    IntLiteralType: IntType,
    FloatLiteralType: FloatType,
    StrLiteralType: StrType,
}


def is_sub_type(sub: Type, sup: Type) -> bool:
    """Return True iff sub <: sup."""
    return _Checker().check(sub, sup)


class _Checker:
    def __init__(self):
        self.memo: dict = {}

    # ------------------------------------------------------------------
    # Type-level dispatch
    # ------------------------------------------------------------------

    def check(self, sub: Type, sup: Type) -> bool:
        if isinstance(sup, PoisonType):
            raise RuleContainsPoisonError(
                "AssertionViolated on the sup side: rules must not contain poison"
            )
        if isinstance(sub, PoisonType):
            return False

        key = (_type_key(sub), _type_key(sup))
        if key in self.memo:
            return self.memo[key]

        result = self._check_uncached(sub, sup)
        self.memo[key] = result
        return result

    def _check_uncached(self, sub: Type, sup: Type) -> bool:
        # never is bottom: any sub is a subtype, including evidence-never.
        if isinstance(sub, NeverType):
            return True
        # A Rule-side never branch admits only never.
        if isinstance(sup, NeverType):
            return isinstance(sub, NeverType)

        # unit admits unit (A * void === A is handled by tag matching).
        if isinstance(sup, UnitType):
            return isinstance(sub, UnitType)

        # Base types are nominal: int is not float, literals only fit
        # their own family.
        if type(sup) in (BoolType, IntType, FloatType, StrType):
            return type(sub) is type(sup) or _BASE_OF_LITERAL.get(type(sub)) is type(sup)

        if isinstance(sup, BoolLiteralType):
            return isinstance(sub, BoolLiteralType) and sub.value == sup.value
        if isinstance(sup, IntLiteralType):
            return isinstance(sub, IntLiteralType) and sub.value == sup.value
        if isinstance(sup, FloatLiteralType):
            return isinstance(sub, FloatLiteralType) and sub.value == sup.value
        if isinstance(sup, StrLiteralType):
            return isinstance(sub, StrLiteralType) and sub.value == sup.value

        # Built-in generic constructors: nominal by name; arguments are
        # compared at the TypeApp layer (covariant), not here.
        if isinstance(sup, BuiltinGenericType):
            return isinstance(sub, BuiltinGenericType) and sub.name == sup.name

        # Opaque nominal atoms (free generic parameters, unresolvable
        # names): equal iff same module identity and same name.
        if isinstance(sup, OpaqueType):
            return (
                isinstance(sub, OpaqueType)
                and sub.container_module is sup.container_module
                and sub.name == sup.name
            )

        # Structure layer.
        if isinstance(sup, AstNodeType) and isinstance(sub, AstNodeType):
            return self._check_ast_pair(sub, sup)

        # A literal/base leaf against structure, modules, etc.
        return False

    # ------------------------------------------------------------------
    # AstNodeType pairs
    # ------------------------------------------------------------------

    def _check_ast_pair(self, sub: AstNodeType, sup: AstNodeType) -> bool:
        sub_def = _as_definition(sub.ast_node)
        sup_def = _as_definition(sup.ast_node)
        if sub_def is not None and sup_def is not None:
            # Nominal: same module object, same name. Different names
            # are different types — bodies are never unfolded.
            return (
                sub.container_module is sup.container_module
                and sub_def.name == sup_def.name
            )
        return self._walk(sub.ast_node, sub.container_module,
                          sup.ast_node, sup.container_module)

    # ------------------------------------------------------------------
    # Structural walk over viba.ast nodes
    # ------------------------------------------------------------------

    def _walk(self, sn, s_mod: ModuleType, sp, p_mod: ModuleType) -> bool:
        # Wildcard on the Rule side: anything goes.
        if isinstance(sp, viba_ast.Ellipsis):
            return True

        # never on the sub side is bottom: fits anywhere. void/never
        # on the sup side only admit their own kind. Decided before
        # any TypeRef lifting so unknown names never need resolving.
        if isinstance(sn, viba_ast.Never):
            return True
        if isinstance(sp, viba_ast.Void):
            return isinstance(sn, viba_ast.Void)
        if isinstance(sp, viba_ast.Never):
            return isinstance(sn, viba_ast.Never)

        # TypeRef vs TypeRef: nominal leaf comparison by name; no
        # resolution needed (unknown names simply do not match).
        if (
            isinstance(sn, viba_ast.TypeRef)
            and isinstance(sp, viba_ast.TypeRef)
            and s_mod is p_mod
        ):
            return sn.name == sp.name

        # Lift either side to the Type level when possible.
        sub_t = _lift(sn, s_mod)
        sup_t = _lift(sp, p_mod)
        if sub_t is not None and sup_t is not None:
            return self.check(sub_t, sup_t)
        if sub_t is not None:
            # A leaf sub must still be allowed to pick a sup branch or
            # pass through a sup tag.
            if isinstance(sp, (viba_ast.Sum, viba_ast.SumChain)):
                return any(
                    self._walk(sn, s_mod, b, p_mod) for b in _flatten_sum(sp)
                )
            if isinstance(sp, viba_ast.Tagged):
                return self._walk(sn, s_mod, sp.type, p_mod)
            return self.check(sub_t, AstNodeType(sp, p_mod))
        if sup_t is not None:
            # A structural sub against a leaf sup: every sub branch (or
            # a single tag's body) must fit the leaf.
            if isinstance(sn, (viba_ast.Sum, viba_ast.SumChain)):
                return all(
                    self._walk(b, s_mod, sp, p_mod) for b in _flatten_sum(sn)
                )
            if isinstance(sn, viba_ast.Tagged):
                return self._walk(sn.type, s_mod, sp, p_mod)
            return self.check(AstNodeType(sn, s_mod), sup_t)

        # Sum: every sub branch must be covered by the sup; a bare
        # witness may pick any sup branch. Chains flatten.
        if isinstance(sp, (viba_ast.Sum, viba_ast.SumChain)) and isinstance(
            sn, (viba_ast.Sum, viba_ast.SumChain)
        ):
            return all(
                self._walk(b, s_mod, sp, p_mod) for b in _flatten_sum(sn)
            )
        if isinstance(sp, (viba_ast.Sum, viba_ast.SumChain)):
            return any(
                self._walk(sn, s_mod, b, p_mod) for b in _flatten_sum(sp)
            )
        if isinstance(sn, (viba_ast.Sum, viba_ast.SumChain)):
            return all(
                self._walk(b, s_mod, sp, p_mod) for b in _flatten_sum(sn)
            )

        if isinstance(sp, (viba_ast.Product, viba_ast.ProductChain)) and isinstance(
            sn, (viba_ast.Product, viba_ast.ProductChain)
        ):
            return self._walk_products(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.Tagged) and isinstance(
            sn, (viba_ast.Product, viba_ast.ProductChain)
        ):
            sub_tagged, _ = _split_product(sn)
            return sp.tag in sub_tagged and self._walk(
                sub_tagged[sp.tag], s_mod, sp.type, p_mod
            )
        if isinstance(sp, viba_ast.Tuple) and isinstance(sn, viba_ast.Tuple):
            if len(sn.elements) != len(sp.elements):
                return False
            return all(
                self._walk(a, s_mod, b, p_mod)
                for a, b in zip(sn.elements, sp.elements)
            )
        if isinstance(sp, (viba_ast.Exponent, viba_ast.ExponentChain)) and isinstance(
            sn, (viba_ast.Exponent, viba_ast.ExponentChain)
        ):
            sub_res, sub_args = _exponent_parts(sn)
            sup_res, sup_args = _exponent_parts(sp)
            # Result covariant; arguments contravariant, pairwise in
            # application order; a longer sub argument list is a
            # subtype of a shorter one (fewer parameters accepted).
            if len(sub_args) < len(sup_args):
                return False
            return self._walk(sub_res, s_mod, sup_res, p_mod) and all(
                self._walk(sb, p_mod, sa, s_mod)
                for sa, sb in zip(sup_args, sub_args)
            )
        if isinstance(sp, viba_ast.TypeApp) and isinstance(sn, viba_ast.TypeApp):
            return self._walk_typeapps(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.Tagged) and isinstance(sn, viba_ast.Tagged):
            return sn.tag == sp.tag and self._walk(sn.type, s_mod, sp.type, p_mod)
        if isinstance(sp, viba_ast.Tagged):
            return self._walk(sn, s_mod, sp.type, p_mod)
        if isinstance(sn, viba_ast.Tagged):
            return self._walk(sn.type, s_mod, sp, p_mod)

        if isinstance(sp, viba_ast.CodeBlock):
            # Opaque content: equal iff verbatim text matches.
            return isinstance(sn, viba_ast.CodeBlock) and sn.code == sp.code

        return False

    def _walk_products(self, sn, s_mod, sp, p_mod) -> bool:
        """Tagged products match by tag (commutative, keyword-style);
        sup's tags must all be present in sub (width). Untagged
        elements compare pairwise in order."""
        sub_tagged, sub_bare = _split_product(sn)
        sup_tagged, sup_bare = _split_product(sp)
        for tag, sup_body in sup_tagged.items():
            if tag not in sub_tagged:
                return False
            if not self._walk(sub_tagged[tag], s_mod, sup_body, p_mod):
                return False
        if len(sub_bare) != len(sup_bare):
            return False
        return all(
            self._walk(a, s_mod, b, p_mod) for a, b in zip(sub_bare, sup_bare)
        )

    def _walk_typeapps(self, sn, s_mod, sp, p_mod) -> bool:
        if len(sn.args) != len(sp.args):
            return False
        sub_ctor = self._resolve_constructor(sn.constructor, s_mod)
        sup_ctor = self._resolve_constructor(sp.constructor, p_mod)
        if isinstance(sub_ctor, BuiltinGenericType) or isinstance(sup_ctor, BuiltinGenericType):
            if not (isinstance(sub_ctor, BuiltinGenericType)
                    and isinstance(sup_ctor, BuiltinGenericType)
                    and sub_ctor.name == sup_ctor.name):
                return False
        else:
            # User-defined generic constructors: nominal (module, name).
            sub_def = _as_definition(getattr(sub_ctor, "ast_node", None))
            sup_def = _as_definition(getattr(sup_ctor, "ast_node", None))
            if sub_def is None or sup_def is None:
                return False
            if not (getattr(sub_ctor, "container_module", None) is getattr(sup_ctor, "container_module", None)
                    and sub_def.name == sup_def.name):
                return False
        return all(
            self._walk(a, s_mod, b, p_mod) for a, b in zip(sn.args, sp.args)
        )

    def _resolve_constructor(self, name: str, module: ModuleType):
        if name == "AssertionViolated":
            return PoisonType()
        resolved = module_get_type(module, name)
        if isinstance(resolved, Err):
            return OpaqueType(module, name)
        return resolved.value


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _as_definition(node):
    if isinstance(node, (viba_ast.TypeDefinition, viba_ast.GenericDefinition)):
        return node
    return None


def _lift(node, module: ModuleType):
    """Lift a Constant or TypeRef to a Type; None for structural nodes."""
    if isinstance(node, viba_ast.Constant):
        value = node.value
        # bool must be checked before int (bool is a subclass of int).
        if isinstance(value, bool):
            return BoolLiteralType(value)
        if isinstance(value, int):
            return IntLiteralType(value)
        if isinstance(value, float):
            return FloatLiteralType(value)
        if isinstance(value, str):
            return StrLiteralType(value)
        raise TypeError(f"unsupported literal: {value!r}")
    if isinstance(node, viba_ast.TypeRef):
        if node.name == "AssertionViolated":
            return PoisonType()
        resolved = module_get_type(module, node.name)
        if isinstance(resolved, Err):
            return OpaqueType(module, node.name)
        return resolved.value
    return None


def _split_product(node):
    """Flatten a (left-nested) Product or ProductChain into
    ({tag: body}, [bare elements])."""
    tagged: dict = {}
    bare = []

    def collect(n):
        if isinstance(n, viba_ast.Product):
            collect(n.left)
            collect(n.right)
        elif isinstance(n, viba_ast.ProductChain):
            for elem in n.elements:
                collect(elem)
        elif isinstance(n, viba_ast.Tagged):
            tagged[n.tag] = n.type
        else:
            bare.append(n)

    collect(node)
    return tagged, bare


def _flatten_sum(node):
    """Flatten a Sum or SumChain into a branch list."""
    if isinstance(node, viba_ast.Sum):
        return [node.left, node.right]
    if isinstance(node, viba_ast.SumChain):
        return list(node.elements)
    return [node]


def _exponent_parts(node):
    """Normalize an Exponent or ExponentChain to (result, args)
    with args in application order."""
    if isinstance(node, viba_ast.Exponent):
        return node.result, [node.argument]
    if isinstance(node, viba_ast.ExponentChain):
        return node.result, list(node.args)
    raise TypeError(f"not an exponent: {node!r}")


def _type_key(t: Type):
    """Canonical comparison key for the memo table."""
    if isinstance(t, AstNodeType):
        defn = _as_definition(t.ast_node)
        node_key = ("defn", defn.name) if defn is not None else ("inline", id(t.ast_node))
        return ("ast", id(t.container_module), node_key)
    if isinstance(t, BuiltinGenericType):
        return ("generic", t.name)
    if isinstance(t, OpaqueType):
        return ("opaque", id(t.container_module), t.name)
    if isinstance(t, BoolLiteralType):
        return ("bool-lit", t.value)
    if isinstance(t, IntLiteralType):
        return ("int-lit", t.value)
    if isinstance(t, FloatLiteralType):
        return ("float-lit", t.value)
    if isinstance(t, StrLiteralType):
        return ("str-lit", t.value)
    return (type(t).__name__,)
