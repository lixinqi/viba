"""Subtype judgment over the Type model: is_sub_type(sub, sup) -> bool.

Contract: consumes Type values only (viba.type). Nothing here knows
about rules, results or compliance semantics.

Semantics (per design):
- GenericDefinition references are nominal: same container module
  (object identity) and same name -> equal; bodies never unfold.
- TypeDefinition references are transparent: they unfold to their
  bodies and compare structurally (B := A * $find bool <: A).
  Recursive plain definitions are guarded by the coinductive
  assumption table: a (sub, sup) pair already in flight is true.
- Leaves compare by family: literal(v) <: base iff same family;
  literal <: literal iff equal values; never <: T; T <: never iff
  T is never (a Rule's never branch forbids any other resident).
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
    NilType,
    module_get_type,
)

_BASE_TYPES = (BoolType, IntType, FloatType, StrType)
_LITERAL_TYPES = (BoolLiteralType, IntLiteralType, FloatLiteralType, StrLiteralType)
_BASE_OF_LITERAL = {
    BoolLiteralType: BoolType,
    IntLiteralType: IntType,
    FloatLiteralType: FloatType,
    StrLiteralType: StrType,
}
_SUM_NODES = (viba_ast.Sum, viba_ast.SumChain)
_PROD_NODES = (viba_ast.Product, viba_ast.ProductChain)
_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)


def is_sub_type(sub: Type, sup: Type) -> bool:
    """Return True iff sub <: sup."""
    _assert_no_poison(sup)
    return _Checker().check(sub, sup)


def _assert_no_poison(sup: Type):
    """Lint: the sup side must not contain AssertionViolated anywhere,
    whether or not the comparison would have reached it."""
    if isinstance(sup, PoisonType):
        raise RuleContainsPoisonError("AssertionViolated on the sup side")
    if not isinstance(sup, AstNodeType):
        return
    poisoned = [n for n in viba_ast.walk(sup.ast_node) if _is_poison_ref(n)]
    if poisoned:
        raise RuleContainsPoisonError("AssertionViolated on the sup side")


class _Checker:
    def __init__(self):
        self.memo: dict = {}
        self._walk_memo: dict = {}
        self._walking: set = set()

    # ------------------------------------------------------------------
    # Type-level dispatch
    # ------------------------------------------------------------------

    def check(self, sub: Type, sup: Type) -> bool:
        if isinstance(sup, PoisonType):
            raise RuleContainsPoisonError("AssertionViolated on the sup side")
        if isinstance(sub, PoisonType):
            return False
        key = (_type_key(sub), _type_key(sup))
        if key in self.memo:
            return self.memo[key]
        result = self._check_uncached(sub, sup)
        self.memo[key] = result
        return result

    def _check_uncached(self, sub: Type, sup: Type) -> bool:
        if isinstance(sub, NeverType):
            return True
        if isinstance(sup, (NeverType, NilType)):
            return type(sub) is type(sup)
        verdict = self._probe_leaves(sub, sup)
        if verdict is not None:
            return verdict
        if isinstance(sup, AstNodeType) and isinstance(sub, AstNodeType):
            return self._check_ast_pair(sub, sup)
        return False

    def _probe_leaves(self, sub: Type, sup: Type):
        probes = (self._check_base, self._check_literal, self._check_nominal)
        verdicts = (probe(sub, sup) for probe in probes)
        return next((v for v in verdicts if v is not None), None)

    def _check_base(self, sub: Type, sup: Type):
        if type(sup) not in _BASE_TYPES:
            return None
        return type(sub) is type(sup) or _BASE_OF_LITERAL.get(type(sub)) is type(sup)

    def _check_literal(self, sub: Type, sup: Type):
        if not isinstance(sup, _LITERAL_TYPES):
            return None
        return isinstance(sub, type(sup)) and sub.value == sup.value

    def _check_nominal(self, sub: Type, sup: Type):
        if isinstance(sup, BuiltinGenericType):
            return isinstance(sub, BuiltinGenericType) and sub.name == sup.name
        if isinstance(sup, OpaqueType):
            return self._same_opaque(sub, sup)
        return None

    def _same_opaque(self, sub: Type, sup: Type) -> bool:
        """Free-variable identity is the name alone (see OpaqueType)."""
        return isinstance(sub, OpaqueType) and sub.name == sup.name

    # ------------------------------------------------------------------
    # AstNodeType pairs
    # ------------------------------------------------------------------

    def _check_ast_pair(self, sub: AstNodeType, sup: AstNodeType) -> bool:
        sn, s_mod = _unwrap_definition(sub)
        sp, p_mod = _unwrap_definition(sup)
        both_generic = isinstance(sn, viba_ast.GenericDefinition)
        if both_generic and isinstance(sp, viba_ast.GenericDefinition):
            return self._same_generic(sn, s_mod, sp, p_mod)
        return self._walk(sn, s_mod, sp, p_mod)

    def _same_generic(self, sn, s_mod, sp, p_mod) -> bool:
        """Nominal: same module object and same name; never unfolded."""
        return s_mod is p_mod and sn.name == sp.name

    # ------------------------------------------------------------------
    # Structural walk over viba.ast nodes
    # ------------------------------------------------------------------

    def _walk(self, sn, s_mod: ModuleType, sp, p_mod: ModuleType) -> bool:
        sn, s_mod = self._unfold_ref(sn, s_mod)
        sp, p_mod = self._unfold_ref(sp, p_mod)
        key = (id(sn), id(sp), id(s_mod), id(p_mod))
        if key in self._walking:
            return True  # coinductive assumption (design: 假设表)
        if key in self._walk_memo:
            return self._walk_memo[key]
        self._walking.add(key)
        result = self._walk_inner(sn, s_mod, sp, p_mod)
        self._walking.discard(key)
        self._walk_memo[key] = result
        return result

    def _unfold_ref(self, node, module: ModuleType):
        """A TypeRef to a plain TypeDefinition is transparent: unfold
        to its body, whose own TypeRefs resolve in its home module."""
        if not isinstance(node, viba_ast.TypeRef):
            return node, module
        resolved = module_get_type(module, node.name)
        if isinstance(resolved, Err):
            return node, module
        target = resolved.value
        if not isinstance(target, AstNodeType):
            return node, module
        if not isinstance(target.ast_node, viba_ast.TypeDefinition):
            return node, module
        return target.ast_node.body, target.container_module

    def _walk_inner(self, sn, s_mod: ModuleType, sp, p_mod: ModuleType) -> bool:
        # Poison beats even the ellipsis wildcard: AssertionViolated
        # fails against ANY sup, and must never appear on the sup side.
        if _is_poison_ref(sn):
            return False
        if _is_poison_ref(sp):
            raise RuleContainsPoisonError("AssertionViolated on the sup side")
        if isinstance(sp, viba_ast.Ellipsis):
            return True
        if isinstance(sn, viba_ast.Never):
            return True  # bottom fits anywhere
        if isinstance(sp, (viba_ast.Nil, viba_ast.Never)):
            return type(sn) is type(sp)  # never branch admits only never
        lifted = self._walk_lifted(sn, s_mod, sp, p_mod)
        if lifted is not None:
            return lifted
        return self._walk_structural(sn, s_mod, sp, p_mod)

    def _walk_lifted(self, sn, s_mod, sp, p_mod):
        """Handle sides that lift to the Type level (Constants, TypeRefs)."""
        sub_t = _lift(sn, s_mod)
        sup_t = _lift(sp, p_mod)
        if sub_t is not None and sup_t is not None:
            return self.check(sub_t, sup_t)
        if sub_t is not None:
            return self._leaf_vs_struct(sn, s_mod, sp, p_mod, sub_t)
        if sup_t is not None:
            return self._struct_vs_leaf(sn, s_mod, sp, p_mod, sup_t)
        return None

    def _leaf_vs_struct(self, sn, s_mod, sp, p_mod, sub_t) -> bool:
        if isinstance(sp, _SUM_NODES):
            return self._any_branch(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.Tagged):
            return self._walk(sn, s_mod, sp.type, p_mod)
        return self.check(sub_t, AstNodeType(sp, p_mod))

    def _struct_vs_leaf(self, sn, s_mod, sp, p_mod, sup_t) -> bool:
        if isinstance(sn, _SUM_NODES):
            return self._all_branches(sn, s_mod, sp, p_mod)
        if isinstance(sn, viba_ast.Tagged):
            return self._walk(sn.type, s_mod, sp, p_mod)
        return self.check(AstNodeType(sn, s_mod), sup_t)

    def _any_branch(self, sn, s_mod, sp, p_mod) -> bool:
        return any(self._walk(sn, s_mod, b, p_mod) for b in _flatten_sum(sp))

    def _all_branches(self, sn, s_mod, sp, p_mod) -> bool:
        return all(self._walk(b, s_mod, sp, p_mod) for b in _flatten_sum(sn))

    # ------------------------------------------------------------------
    # Structural cases (neither side lifts)
    # ------------------------------------------------------------------

    def _walk_structural(self, sn, s_mod, sp, p_mod) -> bool:
        if isinstance(sp, _SUM_NODES):
            return self._walk_sum(sn, s_mod, sp, p_mod)
        if isinstance(sn, _SUM_NODES):
            return self._all_branches(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.Tagged):
            return self._walk_tagged(sn, s_mod, sp, p_mod)
        if isinstance(sp, _PROD_NODES) and isinstance(sn, _PROD_NODES):
            return self._walk_products(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.Tuple):
            return self._walk_tuple(sn, s_mod, sp, p_mod)
        if isinstance(sp, _EXP_NODES):
            return self._walk_exponent(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.TypeApp) and isinstance(sn, viba_ast.TypeApp):
            return self._walk_typeapps(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.CodeBlock):
            return isinstance(sn, viba_ast.CodeBlock) and sn.code == sp.code
        return False

    def _walk_sum(self, sn, s_mod, sp, p_mod) -> bool:
        if isinstance(sn, _SUM_NODES):
            return self._all_branches(sn, s_mod, sp, p_mod)
        return self._any_branch(sn, s_mod, sp, p_mod)

    def _walk_tagged(self, sn, s_mod, sp, p_mod) -> bool:
        if isinstance(sn, _PROD_NODES):
            tagged, _ = self._split_product(sn, s_mod)
            return sp.tag in tagged and self._walk(tagged[sp.tag], s_mod, sp.type, p_mod)
        if not isinstance(sn, viba_ast.Tagged):
            return self._walk(sn, s_mod, sp.type, p_mod)
        return sn.tag == sp.tag and self._walk(sn.type, s_mod, sp.type, p_mod)

    def _walk_products(self, sn, s_mod, sp, p_mod) -> bool:
        """Tagged products match by tag (commutative); sup's tags must
        all be present in sub (width). Untagged elements pair in order.
        Bare TypeRefs to plain TypeDefinitions unfold so their tags
        participate (B := A * $find bool carries A's tags)."""
        sub_tagged, sub_bare = self._split_product(sn, s_mod)
        sup_tagged, sup_bare = self._split_product(sp, p_mod)
        if not self._tags_covered(sub_tagged, s_mod, sup_tagged, p_mod):
            return False
        if len(sub_bare) != len(sup_bare):
            return False
        pairs = zip(sub_bare, sup_bare)
        return all(self._walk(a, s_mod, b, p_mod) for a, b in pairs)

    def _split_product(self, node, module):
        """Flatten a Product/ProductChain into ({tag: body}, [bare]),
        unfolding transparent TypeDefinitions along the way."""
        tagged, bare = {}, []
        for elem in _product_elements(node):
            t, b = self._split_element(elem, module)
            tagged.update(t)
            bare.extend(b)
        return tagged, bare

    def _split_element(self, elem, module):
        elem, elem_mod = self._unfold_ref(elem, module)
        if isinstance(elem, _PROD_NODES):
            return self._split_product(elem, elem_mod)
        if isinstance(elem, viba_ast.Tagged):
            return {elem.tag: elem.type}, []
        return {}, [elem]

    def _tags_covered(self, sub_tagged, s_mod, sup_tagged, p_mod) -> bool:
        items = sup_tagged.items()
        checks = (self._match_tag(sub_tagged, t, s_mod, b, p_mod) for t, b in items)
        return all(checks)

    def _match_tag(self, sub_tagged, tag, s_mod, sup_body, p_mod) -> bool:
        if tag not in sub_tagged:
            return False
        return self._walk(sub_tagged[tag], s_mod, sup_body, p_mod)

    def _walk_tuple(self, sn, s_mod, sp, p_mod) -> bool:
        if not isinstance(sn, viba_ast.Tuple) or len(sn.elements) != len(sp.elements):
            return False
        pairs = zip(sn.elements, sp.elements)
        return all(self._walk(a, s_mod, b, p_mod) for a, b in pairs)

    def _walk_exponent(self, sn, s_mod, sp, p_mod) -> bool:
        """Result covariant; arguments contravariant in application
        order; a longer sub argument list is a subtype of a shorter."""
        if not isinstance(sn, _EXP_NODES):
            return False
        sub_res, sub_args = _exponent_parts(sn)
        sup_res, sup_args = _exponent_parts(sp)
        if len(sub_args) < len(sup_args):
            return False
        if not self._walk(sub_res, s_mod, sup_res, p_mod):
            return False
        pairs = zip(sup_args, sub_args)
        return all(self._walk(sa, p_mod, sb, s_mod) for sa, sb in pairs)

    def _walk_typeapps(self, sn, s_mod, sp, p_mod) -> bool:
        if len(sn.args) != len(sp.args):
            return False
        sub_c = self._resolve_constructor(sn.constructor, s_mod)
        sup_c = self._resolve_constructor(sp.constructor, p_mod)
        if not self._generic_equal(sub_c, sup_c):
            return False
        pairs = zip(sn.args, sp.args)
        return all(self._walk(a, s_mod, b, p_mod) for a, b in pairs)

    def _resolve_constructor(self, name: str, module: ModuleType):
        if name == "AssertionViolated":
            return PoisonType()
        resolved = module_get_type(module, name)
        if isinstance(resolved, Err):
            return OpaqueType(module, name)
        return resolved.value

    def _generic_equal(self, sub_c, sup_c) -> bool:
        if isinstance(sub_c, BuiltinGenericType) or isinstance(sup_c, BuiltinGenericType):
            same_kind = isinstance(sub_c, BuiltinGenericType)
            return same_kind and isinstance(sup_c, BuiltinGenericType) and sub_c.name == sup_c.name
        sub_def = _as_definition(getattr(sub_c, "ast_node", None))
        sup_def = _as_definition(getattr(sup_c, "ast_node", None))
        if sub_def is None or sup_def is None:
            return False
        same_module = getattr(sub_c, "container_module", None) is getattr(sup_c, "container_module", None)
        return same_module and sub_def.name == sup_def.name


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _as_definition(node):
    kinds = (viba_ast.TypeDefinition, viba_ast.GenericDefinition)
    return node if isinstance(node, kinds) else None


def _unwrap_definition(entry: AstNodeType):
    """Plain TypeDefinitions are transparent: compare their bodies.
    GenericDefinitions stay wrapped (they are nominal)."""
    node = entry.ast_node
    if isinstance(node, viba_ast.TypeDefinition):
        return node.body, entry.container_module
    return node, entry.container_module


def _is_poison_ref(node) -> bool:
    return isinstance(node, viba_ast.TypeRef) and node.name == "AssertionViolated"


def _lift(node, module: ModuleType):
    """Lift a Constant, TypeRef, Nil or Never to a Type; else None."""
    if isinstance(node, viba_ast.Constant):
        return _literal_type(node.value)
    if isinstance(node, viba_ast.Nil):
        return NilType()
    if isinstance(node, viba_ast.Never):
        return NeverType()
    if isinstance(node, viba_ast.TypeRef):
        if node.name == "AssertionViolated":
            return PoisonType()
        resolved = module_get_type(module, node.name)
        if isinstance(resolved, Err):
            return OpaqueType(module, node.name)
        return resolved.value
    return None


def _literal_type(value) -> Type:
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


def _product_elements(node):
    if isinstance(node, viba_ast.Product):
        return _product_elements(node.left) + _product_elements(node.right)
    if isinstance(node, viba_ast.ProductChain):
        return list(node.elements)
    return [node]


def _flatten_sum(node):
    """Flatten a Sum or SumChain into a branch list."""
    if isinstance(node, viba_ast.Sum):
        return [node.left, node.right]
    if isinstance(node, viba_ast.SumChain):
        return list(node.elements)
    return [node]


def _exponent_parts(node):
    """Normalize an Exponent or ExponentChain to (result, args-in-
    application-order); nested binary Exponents flatten fully so raw
    parses and canonical chains line up."""
    if isinstance(node, viba_ast.ExponentChain):
        return node.result, list(node.args)
    if not isinstance(node, viba_ast.Exponent):
        raise TypeError(f"not an exponent: {node!r}")
    head = node.result
    base = head if not isinstance(head, _EXP_NODES) else None
    res, args = (base, []) if base is not None else _exponent_parts(head)
    return res, [node.argument] + args


def _type_key(t: Type):
    """Canonical comparison key for the memo table."""
    if isinstance(t, AstNodeType):
        defn = _as_definition(t.ast_node)
        node_key = ("defn", defn.name) if defn is not None else ("inline", id(t.ast_node))
        return ("ast", id(t.container_module), node_key)
    if isinstance(t, BuiltinGenericType):
        return ("generic", t.name)
    if isinstance(t, OpaqueType):
        return ("opaque", t.name)
    if isinstance(t, _LITERAL_TYPES):
        return (type(t).__name__, t.value)
    return (type(t).__name__,)
