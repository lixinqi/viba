"""Subtype judgment over the Type model: is_sub_type(sub, sup) -> Result[bool].

Contract: consumes Type values only (viba.type). Nothing here knows
about rules, results or compliance semantics.

Ok(True/False) is the judgment. Err is a lint error, never a verdict:
- ellipsis (...) anywhere on either side -> Err (an open type is a
  contract-authoring mistake, not something to judge);
- AssertionFailed on the sup side -> Err; on the sub side it is a
  normal negative witness, Ok(False).

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
  module (lexical scoping); when that fails, the env_get bindings of
  the side's AstNodeType stack are consulted, innermost first (this
  is how free names — e.g. generic parameters — get their meanings).
  A TypeRef the judgment actually reaches and cannot resolve by
  either channel is a contract authoring mistake: the check aborts
  with Err (UnresolvedTypeError caught at the boundary). Generic
  definition bodies never trigger this — they are nominal and never
  unfold.
- AssertionFailed is not special here: it is a plain nominal
  generic from the builtin library (viba/builtin.viba). On the sub
  side it simply never seats (Ok(False)); on the sup side it is a
  lint error, detected by name.
- Applied generics (TypeApp): both sides applied stays nominal —
  same constructor and pairwise actuals. Exactly one side applied
  unfolds structurally: the generic's body is compared with formal
  parameters bound to the actuals through env_get (a TypeRef actual
  resolves eagerly in its lexical scope). Unfoldings are cached and
  guarded by a coinductive assumption table keyed on the node and
  its resolved actuals, so recursive generics terminate.
  AssertionFailed is excluded from unfolding: poison stays nominal
  and a sub-side witness never seats.
- Literal containers: ListLiteral[a, b, c] is a resident of
  list[a | b | c] (containment: every element fits the union);
  SetLiteral likewise; DictLiteral[(k, v), ...] of
  dict[k_union, v_union]. The reverse is False, and so is
  mixing container families.
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
    Ok,
    Result,
    StrLiteralType,
    StrType,
    Type,
    NilType,
    UnresolvedTypeError,
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


def is_sub_type(sub: Type, sup: Type) -> Result:
    """Ok(True/False) is the judgment; Err is a lint/resolution error."""
    err = _lint_error(sub, sup)
    if err is not None:
        return Err(err)
    try:
        return Ok(_Checker().check(sub, sup))
    except UnresolvedTypeError as exc:
        return Err(str(exc))


def _lint_error(sub: Type, sup: Type):
    """Ellipsis anywhere, or AssertionFailed on the sup side."""
    for label, side in (("sub", sub), ("sup", sup)):
        if not isinstance(side, AstNodeType):
            continue
        nodes = viba_ast.walk(side.ast_node)
        if any(isinstance(n, viba_ast.Ellipsis) for n in nodes):
            return f"ellipsis is not allowed on the {label} side"
    return _poison_error(sup)


def _poison_error(sup: Type):
    """Name-based lint: a Rule must not require its own violation."""
    if not isinstance(sup, AstNodeType):
        return None
    poisoned = [n for n in viba_ast.walk(sup.ast_node) if _is_poison_ref(n)]
    return "AssertionFailed on the sup side" if poisoned else None


class _Checker:
    def __init__(self):
        self.memo: dict = {}
        self._walk_memo: dict = {}
        self._walking: set = set()
        self._env_stacks = {"sub": [], "sup": []}
        self._unfolding: set = set()
        self._unfolded: dict = {}

    # ------------------------------------------------------------------
    # Type-level dispatch
    # ------------------------------------------------------------------

    def check(self, sub: Type, sup: Type) -> bool:
        sub_key = _type_key(sub, self._env_ids("sub"))
        sup_key = _type_key(sup, self._env_ids("sup"))
        key = (sub_key, sup_key)
        if key in self.memo:
            return self.memo[key]
        pushed = self._push_envs(sub, sup)
        try:
            result = self._check_uncached(sub, sup)
        finally:
            self._pop_envs(pushed)
        self.memo[key] = result
        return result

    def _push_envs(self, sub: Type, sup: Type):
        """An AstNodeType carrying env_get scopes its own free names."""
        pushed = []
        for side, t in (("sub", sub), ("sup", sup)):
            self._push_one(side, t, pushed)
        return pushed

    def _push_one(self, side: str, t, pushed):
        env = getattr(t, "env_get", None)
        if env is None:
            return
        self._env_stacks[side].append(env)
        pushed.append(side)

    def _pop_envs(self, pushed):
        for side in pushed:
            self._env_stacks[side].pop()

    def _env_ids(self, side: str) -> tuple:
        return tuple(id(e) for e in self._env_stacks[side])

    def _resolve_name(self, name: str, module: ModuleType, side: str) -> Result:
        """Module definitions win; the side's env_get supplements
        (innermost binding first) — the free-variable channel."""
        resolved = module_get_type(module, name)
        if isinstance(resolved, Ok):
            return resolved
        envs = reversed(self._env_stacks[side])
        hits = (e(name) for e in envs)
        return next((h for h in hits if isinstance(h, Ok)), resolved)

    def _lift(self, node, module: ModuleType, side: str):
        """Lift a Constant, TypeRef, Nil or Never to a Type; else None.
        An unresolvable TypeRef is a contract authoring mistake."""
        if isinstance(node, viba_ast.Constant):
            return _literal_type(node.value)
        if isinstance(node, viba_ast.Nil):
            return NilType()
        if isinstance(node, viba_ast.Never):
            return NeverType()
        if isinstance(node, viba_ast.TypeRef):
            return self._lift_ref(node, module, side)
        return None

    def _lift_ref(self, node, module: ModuleType, side: str):
        resolved = self._resolve_name(node.name, module, side)
        if isinstance(resolved, Err):
            raise UnresolvedTypeError(f"unresolvable TypeRef {node.name!r}")
        return resolved.value

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
        return None

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
        sn, s_mod = self._unfold_ref(sn, s_mod, "sub")
        sp, p_mod = self._unfold_ref(sp, p_mod, "sup")
        senv, penv = self._env_ids("sub"), self._env_ids("sup")
        key = (id(sn), id(sp), id(s_mod), id(p_mod), senv, penv)
        if key in self._walking:
            return True  # coinductive assumption (design: 假设表)
        if key in self._walk_memo:
            return self._walk_memo[key]
        self._walking.add(key)
        result = self._walk_inner(sn, s_mod, sp, p_mod)
        self._walking.discard(key)
        self._walk_memo[key] = result
        return result

    def _unfold_ref(self, node, module: ModuleType, side: str):
        """A TypeRef to a plain TypeDefinition is transparent: unfold
        to its body, whose own TypeRefs resolve in its home module."""
        if not isinstance(node, viba_ast.TypeRef):
            return node, module
        resolved = self._resolve_name(node.name, module, side)
        if isinstance(resolved, Err):
            return node, module
        target = resolved.value
        if not isinstance(target, AstNodeType):
            return node, module
        if not isinstance(target.ast_node, viba_ast.TypeDefinition):
            return node, module
        return target.ast_node.body, target.container_module

    def _walk_inner(self, sn, s_mod: ModuleType, sp, p_mod: ModuleType) -> bool:
        if isinstance(sn, viba_ast.Never):
            return True  # bottom fits anywhere
        if isinstance(sp, (viba_ast.Nil, viba_ast.Never)):
            return type(sn) is type(sp)  # never branch admits only never
        mixed = self._walk_mixed_typeapp(sn, s_mod, sp, p_mod)
        if mixed is not None:
            return mixed
        lifted = self._walk_lifted(sn, s_mod, sp, p_mod)
        if lifted is not None:
            return lifted
        return self._walk_structural(sn, s_mod, sp, p_mod)

    # ------------------------------------------------------------------
    # Applied generics: unfold one side, params bound via env_get
    # ------------------------------------------------------------------

    def _walk_mixed_typeapp(self, sn, s_mod, sp, p_mod):
        """Exactly one side is a TypeApp. Generic definitions unfold
        structurally; everything else (builtin generics, literal
        containers) falls through to lifting. Both sides applied
        stays nominal (see _walk_typeapps)."""
        sn_app = isinstance(sn, viba_ast.TypeApp)
        if sn_app == isinstance(sp, viba_ast.TypeApp):
            return None
        node, module, side, other, other_mod = (
            (sn, s_mod, "sub", sp, p_mod) if sn_app
            else (sp, p_mod, "sup", sn, s_mod))
        if not self._is_generic_application(node, module, side):
            return None
        return self._unfold_typeapp(node, module, side, other, other_mod)

    def _is_generic_application(self, node, module, side) -> bool:
        """True only when the constructor is a GenericDefinition:
        builtin generics and literal containers lift instead."""
        resolved = self._resolve_name(node.constructor, module, side)
        if isinstance(resolved, Err):
            raise UnresolvedTypeError(f"unresolvable constructor {node.constructor!r}")
        target = resolved.value
        return isinstance(target, AstNodeType) and isinstance(
            target.ast_node, viba_ast.GenericDefinition)

    def _unfold_typeapp(self, node, module, side, other, other_mod) -> bool:
        app_key = self._app_key(node, module, side)
        if app_key in self._unfolding:
            return True  # coinductive assumption (递归展开兜底)
        entry = self._applied_meaning(app_key, node, module, side)
        if entry is None:
            return False
        self._unfolding.add(app_key)
        self._env_stacks[side].append(entry.env_get)
        try:
            return self._walk_unfolded(entry, side, other, other_mod)
        finally:
            self._env_stacks[side].pop()
            self._unfolding.discard(app_key)

    def _walk_unfolded(self, entry, side, other, other_mod) -> bool:
        """The generic's body takes the TypeApp's side in the walk."""
        body, home = entry.ast_node, entry.container_module
        if side == "sub":
            return self._walk(body, home, other, other_mod)
        return self._walk(other, other_mod, body, home)

    def _applied_meaning(self, app_key, node, module, side):
        """TypeApp -> the generic's body as an AstNodeType whose
        env_get binds formal parameters to the actual arguments."""
        if app_key in self._unfolded:
            return self._unfolded[app_key]
        target = self._constructor_target(node, module, side)
        if not isinstance(target, AstNodeType):
            return None
        defn = target.ast_node
        params = getattr(defn, "generic_params", None)
        if not params or len(params) != len(node.args):
            return None
        env_get = self._binder(params, node.args, module, side)
        entry = AstNodeType(defn.body, target.container_module, env_get)
        self._unfolded[app_key] = entry
        return entry

    def _constructor_target(self, node, module, side):
        if node.constructor == "AssertionFailed":
            return None  # poison stays nominal: a sub witness never seats
        resolved = self._resolve_name(node.constructor, module, side)
        if isinstance(resolved, Err):
            raise UnresolvedTypeError(f"unresolvable constructor {node.constructor!r}")
        return resolved.value

    def _binder(self, params, args, module, side):
        meanings = {}
        for name, arg in zip(params, args):
            meanings[name] = self._meaning(arg, module, side)
        def env_get(name):
            return meanings.get(name, Err(f"unbound parameter {name!r}"))
        return env_get

    def _meaning(self, arg, module, side) -> Result:
        """Eager meaning of an actual argument. A TypeRef actual is
        resolved now (lexical scope); a leaf (constant/nil/never)
        becomes its literal Type; anything else stays structural."""
        if isinstance(arg, viba_ast.TypeRef):
            return self._resolve_name(arg.name, module, side)
        return Ok(_as_leaf(AstNodeType(arg, module)))

    def _app_key(self, node, module, side):
        """Stable identity for an unfolding: same node, same resolved
        actuals -> same entry, so recursion hits the assumption table."""
        parts = []
        for arg in node.args:
            meaning = self._meaning(arg, module, side)
            part = _type_key(meaning.value) if isinstance(meaning, Ok) else ("err",)
            parts.append(part)
        return (id(node), side, id(module), tuple(parts))

    def _walk_lifted(self, sn, s_mod, sp, p_mod):
        """Handle sides that lift to the Type level (Constants, TypeRefs)."""
        sub_t = self._lift(sn, s_mod, "sub")
        sup_t = self._lift(sp, p_mod, "sup")
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
            tagged, _ = self._split_product(sn, s_mod, "sub")
            return sp.tag in tagged and self._walk(tagged[sp.tag], s_mod, sp.type, p_mod)
        if not isinstance(sn, viba_ast.Tagged):
            return self._walk(sn, s_mod, sp.type, p_mod)
        return sn.tag == sp.tag and self._walk(sn.type, s_mod, sp.type, p_mod)

    def _walk_products(self, sn, s_mod, sp, p_mod) -> bool:
        """Tagged products match by tag (commutative); sup's tags must
        all be present in sub (width). Untagged elements pair in order.
        Bare TypeRefs to plain TypeDefinitions unfold so their tags
        participate (B := A * $find bool carries A's tags)."""
        sub_tagged, sub_bare = self._split_product(sn, s_mod, "sub")
        sup_tagged, sup_bare = self._split_product(sp, p_mod, "sup")
        if not self._tags_covered(sub_tagged, s_mod, sup_tagged, p_mod):
            return False
        if len(sub_bare) != len(sup_bare):
            return False
        pairs = zip(sub_bare, sup_bare)
        return all(self._walk(a, s_mod, b, p_mod) for a, b in pairs)

    def _split_product(self, node, module, side: str):
        """Flatten a Product/ProductChain into ({tag: body}, [bare]),
        unfolding transparent TypeDefinitions along the way."""
        tagged, bare = {}, []
        for elem in _product_elements(node):
            t, b = self._split_element(elem, module, side)
            tagged.update(t)
            bare.extend(b)
        return tagged, bare

    def _split_element(self, elem, module, side: str):
        elem, elem_mod = self._unfold_ref(elem, module, side)
        if isinstance(elem, _PROD_NODES):
            return self._split_product(elem, elem_mod, side)
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
        order; a longer sub argument list is a subtype of a shorter.
        Argument walks swap modules, so the env stacks swap with them:
        a free name resolves through the env of the syntax it came from."""
        if not isinstance(sn, _EXP_NODES):
            return False
        sub_res, sub_args = _exponent_parts(sn)
        sup_res, sup_args = _exponent_parts(sp)
        if len(sub_args) < len(sup_args):
            return False
        if not self._walk(sub_res, s_mod, sup_res, p_mod):
            return False
        self._swap_envs()
        try:
            pairs = zip(sup_args, sub_args)
            walks = (self._walk(sa, p_mod, sb, s_mod) for sa, sb in pairs)
            return all(walks)
        finally:
            self._swap_envs()

    def _swap_envs(self):
        stacks = self._env_stacks
        stacks["sub"], stacks["sup"] = stacks["sup"], stacks["sub"]

    # Literal containers: ListLiteral[a, b, c] is a resident of
    # list[a | b | c]; SetLiteral likewise; DictLiteral[(k, v), ...]
    # of dict[k_union, v_union]. Containment, not nominal equality.
    _LITERAL_OF = {"list": "ListLiteral", "set": "SetLiteral", "dict": "DictLiteral"}

    def _walk_typeapps(self, sn, s_mod, sp, p_mod) -> bool:
        sub_c = self._resolve_constructor(sn.constructor, s_mod, "sub")
        sup_c = self._resolve_constructor(sp.constructor, p_mod, "sup")
        if self._literal_resident(sn, s_mod, sub_c, sp, p_mod, sup_c):
            return True
        if not self._generic_equal(sub_c, sup_c):
            return self._unequal_typeapps(sn, s_mod, sp, p_mod)
        if len(sn.args) != len(sp.args):
            return False
        pairs = zip(sn.args, sp.args)
        return all(self._walk(a, s_mod, b, p_mod) for a, b in pairs)

    def _unequal_typeapps(self, sn, s_mod, sp, p_mod) -> bool:
        """Constructors differ: only transparent generics (identity
        bodies, e.g. Metric[T] := T) unfold. User generics stay
        nominal — same shape from different origins is not a match."""
        if self._is_transparent(sn, s_mod, "sub"):
            return self._unfold_typeapp(sn, s_mod, "sub", sp, p_mod)
        if self._is_transparent(sp, p_mod, "sup"):
            return self._unfold_typeapp(sp, p_mod, "sup", sn, s_mod)
        return False

    def _is_transparent(self, node, module, side) -> bool:
        """An identity generic: Metric[T] := T. Substitution lives in
        env_get; the body carries no structure of its own."""
        resolved = self._resolve_name(node.constructor, module, side)
        if not isinstance(resolved, Ok):
            return False
        target = resolved.value
        if not isinstance(target, AstNodeType):
            return False
        defn = target.ast_node
        if not isinstance(defn, viba_ast.GenericDefinition):
            return False
        params = defn.generic_params or []
        body = defn.body
        identity = isinstance(body, viba_ast.TypeRef) and body.name
        return len(params) == 1 and identity == params[0]

    def _literal_resident(self, sn, s_mod, sub_c, sp, p_mod, sup_c) -> bool:
        """A *Literal constructor against its container: every literal
        element must fit the container's element union."""
        if not isinstance(sub_c, BuiltinGenericType):
            return False
        if not isinstance(sup_c, BuiltinGenericType):
            return False
        if self._LITERAL_OF.get(sup_c.name) != sub_c.name:
            return False
        if sup_c.name == "dict":
            return self._dict_literal_resident(sn, s_mod, sp, p_mod)
        if len(sp.args) != 1:
            return False
        elem = sp.args[0]
        return all(self._walk(a, s_mod, elem, p_mod) for a in sn.args)

    def _dict_literal_resident(self, sn, s_mod, sp, p_mod) -> bool:
        if len(sp.args) != 2:
            return False
        key_t, val_t = sp.args
        good = (self._dict_pair(a, s_mod, key_t, val_t, p_mod) for a in sn.args)
        return all(good)

    def _dict_pair(self, pair, s_mod, key_t, val_t, p_mod) -> bool:
        if not isinstance(pair, viba_ast.Tuple) or len(pair.elements) != 2:
            return False
        k, v = pair.elements
        keys_ok = self._walk(k, s_mod, key_t, p_mod)
        vals_ok = self._walk(v, s_mod, val_t, p_mod)
        return keys_ok and vals_ok

    def _resolve_constructor(self, name: str, module: ModuleType, side: str):
        resolved = self._resolve_name(name, module, side)
        if isinstance(resolved, Err):
            raise UnresolvedTypeError(f"unresolvable constructor {name!r}")
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


def _as_leaf(t: Type) -> Type:
    """An AstNodeType wrapping a leaf node IS that leaf: constants,
    nil and never carry no references, so the wrapper adds nothing."""
    if not isinstance(t, AstNodeType):
        return t
    node = t.ast_node
    if isinstance(node, viba_ast.Nil):
        return NilType()
    if isinstance(node, viba_ast.Never):
        return NeverType()
    if isinstance(node, viba_ast.Constant):
        return _literal_type(node.value)
    return t


def _unwrap_definition(entry: AstNodeType):
    """Plain TypeDefinitions are transparent: compare their bodies.
    GenericDefinitions stay wrapped (they are nominal)."""
    node = entry.ast_node
    if isinstance(node, viba_ast.TypeDefinition):
        return node.body, entry.container_module
    return node, entry.container_module


def _is_poison_ref(node) -> bool:
    if isinstance(node, viba_ast.TypeRef):
        return node.name == "AssertionFailed"
    ctor = getattr(node, "constructor", None)
    return ctor == "AssertionFailed"


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


def _type_key(t: Type, env: tuple = ()):
    """Canonical comparison key for the memo table. `env` is the id
    tuple of the env_get stack the type is judged under: the same
    node resolves differently under different free-name bindings."""
    if isinstance(t, AstNodeType):
        defn = _as_definition(t.ast_node)
        node_key = ("defn", defn.name) if defn is not None else ("inline", id(t.ast_node))
        return ("ast", id(t.container_module), node_key, env)
    if isinstance(t, BuiltinGenericType):
        return ("generic", t.name)
    if isinstance(t, _LITERAL_TYPES):
        return (type(t).__name__, t.value)
    return (type(t).__name__,)
