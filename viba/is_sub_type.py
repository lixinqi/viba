"""Subtype judgment over the Type model: is_sub_type(sub, sup) -> Result[bool].

Consumes Type values only (viba.type); nothing above the type level is
visible here.

Ok(True/False) is the judgment. Err reports malformed input, never a
judgment: ellipsis (...) anywhere on either side -> Err (an open type
has no judgment).

Semantics (per design):
- GenericDefinition references are nominal: same container module
  (object identity) and same name -> equal; bodies never unfold.
- TypeDefinition references are transparent: they unfold to their
  bodies and compare structurally (B := A * $find bool <: A).
  Recursive plain definitions are guarded by the coinductive
  assumption table: a (sub, sup) pair already in flight is true.
- Leaves compare by family: literal(v) <: base iff same family;
  literal <: literal iff equal values; never <: T; T <: never iff
  T is never (only never fits a never branch).
- AstNodeType wrapping inline structure (sums, products, tuples,
  exponents, type applications) is compared structurally, recursing
  at the Type level whenever a side lifts to a Type (TypeRef
  resolution, constants).
- TypeRef resolves through module_get_type in its own container
  module (lexical scoping); when that fails, the env_get bindings of
  the side's AstNodeType stack are consulted, innermost first (this
  is how free names — e.g. generic parameters — get their meanings).
  A TypeRef the judgment actually reaches and cannot resolve by
  either channel is malformed: the check aborts with Err
  (UnresolvedTypeError caught at the boundary). Generic definition
  bodies never trigger this — they are nominal and never unfold.
- not[A] is never <- A. A sub not[B] with the same branch tags and the
  poison PredicationFailed at every branch is the refutation: the
  not[oneof] shell is preserved, only the leaves are swapped. A sub
  written as an exponent (never <- B) compares by the exponent case —
  never <- B <: never <- A iff A <: B. A sub written as a tagged
  product is read through never <- (A | B) = (never <- A) * (never <- B),
  so the slot at every branch tag must carry a refutation: the poison
  PredicationFailed, or a type that fits never <- that branch.
- Applied generics (TypeApp): both sides applied stays nominal —
  same constructor and pairwise actuals. Exactly one side applied
  unfolds structurally: the generic's body is compared with formal
  parameters bound to the actuals through env_get (a TypeRef actual
  resolves eagerly in its lexical scope). Unfoldings are cached and
  guarded by a coinductive assumption table keyed on the node and
  its resolved actuals, so recursive generics terminate.
- Literal containers: ListLiteral[a, b, c] is a resident of
  list[a | b | c] (containment: every element fits the union);
  SetLiteral likewise; DictLiteral[(k, v), ...] of
  dict[k_union, v_union]. The reverse is False, and so is
  mixing container families.
- The memo table maps (sub_key, sup_key) to the result in flight;
  under nominal semantics no cycle can require assuming a pair, so
  it is a pure memoization / shared-subgraph guard.
"""

from viba import viba_ast
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
    """Ok(True/False) is the judgment; Err reports malformed input."""
    err = _input_error(sub, sup)
    if err is not None:
        return Err(err)
    try:
        return Ok(_Checker().check(sub, sup))
    except UnresolvedTypeError as exc:
        return Err(str(exc))


def _input_error(sub: Type, sup: Type):
    """Ellipsis anywhere is malformed input, not a judgment."""
    for label, side in (("sub", sub), ("sup", sup)):
        if not isinstance(side, AstNodeType):
            continue
        nodes = viba_ast.walk(side.ast_node)
        if any(isinstance(n, viba_ast.Ellipsis) for n in nodes):
            return f"ellipsis is not allowed on the {label} side"
    return None


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
        An unresolvable TypeRef is malformed."""
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
        return resolved.ok_value

    def _check_uncached(self, sub: Type, sup: Type) -> bool:
        if isinstance(sub, NeverType):
            return True
        if isinstance(sup, (NeverType, NilType)):
            return type(sub) is type(sup)
        result = self._probe_leaves(sub, sup)
        if result is not None:
            return result
        if isinstance(sup, AstNodeType) and isinstance(sub, AstNodeType):
            return self._check_ast_pair(sub, sup)
        return False

    def _probe_leaves(self, sub: Type, sup: Type):
        probes = (self._check_base, self._check_literal, self._check_nominal)
        results = (probe(sub, sup) for probe in probes)
        return next((r for r in results if r is not None), None)

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
    # Structural walk over viba.viba_ast nodes
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
        """A TypeRef is transparent unless it names a generic: it unfolds
        to the bound body, whose own TypeRefs resolve in its home module.
        A generic parameter bound to inline structure (sum, product,
        exponent, tag, tuple) unfolds to that structure."""
        if not isinstance(node, viba_ast.TypeRef):
            return node, module
        resolved = self._resolve_name(node.name, module, side)
        if isinstance(resolved, Err):
            return node, module
        target = resolved.ok_value
        if not isinstance(target, AstNodeType):
            return node, module
        body = target.ast_node
        if isinstance(body, viba_ast.GenericDefinition):
            return node, module
        if isinstance(body, viba_ast.TypeDefinition):
            body = body.body
        return body, target.container_module

    def _walk_inner(self, sn, s_mod: ModuleType, sp, p_mod: ModuleType) -> bool:
        if isinstance(sn, viba_ast.Never):
            return True  # bottom fits anywhere
        if isinstance(sp, (viba_ast.Nil, viba_ast.Never)):
            # 叶子对叶子：写名字（含泛型形参）也要认出来，名字是透明的。
            sub_leaf = self._lift(sn, s_mod, "sub")
            sup_leaf = self._lift(sp, p_mod, "sup")
            if sub_leaf is not None and sup_leaf is not None:
                return type(sub_leaf) is type(sup_leaf)
            return type(sn) is type(sp)
        operand = self._prohibition_operand(sp, p_mod)
        if operand is not None:
            return self._walk_prohibition(sn, s_mod, sp, p_mod, operand)
        mixed = self._walk_mixed_typeapp(sn, s_mod, sp, p_mod)
        if mixed is not None:
            return mixed
        lifted = self._walk_lifted(sn, s_mod, sp, p_mod)
        if lifted is not None:
            return lifted
        return self._walk_structural(sn, s_mod, sp, p_mod)

    # ------------------------------------------------------------------
    # 禁止 = never 头的指数链（`not[A]` 的定义体就是它，`never <- …` 是
    # 直接写出来）。判据是形状，不是名字：三种链一个读法，指向定义时先
    # 看定义体的形状。
    # - 外壳：两边都是禁止，分支 tag 对得上，且 sub 的每个分支都是毒剂
    #   PredicationFailed。
    # - 逐分支否证：sub 写成带 tag 的积时，按
    #   never <- (A | B) = (never <- A) * (never <- B)，每个分支 tag 都要
    #   带上自己的否证（毒剂，或者一个能入席 never <- 那一支的类型），
    #   缺一支就不算。
    # - 其余落到普通的指数规则：never <- B <: never <- A 当且仅当 A <: B。
    # ------------------------------------------------------------------

    def _walk_prohibition(self, sn, s_mod, sp, p_mod, operand) -> bool:
        sub_operand = self._prohibition_operand(sn, s_mod)
        if sub_operand is not None and self._prohibition_shell(sub_operand, operand):
            return True
        if self._prohibition_evidence(sn, s_mod, operand):
            return True
        node, _ = self._unfold_ref(sn, s_mod, "sub")
        if isinstance(sp, viba_ast.TypeApp):
            if isinstance(node, viba_ast.TypeApp):
                # 两边都写成应用（外壳）：只认全毒剂的外壳，不是指数读法。
                return False
            # 应用形式：展开到定义体（形参代实参）再按指数规则比。
            return self._unfold_typeapp(sp, p_mod, "sup", sn, s_mod)
        mixed = self._walk_mixed_typeapp(sn, s_mod, sp, p_mod)
        if mixed is not None:
            return mixed
        return self._walk_exponent(sn, s_mod, sp, p_mod)

    def _prohibition_operand(self, node, module):
        """(操作数节点, 模块) when this段写成禁止，否则 None。

        应用形式看构造子的定义体（体是 never 头的指数链，操作数是某个形参）；
        指数形式看链本身：首元 never、第二元是操作数。
        """
        operand = self._application_operand(node, module)
        if operand is not None:
            return operand
        return self._exponent_operand(node, module)

    def _application_operand(self, node, module):
        if not isinstance(node, viba_ast.TypeApp):
            return None
        resolved = self._resolve_name(node.constructor, module, "sup")
        if isinstance(resolved, Err):
            return None
        target = resolved.ok_value
        if not (isinstance(target, AstNodeType)
                and isinstance(target.ast_node, viba_ast.GenericDefinition)):
            return None
        definition = target.ast_node
        params = list(definition.generic_params or [])
        if len(params) != len(node.args):
            return None
        body = viba_ast.convert_to_chain_style(definition.body)
        if not isinstance(body, _EXP_NODES):
            return None
        elements = _exponent_elements(body)
        if len(elements) != 2 or not isinstance(elements[0], viba_ast.Never):
            return None
        index = _param_index(elements[1], params)
        if index is None:
            return None
        return node.args[index], module

    def _exponent_operand(self, node, module):
        if not isinstance(node, _EXP_NODES):
            return None
        elements = _exponent_elements(node)
        if len(elements) != 2 or not isinstance(elements[0], viba_ast.Never):
            return None
        operand = elements[1]
        return (operand.type if isinstance(operand, viba_ast.Tagged) else operand), module

    def _prohibition_shell(self, sub_operand, sup_operand) -> bool:
        """两边都是禁止：分支 tag 对得上，且 sub 的每个分支都是毒剂。"""
        sub_node, sub_mod = sub_operand
        sup_node, sup_mod = sup_operand
        sup_branches = self._sum_branches(sup_node, sup_mod)
        sub_branches = self._sum_branches(sub_node, sub_mod)
        if sup_branches is None or sub_branches is None:
            return False
        sup_tags = sorted(tag for tag, _, _ in sup_branches)
        sub_tags = sorted(tag for tag, _, _ in sub_branches)
        if sup_tags != sub_tags:
            return False
        return all(self._is_poison(branch, mod) for _, branch, mod in sub_branches)

    def _is_poison(self, node, module) -> bool:
        node, module = self._unfold_ref(node, module, "sub")
        if _is_predication_failed(node):
            return True
        return isinstance(node, viba_ast.Never)

    def _prohibition_evidence(self, sn, s_mod, sup_operand) -> bool:
        branches = self._sum_branches(sup_operand[0], sup_operand[1])
        if branches is None:
            return False
        fields = self._tagged_fields(sn, s_mod)
        return all(self._branch_evidence(fields.get(tag), b_type, b_mod)
                   for tag, b_type, b_mod in branches)

    def _branch_evidence(self, field, branch_type, branch_mod) -> bool:
        """A branch is refuted by the poison PredicationFailed, or by a
        field that reads as never <- that branch."""
        if field is None:
            return False
        node, module = field
        node, module = self._unfold_ref(node, module, "sub")
        if self._is_poison(node, module):
            return True
        meaning = self._function_argument(node, module)
        if meaning is None:
            return False
        sub_arg, sub_mod = meaning
        self._swap_envs()
        try:
            return self._walk(branch_type, branch_mod, sub_arg, sub_mod)
        finally:
            self._swap_envs()

    def _function_argument(self, node, module):
        """(argument, module) when node reads as never <- argument: a
        prohibition application, or an exponent whose result is never."""
        node, module = self._unfold_ref(node, module, "sub")
        return self._prohibition_operand(node, module)

    def _sum_branches(self, node, module):
        """[(tag, body, module)] for a not argument; None when a branch
        is not tagged, since only a tag can hold a never-arrow."""
        out = []
        stack = [(node, module)]
        while stack:
            current, current_mod = stack.pop()
            current, current_mod = self._unfold_ref(current, current_mod, "sup")
            if isinstance(current, viba_ast.Sum):
                stack.append((current.right, current_mod))
                stack.append((current.left, current_mod))
            elif isinstance(current, viba_ast.SumChain):
                stack.extend((e, current_mod) for e in reversed(current.elements))
            elif isinstance(current, viba_ast.Tagged):
                out.append((current.tag, current.type, current_mod))
            else:
                return None
        return out

    def _tagged_fields(self, node, module):
        """{tag: (body, module)} over a product of tags, unfolding named
        definitions and named products along the way."""
        node, module = self._unfold_ref(node, module, "sub")
        if isinstance(node, viba_ast.Tagged):
            return {node.tag: (node.type, module)}
        if isinstance(node, viba_ast.Product):
            fields = self._tagged_fields(node.left, module)
            fields.update(self._tagged_fields(node.right, module))
            return fields
        if isinstance(node, viba_ast.ProductChain):
            fields = {}
            for element in node.elements:
                fields.update(self._tagged_fields(element, module))
            return fields
        return {}

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
        target = resolved.ok_value
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
        resolved = self._resolve_name(node.constructor, module, side)
        if isinstance(resolved, Err):
            raise UnresolvedTypeError(f"unresolvable constructor {node.constructor!r}")
        return resolved.ok_value

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
            part = _type_key(meaning.ok_value) if isinstance(meaning, Ok) else ("err",)
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
        bodies, e.g. Id[T] := T) unfold. Other generics stay nominal —
        same shape from different origins is not a match."""
        if self._is_transparent(sn, s_mod, "sub"):
            return self._unfold_typeapp(sn, s_mod, "sub", sp, p_mod)
        if self._is_transparent(sp, p_mod, "sup"):
            return self._unfold_typeapp(sp, p_mod, "sup", sn, s_mod)
        return False

    def _is_transparent(self, node, module, side) -> bool:
        """An identity generic: Id[T] := T. Substitution lives in
        env_get; the body carries no structure of its own."""
        resolved = self._resolve_name(node.constructor, module, side)
        if not isinstance(resolved, Ok):
            return False
        target = resolved.ok_value
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
        return resolved.ok_value

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


def _is_predication_failed(node) -> bool:
    """The poison: a failed predication, named by its constructor. It
    is the required refutation evidence at a not branch."""
    return isinstance(node, viba_ast.TypeApp) and node.constructor == "PredicationFailed"


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


def _param_index(node, params) -> "Optional[int]":
    """禁止的操作数写着哪个形参（可能带一层标签）；没有就 None。"""
    if isinstance(node, viba_ast.Tagged):
        node = node.type
    if isinstance(node, viba_ast.TypeRef) and node.name in params:
        return list(params).index(node.name)
    return None


def _exponent_elements(node):
    """指数的元素表（书写顺序）：首元是结果，其余是参数。

    链与二元写法一样：``A <- B <- C`` 与 ``(A <- B) <- C`` 都是 [A, B, C]；
    ``A <- (B <- C)`` 是 [A, [B, C]]，支链算一个元素。
    """
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _exponent_elements(node.result) + [node.argument]
    return [node]


def _exponent_parts(node):
    """指数读成 (结果, 参数表)：结果在前，参数按应用顺序（最右边的先喂）。

    ``A <- B <- C`` 读成 (A, [B, C])——先喂 B 再喂 C。
    """
    elements = _exponent_elements(node)
    if not isinstance(node, _EXP_NODES):
        raise TypeError(f"not an exponent: {node!r}")
    return elements[0], list(reversed(elements[1:]))


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
