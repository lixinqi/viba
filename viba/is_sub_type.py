"""Subtype judgment over the Type model: is_sub_type(sub, sup) -> Result[bool].

Consumes Type values only (viba.type); nothing above the type level is
visible here.

Ok(True/False) is the judgment. VibaProgramErr reports malformed input, never a
judgment: ellipsis (...) anywhere on either side -> VibaProgramErr (an open type
has no judgment), and a product that writes the same tag twice ->
VibaProgramErr.

Semantics (per design): there are no nominal types — a name is an
alias of what it is written as, and judgment is structural throughout.
- Every definition unfolds to its body and is compared structurally
  (B = A * $find bool <: A, and X[T] = list[T] gives X[int] the
  body of list[int]). A definition reference that cannot be unfolded
  — a bare generic name, whose parameters have no actuals — compares
  by name, which is all that is left of it.
- Cycles are read coinductively (equi-recursive types), and the
  assumption is the greatest fixed point: a (sub, sup) pair already in
  flight is true. `Tree[T] = $leaf T * $kids list[Tree[T]]` is
  therefore its own unfolding, `MyList[T] = $head T * $tail MyList[T]
  | nil` and the same type under another name are each other's
  subtype, and a definition that reaches only itself (`Loop[T] =
  Loop[T]`) is the largest type: it is both a subtype and a supertype
  of anything it is compared with: whoever writes such a definition
  gives their design a type everything fits.
- Leaves compare by family: literal(v) <: base iff same family;
  literal <: literal iff equal values; never <: T; T <: never iff
  T is never (only never fits a never branch).
- AstNodeType wrapping inline structure (sums, products, tuples,
  exponents, type applications) is compared structurally, recursing
  at the Type level whenever a side lifts to a Type (TypeRef
  resolution, constants).
- Products compose by inlining: an untagged member that unfolds to a
  product contributes that product's own members, recursively, and an
  untagged member that unfolds to the product unit (nil, or a name
  bound to it) contributes nothing — `A * $z T` carries A's tags, and
  `Object * $a int` is `$a int` in both directions. Any other
  untagged member is one positional member, paired with the other
  side's positionals in order. Tags hold the members together, so the
  same tag twice in one product (inlined or written) is malformed
  input: VibaProgramErr. So is a chain that never reaches a body because it
  comes back to a definition it is already expanding — `A = A * $x
  int` writes A as itself, and an alias or a generic can close the
  same loop.
- TypeRef resolves through module_get_type in its own container
  module (lexical scoping); when that fails, the env_get bindings of
  the side's AstNodeType stack are consulted, innermost first (this
  is how free names — e.g. generic parameters — get their meanings).
  A TypeRef the judgment actually reaches and cannot resolve by
  either channel is malformed: the check aborts with VibaProgramErr
  (UnresolvedTypeError caught at the boundary). Unfolding a definition
  body can reach such a name; that is then a VibaProgramErr, not a False.
- Functions (exponent chains) compare as functions: the result
  covariantly, the arguments contravariantly, position by position
  from $arg0. Only functions compare with functions (never, the
  bottom, fits anywhere and is handled before this). The sub is first
  brought to the sup's arity: shorter it is padded with never at its
  end, longer it is cut, the sup never moving. So
  (int <- $a int <- $b str) <: (int <- $a int) holds (the extra
  argument is one the sup never asks about) and (int <- $a int) <:
  (int <- $a int <- never) holds too (the never it is padded with is
  what the sup asks for), while (int <- $a int) <:
  (int <- $a int <- $b bool) does not: padding never does not answer a
  bool.
- A chain whose result is never (`never <- A`) is an exponent like any
  other: result covariant, argument contravariant, so never <- B <:
  never <- A iff A <: B. When the caller named terminators, two
  readings of its branches apply on top: a branch is matched by a
  terminator, or by a field that reads as `never <- branch`, and a sub
  that only copies the application matches nothing. With no terminators
  named, the exponent rule is all there is and never <- B <: never <- B
  holds.
- Literal containers: ListLiteral[a, b, c] is a resident of
  list[a | b | c] (containment: every element fits the union);
  SetLiteral likewise; DictLiteral[(k, v), ...] of
  dict[k_union, v_union]. The reverse is False, and so is
  mixing container families.
- The memo table maps (sub_key, sup_key) to the result in flight; the
  walk's own (node pair, module pair, env pair) table is the
  coinductive assumption above it.
"""

from viba import viba_ast
from viba.partial import (by_need_type, module_as_function, product_elements,
                          reduce_partial)
from viba.type import (
    PartialError,
    AnyType,
    AstNodeType,
    BoolLiteralType,
    BoolType,
    BuiltinGenericType,
    DuplicateTagError,
    VibaProgramErr,
    FloatLiteralType,
    FloatType,
    InlineCycleError,
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


def is_sub_type(sub: Type, sup: Type, terminators=frozenset(), config=None) -> Result:
    """Ok(True/False) is the judgment; VibaProgramErr reports malformed input.

    `terminators` names the written types a never-headed chain accepts
    where `never` itself would do. The core is told, not told about: which
    word that is belongs to the layer that uses the chain.

    `config` is the same value the address layer reads with
    (`VibaReflectConfig`: `never_eqv` and `nil_eqv`, the written names that
    stand for the units). A name in either set is that unit here too, bare
    or applied — `U[{...}]` is the unit when `U` is one — so a layer that
    parks documentation in such a block gets it bypassed instead of
    resolved. Such an argument is also not a position: documentation drops
    out of a function's argument list, so `A <- B <- U[{...}]` and
    `A <- U[{...}] <- B` are both `A <- B`. A unit written as a plain name
    does keep its position (it is the nil of the product, and nil is a real
    slot in a tuple). Nothing named, nothing bypassed.
    """
    err = _input_error(sub, sup)
    if err is not None:
        return VibaProgramErr(err)
    try:
        return Ok(_Checker(terminators, config).check(sub, sup))
    except UnresolvedTypeError as exc:
        return VibaProgramErr(str(exc))
    except PartialError as exc:
        return VibaProgramErr(str(exc))
    except DuplicateTagError as exc:
        return VibaProgramErr(str(exc))
    except InlineCycleError as exc:
        return VibaProgramErr(str(exc))


def _input_error(sub: Type, sup: Type):
    """Ellipsis anywhere is malformed input, not a judgment."""
    for label, side in (("sub", sub), ("sup", sup)):
        if not isinstance(side, AstNodeType):
            continue
        nodes = viba_ast.walk(side.ast_node)
        if any(isinstance(n, viba_ast.Ellipsis) for n in nodes):
            return f"ellipsis is not allowed on the {label} side"
    return None


# The written units a config can name: shared, so a substituted unit keeps one
# node and the memo keys stay put.
_UNIT_NIL = viba_ast.Nil()
_UNIT_NEVER = viba_ast.Never()


def _has_code_block(node) -> bool:
    """A {...} somewhere under this piece, behind tags and applications too."""
    return any(isinstance(part, viba_ast.CodeBlock) for part in viba_ast.walk(node))


class _Checker:
    def __init__(self, terminators=frozenset(), config=None):
        self.terminators = frozenset(terminators)
        self.nil_eqv = frozenset(getattr(config, "nil_eqv", ()) or ())
        self.never_eqv = frozenset(getattr(config, "never_eqv", ()) or ())
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
        """Module definitions win; a product member comes next; the side's
        env_get supplements (innermost binding first) — the free-variable
        channel."""
        resolved = module_get_type(module, name)
        if isinstance(resolved, Ok):
            return resolved
        member = self._product_member_type(name, module, side)
        if member is not None:
            return member
        envs = reversed(self._env_stacks[side])
        hits = (e(name) for e in envs)
        return next((h for h in hits if isinstance(h, Ok)), resolved)

    def _product_member_type(self, name: str, module: ModuleType, side: str):
        """`point.x` read as a type: the `$x` member's declared type.

        A member of a product is addressed by its tag, and that is what a dotted
        name does — `__args__.a` is `int` when the module declares
        `__args__ = Object * $a int`. The value layer reads the same name as the
        argument the call was given (viba-interpreter.md): one name, two layers,
        what it denotes each time. None when the head is no product.
        """
        head, dot, tag = name.rpartition(".")
        if not dot or not head:
            return None
        body, home = self._head_body(head, module, side)
        if body is None:
            return None
        for factor in product_elements(body):
            if isinstance(factor, viba_ast.Tagged) and factor.tag == "$" + tag:
                return Ok(AstNodeType(factor.type, home))
        return None

    def _head_body(self, name: str, module: ModuleType, side: str):
        """(body, home) for the name, following aliases to the end."""
        seen = set()
        while name not in seen:
            seen.add(name)
            resolved = self._resolve_name(name, module, side)
            if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
                return None, module
            entry = resolved.ok_value
            node = entry.ast_node
            if isinstance(node, viba_ast.TypeDefinition):
                node = node.body
            if not isinstance(node, viba_ast.TypeRef):
                return node, entry.container_module
            name, module = node.name, entry.container_module
        return None, module

    def _lift(self, node, module: ModuleType, side: str):
        """Lift a Constant, TypeRef, Nil or Never to a Type; else None.
        An unresolvable TypeRef is malformed."""
        if isinstance(node, viba_ast.Constant):
            return _literal_type(node.value)
        if isinstance(node, viba_ast.Nil):
            return NilType()
        if isinstance(node, viba_ast.Never):
            return NeverType()
        # Any is not lifted: what it is *below* depends on the type on the
        # other side (`Any * nil` is Any, `Any | int` is Any), so the walk
        # decides it. AnyType serves the Type-level API, and the top rule
        # (`sup is AnyType`: everything fits) is what the walk asks first.
        if isinstance(node, viba_ast.TypeRef):
            return self._lift_ref(node, module, side)
        if isinstance(node, (viba_ast.Let, viba_ast.Binding)):
            # `:=` belongs to computation; what a definition judges is a type.
            # A binding here is misplaced, not a judgment this layer owes
            # (viba-style.md).
            raise UnresolvedTypeError(
                "a binding is computation, not a type: "
                f"{viba_ast.unparse_type(node)}")
        return None

    def _lift_ref(self, node, module: ModuleType, side: str):
        resolved = self._resolve_name(node.name, module, side)
        if isinstance(resolved, VibaProgramErr):
            raise UnresolvedTypeError(f"unresolvable TypeRef {node.name!r}")
        return resolved.ok_value

    def _check_uncached(self, sub: Type, sup: Type) -> bool:
        if isinstance(sub, NeverType):
            return True
        if isinstance(sup, AnyType):
            return True                     # Any is the top: everything fits
        if isinstance(sub, AnyType):
            return isinstance(sup, AnyType)     # Any is only below Any
        if isinstance(sup, (NeverType, NilType)):
            return type(sub) is type(sup)
        result = self._probe_leaves(sub, sup)
        if result is not None:
            return result
        if isinstance(sup, AstNodeType) and isinstance(sub, AstNodeType):
            return self._check_ast_pair(sub, sup)
        if isinstance(sup, AstNodeType) and isinstance(sup.ast_node, viba_ast.CodeBlock):
            return isinstance(sub, NilType)     # 代码块没有成员，单位是它的居民
        return False

    def _probe_leaves(self, sub: Type, sup: Type):
        probes = (self._check_base, self._check_literal, self._check_builtin_name)
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

    def _check_builtin_name(self, sub: Type, sup: Type):
        """A builtin container with no arguments is its name."""
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
        """Two bare generic names: the same name, and nothing else.

        A generic with no arguments has no parameters bound, so there is no
        body to unfold; the name is all that can be compared. Which module it
        was written in is how it was resolved, not part of the type.
        """
        return sn.name == sp.name

    # ------------------------------------------------------------------
    # Structural walk over viba.viba_ast nodes
    # ------------------------------------------------------------------

    def _walk(self, sn, s_mod: ModuleType, sp, p_mod: ModuleType) -> bool:
        # A unit a config names is bypassed, not resolved: the name alone is
        # what the layer said it is, so it answers before anything unfolds.
        sn = self._config_unit_node(sn)
        sp = self._config_unit_node(sp)
        # Names unfold, `<<` is given, and both can hand the other work (`A = B`
        # with `B = X << Y`): run them until neither has anything left.
        sn, s_mod = self._normalize(sn, s_mod, "sub")
        sp, p_mod = self._normalize(sp, p_mod, "sup")
        sn = self._config_unit_node(sn)
        sp = self._config_unit_node(sp)
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
        """A TypeRef is transparent unless it names a generic: it unfolds,
        name after name, to the bound body, whose own TypeRefs resolve in its
        home module. A generic parameter bound to inline structure (sum,
        product, exponent, tag, tuple) unfolds to that structure.

        A name whose body is another name is still an alias of that name's
        body (`A = B` with `B = int` makes A int), so the unfolding runs to
        the end of the chain. Names already visited pin the walk down, which
        leaves a cycle (`A = B` with `B = A`) standing as the name it is."""
        seen = set()
        while isinstance(node, viba_ast.TypeRef):
            if node.name in seen:
                return node, module
            seen.add(node.name)
            resolved = self._resolve_name(node.name, module, side)
            if isinstance(resolved, VibaProgramErr):
                return node, module
            target = resolved.ok_value
            if not isinstance(target, AstNodeType):
                return node, module
            body = target.ast_node
            if isinstance(body, viba_ast.GenericDefinition):
                return node, module
            if isinstance(body, viba_ast.TypeDefinition):
                body = body.body
            node, module = body, target.container_module
        return node, module

    def _walk_inner(self, sn, s_mod: ModuleType, sp, p_mod: ModuleType) -> bool:
        if isinstance(sn, viba_ast.Never):
            return True  # bottom fits anywhere
        if isinstance(sp, viba_ast.Any):
            # Any is the top: everything is its subtype. The other direction is
            # no special case: Any fits a sum that has an Any branch, or a
            # product whose other members are units, and nothing else - the
            # ordinary walk already says so.
            return True
        if isinstance(sp, (viba_ast.Nil, viba_ast.Never)):
            # 叶子对叶子：写名字（含泛型形参）也要认出来，名字是透明的。
            sub_leaf = self._lift(sn, s_mod, "sub")
            sup_leaf = self._lift(sp, p_mod, "sup")
            if sub_leaf is not None and sup_leaf is not None:
                return type(sub_leaf) is type(sup_leaf)
            if isinstance(sn, viba_ast.TypeApp):
                # 应用形的 sub 一样要展开：X[T] = nil 的居民就是 nil。
                return self._unfold_typeapp(sn, s_mod, "sub", sp, p_mod)
            if isinstance(sn, viba_ast.CodeBlock) and isinstance(sp, viba_ast.Nil):
                return True                     # 代码块是单位，nil 收得下
            return type(sn) is type(sp)
        if self.terminators:
            operand = self._never_head_operand(sp, p_mod)
            if operand is not None:
                return self._walk_never_head(sn, s_mod, sp, p_mod, operand)
        mixed = self._walk_mixed_typeapp(sn, s_mod, sp, p_mod)
        if mixed is not None:
            return mixed
        lifted = self._walk_lifted(sn, s_mod, sp, p_mod)
        if lifted is not None:
            return lifted
        return self._walk_structural(sn, s_mod, sp, p_mod)

    # ------------------------------------------------------------------
    # never-headed chains: an exponent chain whose result is never, either
    # written out (never <- $x T) or reached through a definition's body.
    # What is written is what counts, not the name.
    #
    # A chain reads as a function type, so its argument is contravariant.
    # Two further readings apply to branches, and both end in a terminator
    # the caller named (self.terminators):
    #   - settled: both sides are never-headed, the branch tags match, and
    #     every sub branch is a terminator;
    #   - per branch: through never <- (A | B) = (never <- A) * (never <- B),
    #     the field at every branch tag must be a terminator or read as
    #     never <- branch, whose argument is then compared contravariantly.
    # A sub that is only a copy of the never-headed application settles
    # nothing: the chain head is not a terminator. Everything else falls to the
    # ordinary exponent rule.
    # ------------------------------------------------------------------

    def _walk_never_head(self, sn, s_mod, sp, p_mod, operand) -> bool:
        sub_operand = self._never_head_operand(sn, s_mod)
        if sub_operand is not None and self._same_branches_settled(sub_operand, operand):
            return True
        if self._branches_settled(sn, s_mod, operand):
            return True
        node, _ = self._unfold_ref(sn, s_mod, "sub")
        if isinstance(sp, viba_ast.TypeApp):
            if isinstance(node, viba_ast.TypeApp):
                # Both sides written as applications: a copy settles
                # nothing, and it is not read as an exponent here.
                return False
            # Application form: unfold to the body and use the exponent rule.
            return self._unfold_typeapp(sp, p_mod, "sup", sn, s_mod)
        mixed = self._walk_mixed_typeapp(sn, s_mod, sp, p_mod)
        if mixed is not None:
            return mixed
        return self._walk_exponent(sn, s_mod, sp, p_mod)

    def _never_head_operand(self, node, module):
        """(operand node, module) when this piece is a never-headed chain."""
        operand = self._never_head_application(node, module)
        if operand is not None:
            return operand
        return self._never_head_exponent(node, module)

    def _never_head_application(self, node, module):
        if not isinstance(node, viba_ast.TypeApp):
            return None
        resolved = self._resolve_name(node.constructor, module, "sup")
        if isinstance(resolved, VibaProgramErr):
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

    def _never_head_exponent(self, node, module):
        if not isinstance(node, _EXP_NODES):
            return None
        elements = _exponent_elements(node)
        if len(elements) != 2 or not isinstance(elements[0], viba_ast.Never):
            return None
        operand = elements[1]
        return (operand.type if isinstance(operand, viba_ast.Tagged) else operand), module

    def _same_branches_settled(self, sub_operand, sup_operand) -> bool:
        """Both sides never-headed: same tags, every sub branch settled."""
        sub_branches = self._sum_branches(sub_operand[0], sub_operand[1])
        sup_branches = self._sum_branches(sup_operand[0], sup_operand[1])
        if sup_branches is None or sub_branches is None:
            return False
        if sorted(tag for tag, _, _ in sup_branches) != sorted(tag for tag, _, _ in sub_branches):
            return False
        return all(self._is_terminator(branch, mod) for _, branch, mod in sub_branches)

    def _is_terminator(self, node, module) -> bool:
        node, module = self._unfold_ref(node, module, "sub")
        if isinstance(node, viba_ast.Never):
            return True
        return (isinstance(node, viba_ast.TypeApp)
                and node.constructor in self.terminators)

    def _branches_settled(self, sn, s_mod, sup_operand) -> bool:
        branches = self._sum_branches(sup_operand[0], sup_operand[1])
        if branches is None:
            return False
        fields = self._tagged_fields(sn, s_mod)
        return all(self._branch_settled(fields.get(tag), b_type, b_mod)
                   for tag, b_type, b_mod in branches)

    def _branch_settled(self, field, branch_type, branch_mod) -> bool:
        """One branch is settled by a terminator at that tag, or by a field
        reading as never <- branch whose argument is a supertype of it."""
        if field is None:
            return False
        node, module = self._unfold_ref(field[0], field[1], "sub")
        if self._is_terminator(node, module):
            return True
        meaning = self._never_head_operand(node, module)
        if meaning is None:
            return False
        sub_arg, sub_mod = meaning
        self._swap_envs()
        try:
            return self._walk(branch_type, branch_mod, sub_arg, sub_mod)
        finally:
            self._swap_envs()

    def _sum_branches(self, node, module):
        """[(tag, body, module)] for the operand of a never-headed chain;
        None when a branch is untagged, since only a tag can hold one.

        A sum written as a cycle (`S = S | $a int`) has no finite branch list
        to read: the walk stops where it would meet a definition twice and
        answers None, which leaves the branches unsettled rather than spinning.
        """
        out = []
        seen = set()
        stack = [(node, module)]
        while stack:
            current, current_mod = stack.pop()
            target = self._inlining_key(current, current_mod, "sup")
            if target is not None:
                key = target[0]
                if key in seen:
                    return None
                seen = seen | {key}
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

    def _tagged_fields(self, node, module, seen=frozenset()):
        """{tag: (body, module)} over a product of tags, unfolding named
        definitions and named products along the way.

        The same tag twice is malformed input, here as everywhere, and so is an
        inline chain that comes back to a definition it is already walking: an
        untagged member is an inline slot, so `A = A * $x int` has no expansion
        to read (and asking for its fields must not spin).
        """
        target = self._inlining_key(node, module, "sub")
        if target is not None:
            key, name = target
            if key in seen:
                raise InlineCycleError(f"the inline chain comes back to {name!r}")
            seen = seen | {key}
        node, module = self._unfold_ref(node, module, "sub")
        if isinstance(node, viba_ast.Tagged):
            return {node.tag: (node.type, module)}
        if isinstance(node, _PROD_NODES):
            fields = {}
            for element in product_elements(node):
                for tag, field in self._tagged_fields(element, module, seen).items():
                    if tag in fields:
                        raise DuplicateTagError(
                            f"the tag {tag} is written twice in one product")
                    fields[tag] = field
            return fields
        return {}

    # ------------------------------------------------------------------
    # Applied generics: unfold one side, params bound via env_get
    # ------------------------------------------------------------------

    def _walk_mixed_typeapp(self, sn, s_mod, sp, p_mod):
        """Exactly one side is a TypeApp. Generic definitions unfold
        structurally; everything else (builtin generics, literal
        containers) falls through to lifting. Both sides applied
        unfolds too (see _walk_typeapps)."""
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
        builtin generics and literal containers have no body to lift."""
        resolved = self._resolve_name(node.constructor, module, side)
        if isinstance(resolved, VibaProgramErr):
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
        if isinstance(resolved, VibaProgramErr):
            raise UnresolvedTypeError(f"unresolvable constructor {node.constructor!r}")
        return resolved.ok_value

    def _binder(self, params, args, module, side):
        meanings = {}
        for name, arg in zip(params, args):
            meanings[name] = self._meaning(arg, module, side)
        def env_get(name):
            return meanings.get(name, VibaProgramErr(f"unbound parameter {name!r}"))
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
        if isinstance(sp, _PROD_NODES):
            # The sub may be written as one tagged field or as something that
            # is no product at all: splitting it reads it as a product of one.
            return self._walk_products(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.Tuple):
            return self._walk_tuple(sn, s_mod, sp, p_mod)
        if isinstance(sp, _EXP_NODES):
            return self._walk_exponent(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.TypeApp) and isinstance(sn, viba_ast.TypeApp):
            return self._walk_typeapps(sn, s_mod, sp, p_mod)
        if isinstance(sp, viba_ast.CodeBlock):
            # A code block has no members: the unit is its only resident, and a
            # code block counts as one. Its text is not compared — the protocol
            # hands no reader the material's code, so there is nothing to
            # compare against.
            if isinstance(sn, viba_ast.CodeBlock):
                return True
            return isinstance(self._lift(sn, s_mod, "sub"), NilType)
        return False

    def _walk_sum(self, sn, s_mod, sp, p_mod) -> bool:
        if isinstance(sn, _SUM_NODES):
            return self._all_branches(sn, s_mod, sp, p_mod)
        return self._any_branch(sn, s_mod, sp, p_mod)

    def _walk_tagged(self, sn, s_mod, sp, p_mod) -> bool:
        if isinstance(sn, _PROD_NODES):
            tagged, _ = self._split_product(sn, s_mod, "sub")
            if sp.tag not in tagged:
                return False
            return self._walk_bound(tagged[sp.tag], (sp.type, p_mod, None))
        if not isinstance(sn, viba_ast.Tagged):
            return self._walk(sn, s_mod, sp.type, p_mod)
        return sn.tag == sp.tag and self._walk(sn.type, s_mod, sp.type, p_mod)

    def _walk_products(self, sn, s_mod, sp, p_mod) -> bool:
        """Tagged products match by tag (commutative); sup's tags must
        all be present in sub (width). An untagged member that unfolds
        to a product is inlined, one that unfolds to the product unit
        disappears, and the rest pair in order. Bare TypeRefs to plain
        TypeDefinitions unfold so their tags participate (B = A * $find
        bool carries A's tags)."""
        sub_tagged, sub_bare = self._split_product(sn, s_mod, "sub")
        sup_tagged, sup_bare = self._split_product(sp, p_mod, "sup")
        if not self._tags_covered(sub_tagged, sup_tagged):
            return False
        if len(sub_bare) != len(sup_bare):
            return False
        return all(self._walk_bound(a, b) for a, b in zip(sub_bare, sup_bare))

    def _walk_bound(self, sub, sup) -> bool:
        """Compare two member bodies, each written under its own bindings.

        A member is ``(node, module, env)`` — see _split_element. The bindings
        are pushed only for the comparison, so a parameter of an inlined
        generic still resolves where the member body is walked.
        """
        pushed = []
        for side, member in (("sub", sub), ("sup", sup)):
            if member[2] is not None:
                self._env_stacks[side].append(member[2])
                pushed.append(side)
        try:
            return self._walk(sub[0], sub[1], sup[0], sup[1])
        finally:
            for side in pushed:
                self._env_stacks[side].pop()

    def _split_product(self, node, module, side: str, seen=frozenset(), env=None):
        """Flatten a Product/ProductChain into ({tag: member}, [member]),
        unfolding transparent TypeDefinitions along the way.

        An untagged member is an inline slot: one whose definition is a
        product contributes its own members here, recursively, and one
        that is the product unit (nil, or a name bound to it) is no
        member at all. Every other untagged member is one positional
        (bare) member. The same tag twice in one product is malformed
        input, and so is an inline chain that comes back to a definition
        it is already expanding (see _split_element)."""
        tagged, bare = {}, []
        for elem in product_elements(node):
            t, b = self._split_element(elem, module, side, seen, env)
            for tag, member in t.items():
                if tag in tagged:
                    raise DuplicateTagError(
                        f"the tag {tag} is written twice in one product")
                tagged[tag] = member
            bare.extend(b)
        return tagged, bare

    def _split_element(self, elem, module, side: str, seen=frozenset(), env=None):
        """One untagged member: what it contributes to ({tag: member}, [member]).

        A member is ``(node, module, env)``: what it is written as, the module
        its other names resolve in, and the bindings it was written under. The
        bindings travel with the member because an inlined generic hands its
        own body over — with `Box[T] = $x T`, what B = Box[int] carries as
        `$x`'s body is the written `T`, which only means `int` under Box's
        binder. A binder resolves its actuals where it is built, so the
        innermost one is all a member needs.

        Names and applications unfold one after another, with a generic's
        actuals bound, until the member is written out: a product inlines its own
        members here, a single tagged thing is one member, the product unit is
        no member. Anything else is one positional (bare) member, kept as it
        was written.

        An inline chain has to bottom out: a definition the chain meets twice
        never bottoms out, so the design is malformed input — `A = A * $x
        int` says A is written as itself. Refusing it is what keeps the walk
        finite as well.
        """
        pushed = []
        ambient = env
        try:
            node, node_mod = elem, module
            while True:
                node, node_mod = self._partial(node, node_mod, side)
                if isinstance(self._config_unit(node), NilType):
                    # A unit a config names is no member: a name is bypassed,
                    # not resolved, so it stays the unit it was declared to be.
                    return {}, []
                target = self._inlining_key(node, node_mod, side)
                if target is not None:
                    key, name = target
                    if key in seen:
                        raise InlineCycleError(
                            f"the inline chain comes back to {name!r}")
                    seen = seen | {key}
                node, node_mod = self._unfold_ref(node, node_mod, side)
                parts = (self._application_parts(node, node_mod, side)
                         if isinstance(node, viba_ast.TypeApp) else None)
                if parts is None:
                    break
                body, home, app_key, binder = parts
                if app_key in seen:
                    raise InlineCycleError(
                        f"the inline chain comes back to {node.constructor!r}")
                seen = seen | {app_key}
                self._env_stacks[side].append(binder)
                pushed.append(side)
                ambient = binder
                node, node_mod = body, home
            if isinstance(node, _PROD_NODES):
                return self._split_product(node, node_mod, side, seen, ambient)
            if isinstance(node, viba_ast.Tagged):
                return {node.tag: (node.type, node_mod, ambient)}, []
            if self._is_product_unit(node, node_mod, side):
                return {}, []
            return {}, [(elem, module, env)]
        finally:
            for _ in pushed:
                self._env_stacks[side].pop()

    def _application_parts(self, node, module, side: str):
        """(body, home, key, binder) for a generic application with a body to
        land on: the body as written, its home module, the definition's
        identity (the cycle key), and the env that binds the actuals to the
        parameters. None when there is no such body."""
        resolved = self._resolve_name(node.constructor, module, side)
        if isinstance(resolved, VibaProgramErr) or not isinstance(resolved.ok_value, AstNodeType):
            return None
        definition = resolved.ok_value.ast_node
        if not isinstance(definition, viba_ast.GenericDefinition):
            return None
        params = list(definition.generic_params or [])
        if len(params) != len(node.args):
            return None
        return (definition.body, resolved.ok_value.container_module,
                ("app", id(definition)), self._binder(params, node.args, module, side))

    def _inlining_key(self, node, module, side: str):
        """(key, written name) of the definition an untagged member names, or
        None when it names no definition of its own. The key is the definition's
        identity, so a chain that meets it twice is caught — the alias case
        included: with `B = A` and `A = B * $x int`, walking into B is walking
        into A."""
        if not isinstance(node, viba_ast.TypeRef):
            return None
        resolved = self._resolve_name(node.name, module, side)
        if isinstance(resolved, VibaProgramErr) or not isinstance(resolved.ok_value, AstNodeType):
            return None
        if not isinstance(resolved.ok_value.ast_node, viba_ast.TypeDefinition):
            return None
        return ("inline", id(resolved.ok_value.ast_node)), node.name

    def _normalize(self, node, module, side: str):
        """Unfold names, read through the lazy marker, give `<<`: until none
        of the three has anything left."""
        while True:
            before = (id(node), id(module))
            node, module = self._unfold_ref(node, module, side)
            node = self._through_marker(node, module)
            node, module = self._partial(node, module, side)
            if (id(node), id(module)) == before:
                return node, module

    def _through_marker(self, node, module):
        """`CalledByNeed[T]` is T: the marker says how that one argument is
        given, not what the type is (viba-interpreter.md)."""
        while True:
            marked = by_need_type(node, module)
            if marked is None:
                return node
            node = marked

    def _partial(self, node, module, side: str):
        """A design's `<<` reduced: the function with that argument given."""
        if not isinstance(node, viba_ast.Partial):
            return node, module
        return reduce_partial(
            node, module,
            lambda name, home: self._partial_target(name, home, side),
            lambda given, given_module, written, written_module: self._walk(
                given, given_module, written, written_module))

    def _partial_target(self, name, module, side: str):
        """(body, home) for the name a `<<` gives to, or None.

        A definition is itself; a bare import name is the module read as a
        function (`module_as_function`), while `module.Name` stays what it
        always was — that module's own definition.
        """
        resolved = self._resolve_name(name, module, side)
        if isinstance(resolved, VibaProgramErr) or not isinstance(resolved.ok_value, AstNodeType):
            return module_as_function(module, name)
        node = resolved.ok_value.ast_node
        if isinstance(node, viba_ast.GenericDefinition):
            return None                     # a bare generic name has no body
        if isinstance(node, viba_ast.TypeDefinition):
            node = node.body
        return node, resolved.ok_value.container_module

    def _config_unit(self, node):
        """The unit the config calls this piece, or None.

        A name and an application of it count the same: `U[{...}]` is the
        unit when `U` is one, and is then not resolved at all. A name reached
        through an import counts too: `m.U` names the unit, so what the config
        lists is the last segment.
        """
        if isinstance(node, viba_ast.TypeRef):
            name = node.name
        elif isinstance(node, viba_ast.TypeApp):
            name = node.constructor
        else:
            return None
        name = name.split(".")[-1]
        if name in self.nil_eqv:
            return NilType()
        if name in self.never_eqv:
            return NeverType()
        return None

    def _config_unit_node(self, node):
        """The written unit a config names, so the rest of the walk reads it
        like any other nil or never; the node itself when it names none."""
        unit = self._config_unit(node)
        if isinstance(unit, NilType):
            return _UNIT_NIL
        if isinstance(unit, NeverType):
            return _UNIT_NEVER
        return node

    def _drop_unit_args(self, args):
        """The arguments that carry documentation drop out of the chain, and
        only those: (A <- B <- U[{...}]) is (A <- B), and so is
        (A <- U[{...}] <- B) — what is dropped is the block, not the position
        it was written in.

        A unit written as a name is an argument like any other: the argument
        list is a tuple, and nil is a real slot in it (`Object` is that nil).
        Only documentation drops, so `never` is untouched too.
        """
        return [arg for arg in args if not self._carries_documentation(arg)]

    def _carries_documentation(self, node) -> bool:
        """A code block, or a unit a config names applied to one: where the
        writing layers park documentation. Documentation is not an
        argument. The block may sit behind a tag, as `U[$t {...}]` writes
        it."""
        if isinstance(node, viba_ast.CodeBlock):
            return True
        if not isinstance(node, viba_ast.TypeApp):
            return False
        if not isinstance(self._config_unit(node), NilType):
            return False
        return any(_has_code_block(arg) for arg in node.args)

    def _is_product_unit(self, node, module, side: str) -> bool:
        """The product's unit: nil by form, a name the config calls nil, or a
        name (a generic parameter among them) bound to nil. never is the sum's
        unit and counts as a member of a product."""
        if isinstance(node, viba_ast.Nil):
            return True
        unit = self._config_unit(node)
        if unit is not None:
            return isinstance(unit, NilType)
        if not isinstance(node, viba_ast.TypeRef):
            return False
        resolved = self._resolve_name(node.name, module, side)
        return isinstance(resolved, Ok) and isinstance(resolved.ok_value, NilType)

    def _tags_covered(self, sub_tagged, sup_tagged) -> bool:
        checks = (self._match_tag(sub_tagged, tag, member)
                  for tag, member in sup_tagged.items())
        return all(checks)

    def _match_tag(self, sub_tagged, tag, sup_member) -> bool:
        if tag not in sub_tagged:
            return False
        return self._walk_bound(sub_tagged[tag], sup_member)

    def _walk_tuple(self, sn, s_mod, sp, p_mod) -> bool:
        if not isinstance(sn, viba_ast.Tuple) or len(sn.elements) != len(sp.elements):
            return False
        pairs = zip(sn.elements, sp.elements)
        return all(self._walk(a, s_mod, b, p_mod) for a, b in pairs)

    def _walk_exponent(self, sn, s_mod, sp, p_mod) -> bool:
        """Two functions: the result covariantly, the arguments
        contravariantly, in the order they are written.

        Only functions compare with functions — never, the bottom, fits
        anywhere and is answered before this. Arguments that carry documentation (a
        code block, or a unit applied to one) drop out first, whatever
        position they were written in (_drop_unit_args), so the chain the sup
        asks about is the chain of its real arguments. The sub is brought to
        the sup's arity then, the sup never moving:

        * shorter, it is padded at its end with never — the argument
          nobody can supply — so (int <- $a int) <: (int <- $a int <-
          never) holds, the padded never being what the sup asks for,
          while a sup that asks for a bool is not answered by it;
        * longer, it is cut to the sup's length, so (int <- $a int <-
          $b str) <: (int <- $a int): the extra argument is one the sup
          never asks about, and is not looked at.

        Then every position compares from $arg0, and the shared prefix has
        to hold.

        Argument walks swap modules, so the env stacks swap with them: a
        free name resolves through the env of the syntax it came from."""
        if not (isinstance(sn, _EXP_NODES) and isinstance(sp, _EXP_NODES)):
            return False
        sub_res, sub_args = _exponent_parts(sn)
        sup_res, sup_args = _exponent_parts(sp)
        sub_args = self._drop_unit_args(sub_args)
        sup_args = self._drop_unit_args(sup_args)
        if len(sub_args) < len(sup_args):
            sub_args += [viba_ast.Never()] * (len(sup_args) - len(sub_args))
        if not self._walk(sub_res, s_mod, sup_res, p_mod):
            return False
        self._swap_envs()
        try:
            pairs = zip(sup_args, sub_args)  # the sub is cut to the sup's length
            walks = (self._walk(sa, p_mod, sb, s_mod) for sa, sb in pairs)
            return all(walks)
        finally:
            self._swap_envs()

    def _swap_envs(self):
        stacks = self._env_stacks
        stacks["sub"], stacks["sup"] = stacks["sup"], stacks["sub"]

    # Literal containers: ListLiteral[a, b, c] is a resident of
    # list[a | b | c]; SetLiteral likewise; DictLiteral[(k, v), ...]
    # of dict[k_union, v_union]. Containment, not name equality.
    _LITERAL_OF = {"list": "ListLiteral", "set": "SetLiteral", "dict": "DictLiteral"}

    def _walk_typeapps(self, sn, s_mod, sp, p_mod) -> bool:
        sub_c = self._resolve_constructor(sn.constructor, s_mod, "sub")
        sup_c = self._resolve_constructor(sp.constructor, p_mod, "sup")
        if self._literal_resident(sn, s_mod, sub_c, sp, p_mod, sup_c):
            return True
        if isinstance(sub_c, BuiltinGenericType) and isinstance(sup_c, BuiltinGenericType):
            return self._builtin_against_builtin(sn, s_mod, sub_c, sp, p_mod, sup_c)
        # Every other constructor names a definition, and a definition is an
        # alias of its body: unfold (one side or both) and compare structurally.
        if self._unfold_typeapp(sn, s_mod, "sub", sp, p_mod):
            return True
        if self._unfold_typeapp(sp, p_mod, "sup", sn, s_mod):
            return True
        # Both sides unfolded and the answer was False; that is the judgment.
        if self._unfolds(sn, s_mod, "sub") or self._unfolds(sp, p_mod, "sup"):
            return False
        # Neither side has a body to unfold: the actuals do not fit the
        # definition's parameters (Pair[int] for Pair[K, V]). The constructor's
        # name with its arguments is then all that is left — exactly what a
        # builtin container gives.
        if self._constructor_name(sub_c) != self._constructor_name(sup_c):
            return False
        if len(sn.args) != len(sp.args):
            return False
        return all(self._walk(a, s_mod, b, p_mod) for a, b in zip(sn.args, sp.args))

    def _builtin_against_builtin(self, sn, s_mod, sub_c, sp, p_mod, sup_c) -> bool:
        """A builtin container is its name and its arguments — there is no
        body to unfold (list / set / dict and the *Literal containers)."""
        if sub_c.name != sup_c.name:
            return False
        if len(sn.args) != len(sp.args):
            return False
        return all(self._walk(a, s_mod, b, p_mod) for a, b in zip(sn.args, sp.args))

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
        if isinstance(resolved, VibaProgramErr):
            raise UnresolvedTypeError(f"unresolvable constructor {name!r}")
        return resolved.ok_value

    def _unfolds(self, node, module, side) -> bool:
        """Whether this application has a body to unfold: its constructor is a
        generic definition and the actuals fit its parameters."""
        resolved = self._resolve_name(node.constructor, module, side)
        if not isinstance(resolved, Ok) or not isinstance(resolved.ok_value, AstNodeType):
            return False
        definition = resolved.ok_value.ast_node
        if not isinstance(definition, viba_ast.GenericDefinition):
            return False
        params = definition.generic_params or []
        return len(params) == len(node.args)

    def _constructor_name(self, constructor) -> str:
        """The name a constructor was written as, builtin or definition."""
        if isinstance(constructor, BuiltinGenericType):
            return constructor.name
        definition = _as_definition(getattr(constructor, "ast_node", None))
        return definition.name if definition is not None else ""


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
    A generic definition with no arguments stays wrapped: there are no
    actuals to bind its parameters, so it has no body to compare."""
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


def _flatten_sum(node):
    """Flatten a Sum or SumChain into a branch list."""
    if isinstance(node, viba_ast.Sum):
        return [node.left, node.right]
    if isinstance(node, viba_ast.SumChain):
        return list(node.elements)
    return [node]


def _param_index(node, params) -> "Optional[int]":
    """Which of the generic's parameters the operand is written as (through
    one tag at most); None when it is none of them."""
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
    """指数读成 (结果, 参数表)：结果在前，参数按书写顺序（$arg0 在前）。

    ``A <- B <- C`` 读成 (A, [B, C])：头一个是 ``$arg0`` 那一位，链长不等
    时按这一头截。支链（``A <- (B <- C)``）算一个参数。
    """
    elements = _exponent_elements(node)
    if not isinstance(node, _EXP_NODES):
        raise TypeError(f"not an exponent: {node!r}")
    return elements[0], list(elements[1:])


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
