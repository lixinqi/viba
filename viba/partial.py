"""`T << X`: the function T with the argument X, as the source has it, given to it.

Internal: this is the reduction `<<` goes through, used by `viba.is_sub_type`
and the descriptor layer. What a caller uses is the judgment (`is_sub_type`)
and a module counting as a function, not this module.

Partial computation, as the design itself has it:

    (A <- $b B <- $c C) << $b B            is  A <- $c C
    (A <- $b B <- $c C) << $c C            is  A <- $b B
    (A <- $b B <- $c C) << $c C << $b B    is  A

Giving every argument leaves the result: the chain is gone, not empty, and
documentation (`Hint[$python_code {...}]`) is no argument, so a function that
ends in one still lands on its result.

The argument is matched the way the layers address one — by its tag when it is
tagged, by its source form otherwise — and what is given has to *fit* the slot
it is given to: `(A <- $b B) << $b C` needs `C <: B`. The judge is the
caller's (the judgment passes its own walk, the descriptor layer passes
`is_sub_type`), and names are unfolded through the caller's resolution too, so
a design is refused where it is built and where it is judged. Giving an
argument the function does not have, or one that does not fit, is a mistake
(`PartialError`), not a judgment.
"""

from typing import Callable, Optional, Tuple

from viba import viba_ast
from viba.type import (BUILTIN_MODULE, DuplicateTagError, InlineCycleError, Ok,
                       PartialError, UnresolvedTypeError, takes_a_product)
from viba.viba_ast.tagged import (GET_ATTR_TAG, GET_ITEM_TAG,
                                  INTERPRETER_MEMBER_TAGS, TAGGED_NAME,
                                  interpreter_member_chain, literal_symbol,
                                  symbol_of, symbol_problem, tag_of,
                                  tagged_node, tagged_problem)

_TAG_NODES = (viba_ast.Tagged, viba_ast.Member)

_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)


class _NamedMember:
    """`$get_attr << X` while the name is still to come.

    `$get_attr` takes a member by a name given as a value, so the member is
    known when the chain gives that name: the next argument is the name and `X`
    is the value the member is taken out of (viba-interpreter.md). It is gone as
    soon as the name is in.
    """

    __slots__ = ("owner", "module")

    def __init__(self, owner, module):
        self.owner = owner
        self.module = module


RET_NAME = "__impl__"
DEF_NAME = "__decl__"
GET_ARGS_NAME = "__get_args__"
ENVIRON_TAG = "$env"
ENV_TYPE = "Env"
ENVIRONMENT_TYPE = "Environment"

# The two names whose call has its name as *data*: the interpreter answers them
# itself, and what such a call answers is `Any` — the design has no name to look
# up, so it cannot say more (viba-interpreter.md, `__dyn_call__` / `__dyn_method__`).
DYN_CALL_NAME = "__dyn_call__"
DYN_METHOD_NAME = "__dyn_method__"


def module_as_function(module, name):
    """(body, source_module) for a bare import name taken as a function, or None.

    A module is a function too, and `__decl__` is that function — the declaration,
    the result first, then the call's parameters: `__decl__ = int <- $env Env <-
    $a int` is a module answering an int and asking for an environment and an
    `$a`. The environment is the call's rule rather than an argument (giving it is
    what runs the call), so it is not one of the slots; everything else is, in
    the order the source has them. A module that gives no `__decl__` is no
    function at all (it is a module of definitions), and one that declares no
    `__impl__` is design, not a program. `name` has to be one of this module's
    imports — the name the import binds, not a member of it (a dotted name is that
    module's own definition and resolves as it always did).
    """
    imports = getattr(module, "imports", None)
    if not imports or name not in imports:
        return None
    imported = module.module_environment(imports[name])
    if not isinstance(imported, Ok):
        return None
    ret = _definition(imported.ok_value, RET_NAME)
    if ret is None:
        return None
    slots = [parameter_node(tag, source_form) for tag, source_form, is_env, _slot
             in _def_parameters(imported.ok_value) if not is_env]
    # The environment does not go into the type: it is a rule of the call (giving it means
    # execute), not a parameter of the design. The type a module counts as when it is taken as a
    # function is `__decl__` without the environment slot: result first, arguments after, the same
    # as a function definition's type; a module taking no arguments is that result itself.
    declared = _definition(imported.ok_value, DEF_NAME)
    if declared is None:
        return None                  # no function signature, no function: a module is only a module
    result = result_of(declared.body)
    chain = declared.body
    problem = environment_result_problem(chain, f"module {name!r}")
    if problem is not None:
        raise PartialError(problem)
    if not slots:
        return (result, imported.ok_value)
    return (viba_ast.ExponentChain([result] + slots), imported.ok_value)


def _def_parameters(module):
    """The parameters a module's `__decl__` declares, or the empty list.

    `__decl__` is a chain, so `parameters_of` counts it as one; a module with no
    `__decl__` declares no parameters at all.
    """
    definition = _definition(module, DEF_NAME)
    if definition is None:
        return []
    body = definition.body
    if not isinstance(body, _EXP_NODES):
        raise PartialError(f"{DEF_NAME} is not a function type: {_source_form(body)}")
    return parameters_of(body)


def parameters_of(chain):
    """Every parameter a chain declares, in the order the source has them.

    A chain counts as a function type: the first element is the result, the last is
    the body (documentation, or the call the chain ends on), and everything between
    them is a parameter. Each comes back as (tag, the source form, is the
    environment, argument slot) — the slot counts arguments only, so the
    environment has none: giving it is what runs the call (viba-interpreter.md).
    """
    elements = _elements(chain)[1:]
    if elements and isinstance(elements[-1], (viba_ast.CodeBlock, viba_ast.Partial)):
        elements = elements[:-1]             # a body the source puts on the chain
    params = []
    slot = 0
    for element in elements:
        if isinstance(element, viba_ast.CodeBlock):
            continue                        # documentation is no parameter
        if isinstance(element, viba_ast.Tagged):
            tag, source_form = element.tag, element.type
        else:
            tag, source_form = None, element  # a parameter with no tag: position only
        if tag == ENVIRON_TAG:
            params.append((tag, source_form, True, None))
        else:
            params.append((tag, source_form, False, slot))
            slot += 1
    return params


def result_of(chain):
    """The result type a chain declares: its first element."""
    elements = _elements(chain)
    return elements[0] if elements else None


def answers_the_environment(chain) -> bool:
    """Whether a chain declares the environment itself as its answer.

    A function answers a value; the environment is the call's rule rather than a
    value (viba-interpreter.md), so no function the source puts in a module
    answers one — only a builtin function does (`Environment.$sub_env`,
    `$tmp_env`). The answer is a chain's first element, and `$env` on it says the
    same as the name does.
    """
    result = result_of(chain)
    return result is not None and names_the_environment(result)


def names_the_environment(node) -> bool:
    """Whether a piece names the environment itself: `Env` or `Environment`.

    A tag says it too: `$env T` is the environment in the source form the
    parameter has (viba-interpreter.md).
    """
    if isinstance(node, viba_ast.Tagged):
        if node.tag == ENVIRON_TAG:
            return True
        node = node.type
    return (isinstance(node, viba_ast.TypeRef)
            and node.name.split(".")[-1] in (ENV_TYPE, ENVIRONMENT_TYPE))


def environment_result_problem(chain, name) -> Optional[str]:
    """Why a chain may not answer the environment, or None when it does not.

    One wording for every layer that takes a chain as a function: the interpreter
    while it runs, the judgment and the descriptor while they take a design.
    """
    if not answers_the_environment(chain):
        return None
    return (f"{name} answers the environment: {ENV_TYPE} is the call's rule rather "
            f"than a value, and only a builtin function may answer one")


def file_environment_result_problem(tree) -> Optional[str]:
    """Why a parsed file may not answer the environment, or None when it does not.

    Every function the file gives is taken here: a definition's own chain
    (`__decl__` among them) and the chains its products carry as members, however
    deep they sit. A file is no builtin library, so a file that gives a function
    answering the environment is refused where it is taken — one place for the
    interpreter and the descriptor to ask (`viba-interpreter.md`).
    """
    for stmt in getattr(tree, "body", ()):
        if not isinstance(stmt, (viba_ast.TypeDefinition, viba_ast.GenericDefinition)):
            continue
        for node in viba_ast.walk(stmt.body):
            if not isinstance(node, _EXP_NODES):
                continue
            if answers_the_environment(node):
                return environment_result_problem(node, stmt.name)
    return None


def parameter_node(tag, source_form):
    """A parameter in its source form: its tag back on it when it carries one."""
    return viba_ast.Tagged(tag, source_form) if tag else source_form


def get_args_product(chain):
    """The product `__get_args__` answers for a chain: its parameters, as a type.

    `__get_args__ << __decl__` is the arguments a call received. Taken as a type
    that is the product of the parameters the chain declares — the environment
    among its members, which is how a module names it (`args.env`) — and run it is
    the product the call was handed (viba.interpret takes it that way).
    """
    members = [viba_ast.Tagged(tag, source_form) if tag else source_form
               for tag, source_form, _is_env, _slot in parameters_of(chain)]
    if not members:
        return viba_ast.Nil()
    if len(members) == 1:
        return members[0]
    return viba_ast.ProductChain(members)


def product_elements(node):
    """The factors of a product, flattened in the order the source has them.

    One definition for the three layers that take a product apart: the
    interpreter (a product is its factors), the judgment and this module.
    """
    if isinstance(node, viba_ast.Product):
        return product_elements(node.left) + product_elements(node.right)
    if isinstance(node, viba_ast.ProductChain):
        return list(node.elements)
    return [node]


def _definition(module, name):
    """The definition this name stands for: the last one the source puts under it."""
    found = None
    for node in getattr(module, "module", module).body:
        if getattr(node, "name", None) == name:
            found = node
    return found


def _dynamic_call(node) -> bool:
    """Whether this chain is a call whose name is data (`__dyn_call__`, `__dyn_method__`)."""
    while isinstance(node, viba_ast.Partial):
        node = node.function
    return (isinstance(node, viba_ast.TypeRef)
            and node.name in (DYN_CALL_NAME, DYN_METHOD_NAME))


def reduce_partial(node, module, resolve: Callable, judge: Callable,
                   outermost: bool = True) -> Tuple[object, object]:
    """(node, module) with every `<<` given.

    `resolve(name, module) -> (body, source_module) | None` is how a name is
    unfolded; an alias is followed to the end of the chain.
    `judge(sub, sub_module, sup, sup_module) -> bool` says whether a given
    argument fits the slot the source puts it at. It is also what tells the
    environment apart from an argument (`is_the_environment`), since the
    environment's own type is what says so.
    """
    if _dynamic_call(node):
        # The name travels as data, so the design has nothing to unfold: what this
        # call answers is `Any`, and its arguments are taken as they come.
        return viba_ast.Any(), module
    while isinstance(node, viba_ast.Partial):
        base, base_module = reduce_partial(node.function, module, resolve, judge,
                                           outermost=False)
        node, module = _give(base, base_module, node.argument, module, resolve, judge)
    if outermost:
        node, module = _nothing_left_to_give(node, module, judge)
    return node, module


def _nothing_left_to_give(node, module, judge=None):
    """A call with only the environment or the empty product left is that call.

    `()` is how the source spells "nothing", and the environment is the call's
    rule rather than an argument, so a chain that owes only one of them has
    nothing left to give: giving it and leaving it out are the same call. It is
    settled here, once the whole chain has been given, so that serializing it
    still works.
    """
    if not isinstance(node, _EXP_NODES):
        return node, module
    elements = _elements(node)
    if len(elements) > 1 and all(_is_documentation(element)
                                 or _is_the_empty_product(element)
                                 or is_the_environment(element, module, judge)
                                 for element in elements[1:]):
        # The environment slot is no argument (`_give` does not fill it), so leaving only it on the
        # chain is like leaving nothing: this call lands on the result.
        return elements[0], module
    return node, module


def _give(base, module, argument, argument_module, resolve, judge):
    base = _tagged_base(base, module, resolve)
    if isinstance(base, _NamedMember):
        return _named_member(base, argument, argument_module, resolve)
    if isinstance(base, viba_ast.Member) and base.tag == GET_ATTR_TAG:
        if is_the_environment(argument, argument_module, judge):
            # `$get_attr << $env … << X << <name>`: the environment is the call's
            # rule rather than an argument, so it is dropped here and the value and
            # the name follow — the environment the member is called in is the
            # member's own slot, which its own type gives (`_named_member`).
            return base, module
        # `$get_attr << X << <name>`: X is the value the member is taken out
        # of, and the name that tags it is the next argument.
        return _NamedMember(argument, argument_module), module
    if isinstance(base, viba_ast.Member) and base.tag in INTERPRETER_MEMBER_TAGS:
        chain = _member_design_chain(base.tag)
        if is_the_environment(argument, argument_module, judge):
            # The environment is no argument of a member the interpreter answers
            # either: giving it is what runs the member, and what the member takes
            # is still to come.
            return chain, BUILTIN_MODULE
        return _give(chain, BUILTIN_MODULE, argument, argument_module, resolve, judge)
    if isinstance(base, viba_ast.Member):
        # The member for `$tag` is taken from the **first argument**, so the environment here is a
        # value, not "execute".
        return _member_of(base.tag, argument, argument_module, resolve, judge)
    if is_the_environment(argument, argument_module, judge):
        # Giving the environment = executing this step: it stops taking a place on the chain. The
        # environment is no parameter of the design, so once it is given, the slots left are
        # everything there is to give (when it is not given, the environment slot is still the
        # first, and other arguments landing on it are still wrong).
        base, module = _unfold(base, module, resolve)
        elements = _elements(base) if isinstance(base, _EXP_NODES) else [base]
        rest = [elements[0]] + [one for one in elements[1:]
                                if not is_the_environment(one, module, judge)]
        if len(rest) == 1:
            return rest[0], module
        return viba_ast.ExponentChain(rest), module
    base, module = _unfold(base, module, resolve)
    if not isinstance(base, _EXP_NODES):
        raise PartialError(
            f"only a function has arguments to give, not {_source_form(base)}")
    elements = _elements(base)
    for index, declared in enumerate(elements[1:], start=1):
        if _takes_the_rest(declared):
            # A slot spelled `...` takes whatever the call has left: the signature
            # says the rest goes to what this call answers, which the design cannot
            # follow any further, so the answer is the result type itself.
            return elements[0], module
        if _matches(declared, argument, module, argument_module, judge):
            rest = elements[:index] + elements[index + 1:]
            if all(_is_documentation(element) for element in rest[1:]):
                # Nothing but the result and documentation is left: documentation
                # is no argument (the judgment drops it too), so this is the
                # result itself - what a finished call declares at that position.
                return rest[0], module
            return viba_ast.ExponentChain(rest), module
    raise PartialError(f"the function has no such argument: {_source_form(argument)}")


def _member_design_chain(tag: str):
    """The design's own call for a member the interpreter answers.

    It is the member's signature (`viba/viba_ast/tagged.py`) with the environment
    left out: the environment is the call's rule rather than an argument, and the
    design drops it from every call this way (`module_as_function`). What is left
    is what that member answers — `int` for `$len`, `bool` for `$in`, `list[str]`
    for `$keys`, `Any` for what a container holds — and the parameters it takes,
    each under its own name (`$container`, `$address`, `$piece`, `$table`), so a
    source that gives them by name still fits, and an argument spelled with the
    tag of the piece itself goes to the first of them that is free (`_matches`,
    `takes_a_product`).

    `$get_item` keeps one slot for the rest (`$args ...`): what a chain spells
    after the address is given to the element it took, and the design cannot
    follow that any further (`_takes_the_rest`).
    """
    signature = _elements(interpreter_member_chain(tag))
    kept = [element for element in signature[1:]
            if not (isinstance(element, viba_ast.Tagged)
                    and element.tag == ENVIRON_TAG)]
    if tag == GET_ITEM_TAG:
        kept.append(viba_ast.Tagged("$args", viba_ast.Ellipsis()))
    return viba_ast.ExponentChain([signature[0]] + kept)


def _takes_the_rest(declared) -> bool:
    """Whether this slot is spelled `...`: the rest of the arguments go there.

    A parameter of a chain may be `$args ...` (`__decl__ = Any <- $f Any <- $args
    ...`, `viba/builtin.viba`): it holds the product of everything left. A design
    that reaches one has nothing left to say about the piece it lands on — the
    result is what the call answers.
    """
    return isinstance(declared, viba_ast.Tagged) and isinstance(declared.type,
                                                                viba_ast.Ellipsis)


def _member_of(tag, owner, owner_module, resolve, judge):
    """(type, source_module) of the `$tag` member of the value a chain gave first,
    given that value.

    `$tag << X << a` is `X.tag << X << a`: X is the value the member is taken
    from, and it is also what the member is given first — a member of an
    environment is a plain function of the environment. So the member's own type
    is reduced with X as its first argument, exactly as `X.tag << X` would be.
    A member that is no function has nothing to give that value to, so a member
    that is a value is answered as it stands: `$y << box` is the piece `$y` holds
    (`$uncompress_relative_path << args.env` is the string or the nil it holds).
    The value is taken the way any design piece is taken: a name runs to its body
    (`Env` is `Environment`), and a product is its factors, one of which the
    tag addresses. There being no such member is a design mistake, like giving a
    function an argument it does not have.
    """
    if isinstance(owner, viba_ast.Tagged):
        owner = owner.type      # the tag addresses the argument; the value is inside
    given_owner, given_owner_module = owner, owner_module
    owner, owner_module = _unfold(owner, owner_module, resolve)
    if isinstance(owner, (viba_ast.Product, viba_ast.ProductChain)):
        for factor in product_elements(owner):
            if isinstance(factor, viba_ast.Tagged) and factor.tag == tag:
                base, source_module = _unfold(factor.type, owner_module, resolve)
                if not isinstance(base, _EXP_NODES):
                    # The member is a value, not a function of the value it was
                    # taken from: there is no argument to give it.
                    return base, source_module
                return _give(base, source_module, given_owner, given_owner_module,
                             resolve, judge)
        raise PartialError(
            f"no member tagged {tag!r} to take from {_source_form(owner)}")
    raise PartialError(
        f"{_source_form(owner)} is no value to take the member {tag!r} from")


def _tagged_base(node, module, resolve):
    """A `tagged[...]` in the source at a chain head as the tag it stands for.

    `tagged["hello"] << X` is `$hello << X`, and the symbol may be a name a
    decision bound instead of a string in the source. A node that is no tagged
    application, or whose symbol this layer cannot take, is handed back as it is
    (viba/viba_ast/tagged.py).
    """
    if not (isinstance(node, viba_ast.TypeApp) and node.constructor == TAGGED_NAME):
        return node
    problem = tagged_problem(node.args)
    if problem is not None:
        raise PartialError(problem)
    symbol = literal_symbol(node.args[0])
    if symbol is None:
        text = _symbol_text_of(node.args[0], module, resolve)
        if text is None:
            return node
        symbol = symbol_of(text)
        if symbol is None:
            raise PartialError(symbol_problem(text))
    return tagged_node(symbol, node.args)


def _named_member(base, name, name_module, resolve):
    """`$get_attr << X << <name>`: the member the name tags.

    `X.<name>` takes the member the same way, and this is where a name that is a
    *value* becomes a tag: the member is the one the member name addresses. A
    name this layer cannot take — not a string in the source, and not a name that
    stands for one — leaves the member unnamed, so the answer is `Any`: which
    member it is is known when the program runs, and no design can say it before
    that (viba-interpreter.md).
    """
    text = _symbol_text_of(name, name_module, resolve)
    if text is None:
        return viba_ast.Any(), base.module
    symbol = symbol_of(text)
    if symbol is None:
        raise PartialError(symbol_problem(text))
    tag = tag_of(symbol)
    owner, owner_module = _unfold(base.owner, base.module, resolve)
    if isinstance(owner, (viba_ast.Product, viba_ast.ProductChain)):
        for factor in product_elements(owner):
            if isinstance(factor, viba_ast.Tagged) and factor.tag == tag:
                return factor.type, owner_module
    raise PartialError(
        f"no member tagged {tag!r} to take from {_source_form(base.owner)}")


def _symbol_text_of(node, module, resolve):
    """The string a name in the source spells here, or None when it spells none.

    A string in the source is itself; a name is whatever it stands for — a
    decision hands a symbol over as the string a `pattern` line extracted.
    """
    if isinstance(node, viba_ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, viba_ast.MemberTaken):
        # On a name chain, a member taken out is that name; when the left side is an application it
        # is no name, and is left to the other cases.
        path = viba_ast.source_path(node)
        if path is None:
            return None
        node = viba_ast.TypeRef(path)
    if not isinstance(node, viba_ast.TypeRef):
        return None
    target = resolve(node.name, module)
    if target is None:
        return None
    body = target[0]
    while isinstance(body, viba_ast.TypeDefinition):
        body = body.body
    if isinstance(body, viba_ast.Constant) and isinstance(body.value, str):
        return body.value
    return None


def is_the_environment(node, module=None, judge=None) -> bool:
    """Whether this piece is the environment itself.

    The environment is the call's rule rather than an argument — giving it is what
    runs the call — so it does not count among a chain's slots. A piece is the
    environment when it says so by tag (`$env ...`), or when what it denotes is the
    environment's own type: a module takes the environment as `args.env`, whose
    declared type is `Env`, which is `Environment`. What takes a type is the
    judgment, so a piece whose type cannot be settled is not the environment and is
    left to the ordinary walk to report.
    """
    if isinstance(node, viba_ast.Tagged):
        if node.tag == ENVIRON_TAG:
            return True
        node = node.type
    if judge is None:
        return False
    environment = viba_ast.TypeRef(ENV_TYPE)
    return (_judged(judge, node, module, environment, BUILTIN_MODULE)
            and _judged(judge, environment, BUILTIN_MODULE, node, module))


def _judged(judge, given, given_module, declared, declared_module) -> bool:
    """`judge`, with a piece it cannot take answered as "no"."""
    try:
        return bool(judge(given, given_module, declared, declared_module))
    except (PartialError, UnresolvedTypeError, DuplicateTagError, InlineCycleError):
        return False


def _is_the_empty_product(node) -> bool:
    """The empty product as the language spells it: the empty tuple `()`."""
    return isinstance(node, viba_ast.Tuple) and not node.elements


def _unfold(node, module, resolve):
    """A name runs to its body, name after name."""
    seen = set()
    while True:
        if isinstance(node, viba_ast.MemberTaken):
            path = viba_ast.source_path(node)
            if path is None:
                return node, module
            node = viba_ast.TypeRef(path)
        if not isinstance(node, viba_ast.TypeRef) or node.name in seen:
            return node, module
        seen.add(node.name)
        target = resolve(node.name, module)
        if target is None:
            return node, module
        node, module = target


def _source_form(node) -> str:
    """The piece as one line: error messages are clearer without the layout."""
    return " ".join(viba_ast.unparse_type(node).split())


def _is_documentation(node) -> bool:
    """A piece that carries a code block: documentation, not an argument."""
    return any(isinstance(part, viba_ast.CodeBlock) for part in viba_ast.walk(node))


def _matches(declared, given, module, given_module, judge) -> bool:
    """Does the given argument go to this declared slot?

    A tag when both carry one (that is how a field is addressed) — and then the
    given type has to fit the declared one, so `(A <- $b B) << $b C` is legal
    only when `C <: B`. A slot that holds a product is the exception: a piece
    tagged with something that names no slot is a product of one member, and that
    is where it goes (`$len << $x xs` in a step of `sequential`), keeping its own
    tag at the call — the same slot the runtime picks (`_Pending.slot_for`,
    `viba/type.py`, `takes_a_product`). An argument with no tag in the source
    takes the next free slot, the way a call gives one, if it fits there: that is
    how a module's `__decl__` parameters are given one by one. Without a tag on
    either side there is no address to name: the two are the same piece, or they
    are not.
    """
    if isinstance(declared, viba_ast.Tagged) and isinstance(given, viba_ast.Tagged):
        if declared.tag != given.tag:
            return (takes_a_product(declared.type)
                    and judge(given.type, given_module, declared.type, module))
        if judge(given.type, given_module, declared.type, module):
            return True
        raise PartialError(
            f"{_source_form(given)} does not fit {_source_form(declared)}: "
            f"{_source_form(given.type)} <: {_source_form(declared.type)} does not hold")
    if isinstance(declared, viba_ast.Tagged):
        if judge(given, given_module, declared.type, module):
            return True
        raise PartialError(
            f"{_source_form(given)} does not fit {_source_form(declared)}: "
            f"{_source_form(given)} <: {_source_form(declared.type)} does not hold")
    return viba_ast.unparse_type(declared) == viba_ast.unparse_type(given)


def _elements(node):
    """The exponent's elements in the order the source has them: result first."""
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _elements(node.result) + [node.argument]
    return [node]


__all__ = ["reduce_partial", "module_as_function", "product_elements",
           "parameters_of", "parameter_node", "result_of", "get_args_product",
           "answers_the_environment", "names_the_environment",
           "environment_result_problem", "file_environment_result_problem"]
