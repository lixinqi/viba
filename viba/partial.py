"""`T << X`: the function T with the written argument X given to it.

Internal: this is the reduction `<<` goes through, used by `viba.is_sub_type`
and the descriptor layer. What a caller uses is the judgment (`is_sub_type`)
and the reading of a module as a function, not this module.

Partial computation, written in the design itself:

    (A <- $b B <- $c C) << $b B            is  A <- $c C
    (A <- $b B <- $c C) << $c C            is  A <- $b B
    (A <- $b B <- $c C) << $c C << $b B    is  A

Giving every argument leaves the result: the chain is gone, not empty, and
documentation (`Hint[$python_code {...}]`) is no argument, so a function that
ends in one still lands on its result.

The argument is matched the way the layers address one — by its tag when it is
tagged, by its written form otherwise — and what is given has to *fit* the slot
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
                       PartialError, UnresolvedTypeError)
from viba.viba_ast.tagged import (GETATTR_TAG, TAGGED_NAME, literal_symbol,
                                  symbol_of, symbol_problem, tag_of,
                                  tagged_node, tagged_problem)

_TAG_NODES = (viba_ast.Tagged, viba_ast.Member)

_EXP_NODES = (viba_ast.Exponent, viba_ast.ExponentChain)


class _NamedMember:
    """`$__getattr__ << X` while the name is still to come.

    `$__getattr__` reads a member by a name given as a value, so the member is
    known when the chain gives that name: the next argument is the name and `X`
    is the value the member is read out of (viba-interpreter.md). It is gone as
    soon as the name is in.
    """

    __slots__ = ("owner", "module")

    def __init__(self, owner, module):
        self.owner = owner
        self.module = module


RET_NAME = "__ret__"
DEF_NAME = "__def__"
GET_ARGS_NAME = "__get_args__"
ENVIRON_TAG = "$env"
ENV_TYPE = "Env"
ENVIRONMENT_TYPE = "Environment"

def module_as_function(module, name):
    """(body, written_in) for a bare import name read as a function, or None.

    A module is a function too, and `__def__` is that function: the result first,
    then the call's parameters — `__def__ = int <- $env Env <- $a int` is a module
    answering an int and asking for an environment and an `$a`. The environment is
    the call's rule rather than an argument (giving it is what runs the call), so
    it is not one of the slots; everything else is, in written order. A module that
    declares no `__def__` takes no arguments at all, and one that declares no
    `__ret__` is design, not a program. `name` has to be one of this module's
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
    slots = [parameter_node(tag, written) for tag, written, is_env, _slot
             in _def_parameters(imported.ok_value) if not is_env]
    # 环境不进类型：它是调用的规矩（给环境就是执行），不是设计的一个参数。所以
    # 模块当函数读出来的类型就是 `__def__` 去掉环境那一格：结果在前，实参在后，
    # 和一份函数定义的类型一样；不收实参的模块就是这个结果本身。
    declared = _definition(imported.ok_value, DEF_NAME)
    result = result_of(declared.body) if declared is not None else ret.body
    chain = declared.body if declared is not None else ret.body
    problem = environment_result_problem(chain, f"module {name!r}")
    if problem is not None:
        raise PartialError(problem)
    if not slots:
        return (result, imported.ok_value)
    return (viba_ast.ExponentChain([result] + slots), imported.ok_value)


def _def_parameters(module):
    """The parameters a module's `__def__` declares, or the empty list.

    `__def__` is a chain, so it reads as `parameters_of` reads one; a module with
    no `__def__` declares no parameters at all.
    """
    definition = _definition(module, DEF_NAME)
    if definition is None:
        return []
    body = definition.body
    if not isinstance(body, _EXP_NODES):
        raise PartialError(f"{DEF_NAME} is not a function type: {_written(body)}")
    return parameters_of(body)


def parameters_of(chain):
    """Every parameter a written chain declares, in written order.

    A chain reads as a function type: the first element is the result, the last is
    the body (documentation, or the call the chain ends on), and everything between
    them is a parameter. Each comes back as (tag, written, is the environment,
    argument slot) — the slot counts arguments only, so the environment has none:
    giving it is what runs the call (viba-interpreter.md).
    """
    elements = _elements(chain)[1:]
    if elements and isinstance(elements[-1], (viba_ast.CodeBlock, viba_ast.Partial)):
        elements = elements[:-1]             # a body written on the chain
    params = []
    slot = 0
    for element in elements:
        if isinstance(element, viba_ast.CodeBlock):
            continue                        # documentation is no parameter
        if isinstance(element, viba_ast.Tagged):
            tag, written = element.tag, element.type
        else:
            tag, written = None, element     # a parameter with no tag: position only
        if tag == ENVIRON_TAG:
            params.append((tag, written, True, None))
        else:
            params.append((tag, written, False, slot))
            slot += 1
    return params


def result_of(chain):
    """The result type a written chain declares: its first element."""
    elements = _elements(chain)
    return elements[0] if elements else None


def answers_the_environment(chain) -> bool:
    """Whether a written chain declares the environment itself as its answer.

    A function answers a value; the environment is the call's rule rather than a
    value (viba-interpreter.md), so no function written in a module answers one —
    only a builtin function does (`Environment.$sub_env`, `$tmp_env`). The answer
    is a chain's first element, and `$env` written on it says the same as the name
    does.
    """
    result = result_of(chain)
    return result is not None and names_the_environment(result)


def names_the_environment(node) -> bool:
    """Whether a written piece names the environment itself: `Env` or `Environment`.

    A tag says it too: `$env T` is the environment written the way the parameter
    is (viba-interpreter.md).
    """
    if isinstance(node, viba_ast.Tagged):
        if node.tag == ENVIRON_TAG:
            return True
        node = node.type
    return (isinstance(node, viba_ast.TypeRef)
            and node.name.split(".")[-1] in (ENV_TYPE, ENVIRONMENT_TYPE))


def environment_result_problem(chain, written) -> Optional[str]:
    """Why a chain may not answer the environment, or None when it does not.

    One wording for every layer that reads a written chain as a function: the
    interpreter while it runs, the judgment and the descriptor while they read a
    design.
    """
    if not answers_the_environment(chain):
        return None
    return (f"{written} answers the environment: {ENV_TYPE} is the call's rule rather "
            f"than a value, and only a builtin function may answer one")


def file_environment_result_problem(tree) -> Optional[str]:
    """Why a parsed file may not answer the environment, or None when it does not.

    Every function the file writes is read here: a definition's own chain
    (`__def__` among them) and the chains its products carry as members, however
    deep they sit. A file is no builtin library, so a file that writes a function
    answering the environment is refused where it is read — one place for the
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


def parameter_node(tag, written):
    """A parameter as written: its tag back on it when it carries one."""
    return viba_ast.Tagged(tag, written) if tag else written


def get_args_product(chain):
    """The product `__get_args__` answers for a chain: its parameters, as a type.

    `__get_args__ << __def__` is the arguments a call received. Read as a type
    that is the product of the parameters the chain declares — the environment
    among its members, which is how a module names it (`args.env`) — and run it is
    the product the call was handed (viba.interpret reads it that way).
    """
    members = [viba_ast.Tagged(tag, written) if tag else written
               for tag, written, _is_env, _slot in parameters_of(chain)]
    if not members:
        return viba_ast.Nil()
    if len(members) == 1:
        return members[0]
    return viba_ast.ProductChain(members)


def product_elements(node):
    """The factors of a written product, flattened in written order.

    One definition for the three layers that read a product apart: the
    interpreter (a product is its factors), the judgment and this module.
    """
    if isinstance(node, viba_ast.Product):
        return product_elements(node.left) + product_elements(node.right)
    if isinstance(node, viba_ast.ProductChain):
        return list(node.elements)
    return [node]


def _definition(module, name):
    """The definition this name stands for: the last one written under it."""
    found = None
    for node in getattr(module, "module", module).body:
        if getattr(node, "name", None) == name:
            found = node
    return found


def reduce_partial(node, module, resolve: Callable, judge: Callable,
                   outermost: bool = True) -> Tuple[object, object]:
    """(node, module) with every `<<` given.

    `resolve(name, module) -> (body, written_in) | None` is how a written name is
    unfolded; an alias is followed to the end of the chain.
    `judge(sub, sub_module, sup, sup_module) -> bool` says whether a given
    argument fits the slot it is written to. It is also what tells the environment
    apart from an argument (`_is_the_environment`), since the environment's own
    type is what says so.
    """
    while isinstance(node, viba_ast.Partial):
        base, base_module = reduce_partial(node.function, module, resolve, judge,
                                           outermost=False)
        node, module = _give(base, base_module, node.argument, module, resolve, judge)
    if outermost:
        node, module = _nothing_left_to_give(node, module, judge)
    return node, module


def _nothing_left_to_give(node, module, judge=None):
    """A call with only the environment or the empty product left is that call.

    `()` is how "nothing" is written, and the environment is the call's rule
    rather than an argument, so a chain that owes only one of them has nothing
    left to give: giving it and leaving it out are the same call. It is read here,
    once the whole chain has been given, so that writing it out still works.
    """
    if not isinstance(node, _EXP_NODES):
        return node, module
    elements = _elements(node)
    if len(elements) > 1 and all(_is_documentation(element)
                                 or _is_the_empty_product(element)
                                 or _is_the_environment(element, module, judge)
                                 for element in elements[1:]):
        # 环境那一格不算实参（`_give` 给环境也不占格），所以链上只剩它也等于没有
        # 剩下的：这一次调用落在结果上。
        return elements[0], module
    return node, module


def _give(base, module, argument, argument_module, resolve, judge):
    base = _tagged_base(base, module, resolve)
    if isinstance(base, _NamedMember):
        return _named_member(base, argument, argument_module, resolve)
    if isinstance(base, viba_ast.Member) and base.tag == GETATTR_TAG:
        # `$__getattr__ << X << <name>`: X is the value the member is read out
        # of, and the name that tags it is the next argument.
        return _NamedMember(argument, argument_module), module
    if isinstance(base, viba_ast.Member):
        # `$tag` 的那个成员是从**第一个实参**身上取的，所以环境在这里不是"执行"，是值。
        return _member_of(base.tag, argument, argument_module, resolve, judge)
    if _is_the_environment(argument, argument_module, judge):
        # 给环境 = 执行这一步：它从此不在链上占位置。环境不是设计的参数，所以给了
        # 之后剩下的格子就是全部要给的（没给的时候，环境那一格仍是第一格，别的实参
        # 落到它上面照样是错的）。
        base, module = _unfold(base, module, resolve)
        elements = _elements(base) if isinstance(base, _EXP_NODES) else [base]
        rest = [elements[0]] + [one for one in elements[1:]
                                if not _is_the_environment(one, module, judge)]
        if len(rest) == 1:
            return rest[0], module
        return viba_ast.ExponentChain(rest), module
    base, module = _unfold(base, module, resolve)
    if not isinstance(base, _EXP_NODES):
        raise PartialError(
            f"only a function has arguments to give, not {_written(base)}")
    elements = _elements(base)
    for index, written in enumerate(elements[1:], start=1):
        if _matches(written, argument, module, argument_module, judge):
            rest = elements[:index] + elements[index + 1:]
            if all(_is_documentation(element) for element in rest[1:]):
                # Nothing but the result and documentation is left: documentation
                # is no argument (the judgment drops it too), so this is the
                # result itself - what a finished call declares at that position.
                return rest[0], module
            return viba_ast.ExponentChain(rest), module
    raise PartialError(f"the function has no such argument: {_written(argument)}")


def _member_of(tag, owner, owner_module, resolve, judge):
    """(type, written_in) of the `$tag` member of the value a chain gave first, given
    that value.

    `$tag << X << a` is `X.tag << X << a`: X is the value the member is taken
    from, and it is also what the member is given first — a member of an
    environment is a plain function of the environment. So the member's own type
    is reduced with X as its first argument, exactly as `X.tag << X` would be.
    The value is read the way any design piece is read: a name runs to its body
    (`Env` is `Environment`), and a product is its factors, one of which the
    tag addresses. Reading no such member is a design mistake, like giving a
    function an argument it does not have.
    """
    if isinstance(owner, viba_ast.Tagged):
        owner = owner.type      # the tag addresses the argument; the value is inside
    written, written_module = owner, owner_module
    owner, owner_module = _unfold(owner, owner_module, resolve)
    if isinstance(owner, (viba_ast.Product, viba_ast.ProductChain)):
        for factor in product_elements(owner):
            if isinstance(factor, viba_ast.Tagged) and factor.tag == tag:
                base, written_in = _unfold(factor.type, owner_module, resolve)
                return _give(base, written_in, written, written_module, resolve, judge)
        raise PartialError(
            f"no member tagged {tag!r} to take from {_written(owner)}")
    raise PartialError(
        f"{_written(owner)} is no value to take the member {tag!r} from")


def _tagged_base(node, module, resolve):
    """A written `__tagged__[...]` at a chain head as the tag it stands for.

    `__tagged__["hello"] << X` is `$hello << X`, and the symbol may be a name a
    decision bound instead of a written string. A node that is no tagged
    application, or whose symbol this layer cannot read, is handed back as it is
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
    """`$__getattr__ << X << <name>`: the member the name tags.

    `X.<name>` is the same read, and this is where a name that is a *value*
    becomes a tag: the member is the one the member name addresses. A name this
    layer cannot read — not a written string, and not a name that stands for one
    — leaves the member unnamed, so the answer is `Any`: which member it is is
    known when the program runs, and no design can say it before that
    (viba-interpreter.md).
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
        f"no member tagged {tag!r} to take from {_written(base.owner)}")


def _symbol_text_of(node, module, resolve):
    """The string a written name spells here, or None when it spells none.

    A written string is itself; a name is whatever it stands for — a decision
    hands a symbol over as the string a `pattern` line extracted.
    """
    if isinstance(node, viba_ast.Constant) and isinstance(node.value, str):
        return node.value
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


def _is_the_environment(node, module=None, judge=None) -> bool:
    """Whether this written piece is the environment itself.

    The environment is the call's rule rather than an argument — giving it is what
    runs the call — so it does not count among a chain's slots. A piece is the
    environment when it says so by tag (`$env ...`), or when what it denotes is the
    environment's own type: a module reads the environment as `args.env`, whose
    declared type is `Env`, which is `Environment`. What reads a type is the
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


def _judged(judge, given, given_module, written, written_module) -> bool:
    """`judge`, with a piece it cannot read answered as "no"."""
    try:
        return bool(judge(given, given_module, written, written_module))
    except (PartialError, UnresolvedTypeError, DuplicateTagError, InlineCycleError):
        return False


def _is_the_empty_product(node) -> bool:
    """The empty product as the language writes it: the empty tuple `()`."""
    return isinstance(node, viba_ast.Tuple) and not node.elements


def _unfold(node, module, resolve):
    """A name runs to its body, name after name."""
    seen = set()
    while True:
        if not isinstance(node, viba_ast.TypeRef) or node.name in seen:
            return node, module
        seen.add(node.name)
        target = resolve(node.name, module)
        if target is None:
            return node, module
        node, module = target


def _written(node) -> str:
    """The piece as one line: error messages read better without the layout."""
    return " ".join(viba_ast.unparse_type(node).split())


def _is_documentation(node) -> bool:
    """A piece that carries a code block: documentation, not an argument."""
    return any(isinstance(part, viba_ast.CodeBlock) for part in viba_ast.walk(node))


def _matches(written, given, module, given_module, judge) -> bool:
    """Does the given argument go to this written slot?

    A tag when both carry one (that is how a field is addressed) — and then the
    given type has to fit the declared one, so `(A <- $b B) << $b C` is legal
    only when `C <: B`. An argument written without a tag takes the next free
    slot, the way a call gives one, if it fits there: that is how a module's
    `__def__` parameters are given one by one. Without a tag on either side there is no
    address to name: the two are the same piece, or they are not.
    """
    if isinstance(written, viba_ast.Tagged) and isinstance(given, viba_ast.Tagged):
        if written.tag != given.tag:
            return False
        if judge(given.type, given_module, written.type, module):
            return True
        raise PartialError(
            f"{_written(given)} does not fit {_written(written)}: "
            f"{_written(given.type)} <: {_written(written.type)} does not hold")
    if isinstance(written, viba_ast.Tagged):
        if judge(given, given_module, written.type, module):
            return True
        raise PartialError(
            f"{_written(given)} does not fit {_written(written)}: "
            f"{_written(given)} <: {_written(written.type)} does not hold")
    return viba_ast.unparse_type(written) == viba_ast.unparse_type(given)


def _elements(node):
    """The exponent's elements in written order: result first."""
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _elements(node.result) + [node.argument]
    return [node]


__all__ = ["reduce_partial", "module_as_function", "product_elements",
           "parameters_of", "parameter_node", "result_of", "get_args_product",
           "answers_the_environment", "names_the_environment",
           "environment_result_problem", "file_environment_result_problem"]
