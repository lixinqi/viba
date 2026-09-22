"""viba.interpreter — run a viba module.

A module is a file, and a file is also a function: its input is `environ`,
its output is `__ret__`. Type inference reads the same file as a type
(viba.is_sub_type); computation runs it (here). A file that wants to be
runnable defines `__ret__`; a file that does not is design only.

    from viba.interpreter import interpret

    interpret("add_demo.viba", environ)          # -> Result[VibaNode]

The interpreter is coupled to no function at all: viba ships with no library
functions, and every implementation comes from the environment's compute
side (`EnvironmentCompute.get_func(module_path, func_name)`), which the
caller writes or generates. Whatever that function is, it takes the
environment — every executable function depends on it — and a module's
sub-environment holds the parent's compute, so a chain of modules shares one
implementation source.

The host side, spelled out:

    EnvironmentStorage(cur_storage_path, sub_storage=None)
        where a module's files and sub-modules live; `sub(name)` hands out a
        child, made on demand, whose path is `<cur>/<name>`.

    EnvironmentCompute(get_func)
        the implementations: `get_func(module_path, func_name)` returns a
        callable, or None when that module has no such function. The path is
        the storage path of the environment the module was given, so the host
        can tell one module's `add` from another's.

    Environment(storage, compute)
        the two together, and `sub_env(name)` for a child — which keeps the
        parent's compute.

A host function is called with the arguments already evaluated, in the order
they are written: a piece of material arrives as a `viba.reflect.VibaNode`,
anything else as itself (the environment among them). It answers with a
`VibaNode`, or with a plain Python value, which lands as a leaf.
"""

from pathlib import Path
from typing import Optional

from viba import viba_ast
from viba.reflect import VibaNode, access as reflect_access
from viba.type import AstNodeType, Err, ModuleType, Ok, Result, custom_module
from viba.viba_type_descriptor import descriptor_of

# A scalar a host answers belongs to no file: its leaf gets an empty module.
# Parsed once, not once per answer.
_NO_MODULE = custom_module("")

RET_NAME = "__ret__"
ENVIRON_NAME = "environ"
ENVIRON_TAG = "$env"
ENVIRON_TYPE = "Environment"


# ----------------------------------------------------------------------
# The host side of an environment
# ----------------------------------------------------------------------


class EnvironmentStorage:
    """Where a module's files and sub-modules live."""

    __slots__ = ("cur_storage_path", "sub_storage")

    def __init__(self, cur_storage_path: str, sub_storage: Optional[dict] = None):
        self.cur_storage_path = cur_storage_path
        self.sub_storage = dict(sub_storage or {})

    def sub(self, name: str) -> "EnvironmentStorage":
        """The child storage for `name`: `<cur>/<name>`, made on demand."""
        if name not in self.sub_storage:
            path = f"{self.cur_storage_path}/{name}" if self.cur_storage_path else name
            self.sub_storage[name] = EnvironmentStorage(path)
        return self.sub_storage[name]


class EnvironmentCompute:
    """The implementations: `get_func(module_path, func_name) -> callable|None`."""

    __slots__ = ("get_func",)

    def __init__(self, get_func):
        self.get_func = get_func


class Environment:
    """Storage and compute, and the sub-environments under it."""

    __slots__ = ("storage", "compute")

    def __init__(self, storage: EnvironmentStorage, compute: EnvironmentCompute):
        self.storage = storage
        self.compute = compute

    def sub_env(self, name) -> "Environment":
        """A child environment: its own storage, the parent's compute.

        `name` is what the viba side wrote: a material node lands as the leaf
        it carries, so `environ.sub_env << "add_demo"` names the module.
        """
        if isinstance(name, VibaNode):
            name = name.value
        return Environment(self.storage.sub(str(name)), self.compute)


# ----------------------------------------------------------------------
# Values
# ----------------------------------------------------------------------


class _Material:
    """A piece of material with its design: a `VibaNode`."""

    __slots__ = ("node",)

    def __init__(self, node: VibaNode):
        self.node = node


class _Host:
    """Anything the host handed over: the environment, a host value."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj


class _Given:
    """An argument on its way to a call: its address and its value."""

    __slots__ = ("tag", "value")

    def __init__(self, tag, value):
        self.tag = tag
        self.value = value


def _argument_value(value):
    """What a host function is handed.

    Material arrives as its node, a viba function as a Python callable (the
    host calls it like any other function: the values it is called with are
    nodes or host objects, and the answer is a node or a host object), and
    anything else as itself — the environment among them.
    """
    if isinstance(value, _Material):
        return value.node
    if isinstance(value, (_VibaFunc, _HostFunction, _ModuleFunc)):
        return value.as_callable()
    if isinstance(value, _Host):
        return value.obj
    return value


def _given_value(value):
    """A value a host handed back into a call: a node, a scalar, or an object."""
    if isinstance(value, _Material):
        return value
    answer = _answer("a host argument", value)
    if isinstance(answer, Err):
        raise RuntimeError(answer.err_msg)
    return answer.ok_value


# ----------------------------------------------------------------------
# interpret
# ----------------------------------------------------------------------


def interpret(viba_main_file: str, environ: Environment,
              viba_path: Optional[str] = None) -> Result:
    """Run `viba_main_file` with `environ`; Result[VibaNode] is its `__ret__`.

    `viba_path` is where modules are looked up, like PYTHONPATH: the
    directories are searched in order for `<name>.viba` (a dotted name as a
    path), and the directory of the file that wrotes the import is searched
    first.
    """
    if not isinstance(environ, Environment):
        return Err("interpret needs an Environment")
    return _Runner(viba_path).run_file(viba_main_file, environ)


class _Runner:
    """One run: the files it has loaded, and where it looks for more."""

    def __init__(self, viba_path: Optional[str] = None):
        self.paths = [Path(p) for p in (viba_path or "").split(":") if p]
        self.by_path: dict = {}        # resolved path -> module
        self.by_name: dict = {}        # module name -> module
        self.path_of: dict = {}        # module name -> file it was loaded from
        self.running: list = []        # module activations, for cycle refusal

    def run_file(self, file: str, environ: Environment) -> Result:
        path = Path(file)
        if not path.exists():
            return Err(f"no such file: {file}")
        module = self._load(path, path.stem)
        if isinstance(module, Err):
            return module
        return run_module(self, module.ok_value, environ, path.stem, str(path))

    def _load(self, path: Path, name: str):
        key = str(path.resolve())
        if key not in self.by_path:
            try:
                source = path.read_text()
            except OSError as exc:
                return Err(f"cannot read {path}: {exc}")
            try:
                self.by_path[key] = custom_module(source)
            except SyntaxError as exc:
                return Err(f"cannot parse {path}: {exc}")
        self.by_name[name] = self.by_path[key]
        self.path_of[name] = str(path)
        return Ok(self.by_path[key])

    def imported(self, name: str, near: Optional[str]):
        """The module `name`: loaded, or found next to `near`, or on the path."""
        if name in self.by_name:
            return Ok(self.by_name[name])
        rel = Path(*name.split(".")).with_suffix(".viba")
        places = []
        if near:
            places += [Path(near).parent / rel, Path(near).parent / f"{name}.viba"]
        for base in self.paths:
            places += [base / rel, base / f"{name}.viba"]
        for place in places:
            if place.exists():
                return self._load(place, name)
        return Err(f"module {name!r} not found (next to {near} and on VIBA_PATH)")


def run_module(runner: _Runner, module: ModuleType, environ: Environment,
               name: str, file: Optional[str]) -> Result:
    """The module as a function: `environ` in, `__ret__` out."""
    if file is None:
        file = runner.path_of.get(name)     # a module called through an import
    ret = _definition(module, RET_NAME)
    if ret is None:
        return Err(f"module {name!r} has no {RET_NAME}: it is design, not a program")
    if name in runner.running:
        return Err(f"module {name!r} is already running: a module call cycle")
    runner.running.append(name)
    try:
        value = _Activation(runner, module, environ, name, file).evaluate(ret.body)
    finally:
        runner.running.pop()
    if isinstance(value, Err):
        return value
    if not isinstance(value.ok_value, (_Material, _Host)):
        return Err(f"{name}.{RET_NAME} is a function still waiting for arguments")
    return Ok(_argument_value(value.ok_value))


def _definition(module: ModuleType, name: str):
    for node in module.module.body:
        if getattr(node, "name", None) == name:
            return node
    return None


class _Activation:
    """One module running with one environment."""

    def __init__(self, runner: _Runner, module: ModuleType, environ: Environment,
                 name: str, file: Optional[str]):
        self.runner = runner
        self.module = module
        self.environ = environ
        self.name = name
        self.file = file
        self.defined: dict = {}

    # ---- expressions ----

    def evaluate(self, node):
        """Result: the value this piece writes."""
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never, viba_ast.Any)):
            return Ok(_Material(VibaNode(reflect_access, self._descriptor(node), node)))
        if isinstance(node, viba_ast.Partial):
            return self._apply_chain(node)
        if isinstance(node, viba_ast.TypeRef):
            return self._resolve(node.name)
        if isinstance(node, viba_ast.CodeBlock):
            return Err("a code block is documentation: it is not a value")
        return Err(f"cannot compute {type(node).__name__}")

    def _descriptor(self, node):
        return descriptor_of(AstNodeType(node, self.module))

    def _imports(self) -> dict:
        """The file's import table: what each import binds, and the module it
        names (`import a.b as c` binds c, `import a.b` binds a.b)."""
        return {stmt.alias or stmt.module: stmt.module
                for stmt in self.module.module.body
                if isinstance(stmt, viba_ast.Import)}

    def _resolve(self, name: str):
        if name == ENVIRON_NAME:
            return Ok(_Host(self.environ))
        if name.startswith(ENVIRON_NAME + "."):
            return self._environ_member(name[len(ENVIRON_NAME) + 1:])
        definition = _definition(self.module, name)
        if definition is not None:
            return self._defined(name, definition)
        bound = self._imported_name(name)
        if bound is not None:
            module_name, rest = bound
            imported = self.runner.imported(module_name, self.file)
            if isinstance(imported, Err):
                return imported
            if rest:
                return self._member_of(imported.ok_value, module_name, rest)
            return Ok(_ModuleFunc(self.runner, imported.ok_value, module_name))
        return Err(f"no definition named {name!r} in module {self.name!r}")

    def _imported_name(self, name: str):
        """(module name, rest) when `name` is written through an import."""
        imports = self._imports()
        if name in imports:
            return imports[name], ""
        for prefix in sorted(imports, key=len, reverse=True):
            if name.startswith(prefix + "."):
                return imports[prefix], name[len(prefix) + 1:]
        return None

    def _defined(self, name: str, definition):
        if name in self.defined:
            return self.defined[name]
        body = definition.body
        if isinstance(body, (viba_ast.Exponent, viba_ast.ExponentChain)):
            value = Ok(_VibaFunc(self, name, body))     # a function is a value
        else:
            value = self.evaluate(body)
        self.defined[name] = value
        return value

    def _member_of(self, module, module_name: str, rest: str):
        """`demo.print`: a function of an imported module."""
        definition = _definition(module, rest)
        if definition is None:
            return Err(f"module {module_name!r} has no {rest!r}")
        other = _Activation(self.runner, module, self.environ, module_name,
                            self.runner.path_of.get(module_name))
        return other._defined(rest, definition)

    def _environ_member(self, rest: str):
        member = getattr(self.environ, rest, None)
        if not callable(member):
            return Err(f"the environment has no {rest!r}")
        return Ok(_HostFunction(rest, member, slots=1))

    # ---- calls ----

    def _apply_chain(self, node):
        given = []
        while isinstance(node, viba_ast.Partial):
            value = self._argument(node.argument)
            if isinstance(value, Err):
                return value
            if value.ok_value is not None:
                given.append(value.ok_value)
            node = node.function
        function = self.evaluate(node)
        if isinstance(function, Err):
            return function
        current = function.ok_value
        for item in reversed(given):
            current = _give(current, item)
            if isinstance(current, Err):
                return current
            current = current.ok_value
        return Ok(current)

    def _argument(self, node):
        """Result: the `_Given` this argument is, or Ok(None) for documentation."""
        if isinstance(node, viba_ast.CodeBlock):
            return Ok(None)
        tag = None
        inner = node
        if isinstance(node, viba_ast.Tagged):
            tag, inner = node.tag, node.type
        value = self.evaluate(inner)
        if isinstance(value, Err):
            return value
        return Ok(_Given(tag, value.ok_value))


def _give(function, item):
    """Give one written argument to `function`, or say what stood there.

    What is not a function is named by its kind: a node by its address (it is
    a value), anything else by its type. The message stays the same between
    runs — an object's repr would carry an address that changes every time.
    """
    if isinstance(function, (_VibaFunc, _HostFunction, _ModuleFunc)):
        return function.give(item)
    if isinstance(function, _Material):
        return Err(f"{function.node!r} is not a function: it is a value")
    if isinstance(function, _Host):
        return Err(f"{type(function.obj).__name__} is not a function")
    return Err(f"{type(function).__name__} is not a function")


class _Callable:
    """The viba values a host may call: one `as_callable` between them."""

    def as_callable(self):
        """This function as a host callable: the host hands over the values it
        wants given, in the function's own order, and gets the answer back."""
        def call(*values):
            current = self
            for value in values:
                given = _give(current, _Given(None, _given_value(value)))
                if isinstance(given, Err):
                    raise RuntimeError(given.err_msg)
                current = given.ok_value
            return _argument_value(current)
        return call


class _VibaFunc(_Callable):
    """A viba function: its chain is the design, `given` what it has so far."""

    def __init__(self, activation: _Activation, name: str, chain, given=None):
        self.activation = activation
        self.name = name
        self.chain = chain
        self.given = dict(given or {})

    @property
    def slots(self):
        """The arguments, in written order: a tag, or None for a position."""
        out = []
        for element in _elements(self.chain)[1:]:
            if isinstance(element, viba_ast.Tagged):
                out.append(element.tag)
            elif isinstance(element, viba_ast.CodeBlock):
                continue                    # documentation is no argument
            else:
                out.append(None)
        return out

    def give(self, item):
        """Give one written argument to the slot it addresses."""
        slots = self.slots
        given = dict(self.given)
        if item.tag is not None:
            if item.tag not in slots:
                return Err(f"{self.name} takes no {item.tag} argument")
            given[slots.index(item.tag)] = item.value
        else:
            # An argument written without a tag is the next slot that is free:
            # the caller did not say which one, and the order is the design's.
            free = [index for index in range(len(slots)) if index not in given]
            if not free:
                return Err(f"{self.name} takes no more arguments")
            given[free[0]] = item.value
        if all(index in given for index in range(len(slots))):
            return self.call(given)
        return Ok(_VibaFunc(self.activation, self.name, self.chain, given))

    def call(self, given):
        """Every slot is filled: the environment's compute side implements it."""
        problem = self._environ_problem(given)
        if problem is not None:
            return Err(problem)
        environ = given[self.slots.index(ENVIRON_TAG)].obj
        storage = getattr(environ, "storage", None)
        compute = getattr(environ, "compute", None)
        if compute is None:
            return Err(f"{self.name}: the environment carries no compute side")
        module_path = getattr(storage, "cur_storage_path", "") if storage else ""
        try:
            host = compute.get_func(module_path, self.name)
        except Exception as exc:            # the host is the host's business
            return Err(f"get_func({module_path!r}, {self.name!r}) raised {exc!r}")
        if host is None:
            return Err(f"no implementation for {self.name!r} in module {module_path!r}")
        args = [_argument_value(given[index]) for index, _ in self._ordered(given)]
        try:
            answer = host(*args)
        except Exception as exc:
            return Err(f"{self.name} raised {exc!r}")
        return _answer(self.name, answer)

    def _ordered(self, given):
        """The arguments in written order: what the host function is handed."""
        return [(index, given[index]) for index in range(len(self.slots))
                if index in given]

    def _environ_problem(self, given):
        """Every executable function depends on the environment."""
        if ENVIRON_TAG not in self.slots:
            return (f"{self.name} takes no {ENVIRON_TAG} {ENVIRON_TYPE} argument: "
                    f"every executable function depends on the environment")
        if self.slots.index(ENVIRON_TAG) not in given:
            return f"{self.name} was not given the environment"
        environ = given[self.slots.index(ENVIRON_TAG)]
        if not isinstance(environ, _Host) or not isinstance(environ.obj, Environment):
            return f"{self.name} was not given an {ENVIRON_TYPE}"
        return None


class _HostFunction(_Callable):
    """A function the host hangs off the environment: `environ.sub_env`."""

    def __init__(self, name: str, func, slots: int = 1, given=None):
        self.name = name
        self.func = func
        self.slots = slots
        self.given = list(given or [])

    def give(self, item):
        values = self.given + [_argument_value(item.value)]
        if len(values) < self.slots:
            return Ok(_HostFunction(self.name, self.func, self.slots, values))
        try:
            answer = self.func(*values)
        except Exception as exc:
            return Err(f"environ.{self.name} raised {exc!r}")
        return _answer(f"environ.{self.name}", answer)


class _ModuleFunc(_Callable):
    """A module as a function: its one argument is the environment."""

    def __init__(self, runner: _Runner, module: ModuleType, name: str):
        self.runner = runner
        self.module = module
        self.name = name
        self.given = []

    def give(self, item):
        values = self.given + [item.value]
        if len(values) < 1:
            return Ok(_ModuleFunc(self.runner, self.module, self.name))
        environ = values[0].obj if isinstance(values[0], _Host) else None
        if not isinstance(environ, Environment):
            return Err(f"module {self.name!r} needs an Environment")
        answer = run_module(self.runner, self.module, environ, self.name, None)
        if isinstance(answer, Err):
            return answer
        # `interpret` hands the node out; inside a run a module's answer is a
        # value like any other, so it goes back into the value model.
        node = answer.ok_value
        return Ok(_Material(node) if isinstance(node, VibaNode) else _Host(node))


def _answer(name, answer):
    """Result: what a host function answered, as a value.

    A `VibaNode` is taken as it is, an `Environment` stays a host value, and
    `None` is `nil` the way it is in the builder. A plain Python value lands
    as a leaf — but only a scalar one: a list, a dict, a callable or any other
    object has no leaf to be, and guessing one would put a shape into the
    material that no design asked for.
    """
    if isinstance(answer, VibaNode):
        return Ok(_Material(answer))
    if isinstance(answer, Environment):
        return Ok(_Host(answer))
    if answer is not None and not isinstance(answer, (bool, int, float, str)):
        return Err(f"{name} answered {type(answer).__name__}, "
                   f"which is no leaf: answer a VibaNode, a scalar, or None")
    node = viba_ast.Nil() if answer is None else viba_ast.Constant(answer)
    return Ok(_Material(VibaNode(reflect_access,
                                 descriptor_of(AstNodeType(node, _NO_MODULE)), node)))


def _elements(node):
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _elements(node.result) + [node.argument]
    return [node]


__all__ = ["interpret", "Environment", "EnvironmentStorage", "EnvironmentCompute"]
