"""viba.interpret — run a viba module.

A module is a file, and a file is also a function: its input is `environ`,
its output is `__ret__`. Type inference reads the same file as a type
(viba.is_sub_type); computation runs it (here). A file that wants to be
runnable defines `__ret__`; a file that does not is design only.

    from viba.interpret import interpret

    interpret("add_demo.viba", environ)  # -> Ok(VibaNode) | VibaProgramErr(str) | a stop
    interpret("main.viba", environ, get_file=files.get)   # sources from anywhere

A function the compute side does not implement is not a failure: the run stops
and answers `NotMyDutyException` — `$not_my_duty_exception Duty` — the deferral
that says this host is not the one to finish it, and carries the step, the
material it was given and why (`roadmap.md`). A step whose implementation broke
answers `UnderlyingVibaOpFailed` — `$underlying_viba_op_failed Failure` — with
the same step in it. What is left is `Ok(node)`, for the `__ret__` that came
out, and `VibaProgramErr(message)` — `$viba_program_err str` — for a program or
an environment that cannot run at all.

The interpreter is coupled to no function at all: viba ships with no library
functions, and every implementation comes from the environment's compute
side (`EnvironmentCompute.get_func(module_path, func_name)`), which the
caller writes or generates. Whatever that function is, it takes the
environment — every executable function depends on it — and a module's
sub-environment holds the parent's compute, so a chain of modules shares one
implementation source.

Neither is it coupled to a filesystem: `get_file(file_path) -> str | None` is
where a file's source comes from. Left out, files are read from disk; given,
nothing else is read — the whole run can be served out of memory, and a path
that has no file is answered `None` (or a `FileNotFoundError`), so the search
moves on to the next place.

The host side, spelled out:

    EnvironmentStorage(cur_storage_path, sub_storage=None, store_root_dir=None)
        where a module's files, sub-modules and snapshots live; `sub(name)`
        hands out a child, made on demand, whose path is `<cur>/<name>`.

    EnvironmentCompute(get_func)
        the implementations: `get_func(module_path, func_name)` returns a
        callable, or None when that module has no such function. The path is
        the storage path of the environment the module was given, so the host
        can tell one module's `add` from another's.

    Environment(storage, compute, viba_path=None)
        the three together, and `sub_env(name)` / `tmp_sub_env()` for a child —
        which keeps the parent's compute and its module search path.

A host function is called with the arguments already evaluated, in the order
they are written: a piece of material arrives as a `viba.reflect.VibaNode`,
anything else as itself (the environment among them). It answers with a
`VibaNode`, or with a plain Python value, which lands as a leaf.

A result has to be replayable, and a host function that is not pure — it
reads a clock, a random number, a service — is where that is decided: it
takes the snapshot of its answer (`write_snapshot`, or `replayed` which does
both sides), and the next run of the same call finds it and plays it again
(`read_snapshot`). The snapshots live in the environment's storage
(`EnvironmentStorage.store_root_dir`), under the path the call runs at, and
they are serialized viba data, so what was stored can be read and checked.
"""

import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from viba import serialize, viba_ast
from viba.reflect import VibaNode, access as reflect_access
from viba.type import (REASON_GET_FUNC_RAISED, REASON_NO_IMPLEMENTATION, REASON_NO_LEAF,
                       REASON_RAISED, REASON_REFUSED, AstNodeType, VibaProgramErr, UnderlyingVibaOpFailed,
                       InterpretResult, ModuleType, NotMyDutyException, Ok, Step,
                       custom_module)
from viba.viba_type_descriptor import descriptor_of

# A scalar a host answers belongs to no file: its leaf gets an empty module.
# Parsed once, not once per answer.
_NO_MODULE = custom_module("")

TMP_PREFIX = "tmp_"
RET_NAME = "__ret__"
ENVIRON_NAME = "environ"
ENVIRON_TAG = "$env"
ENVIRON_TYPE = "Environment"

# Where a snapshot goes when the storage did not name a store root, and what
# the definition in a snapshot file is called.
DEFAULT_STORE_ROOT = os.path.join(tempfile.gettempdir(), "viba-store")
SNAPSHOT_NAME = "value"
SNAPSHOT_SUFFIX = ".viba"


# ----------------------------------------------------------------------
# The host side of an environment
# ----------------------------------------------------------------------


class EnvironmentStorage:
    """Where a module's files, sub-modules and snapshots live."""

    __slots__ = ("cur_storage_path", "sub_storage", "store_root_dir")

    def __init__(self, cur_storage_path: str, sub_storage: Optional[dict] = None,
                 store_root_dir: Optional[str] = None):
        self.cur_storage_path = cur_storage_path
        self.sub_storage = dict(sub_storage or {})
        self.store_root_dir = store_root_dir or DEFAULT_STORE_ROOT

    def sub(self, name: str) -> "EnvironmentStorage":
        """The child storage for `name`: `<cur>/<name>`, made on demand."""
        if name not in self.sub_storage:
            path = f"{self.cur_storage_path}/{name}" if self.cur_storage_path else name
            self.sub_storage[name] = EnvironmentStorage(path, None, self.store_root_dir)
        return self.sub_storage[name]

    def tmp(self) -> "EnvironmentStorage":
        """A child under a name nobody chose, fresh on every call — the way a
        temporary file is. Two module calls then cannot land on one path."""
        while True:
            name = TMP_PREFIX + uuid.uuid4().hex[:12]
            if name not in self.sub_storage:
                return self.sub(name)

    # ---- the store: text under the store root ----

    def read_text(self, file_path: str) -> Optional[str]:
        """The text stored at `file_path`, or None when nothing is there.

        `file_path` is read under `store_root_dir`, so a snapshot written by an
        earlier run (or another process) is found again.
        """
        try:
            return self._store_path(file_path).read_text()
        except FileNotFoundError:
            return None

    def write_text(self, file_path: str, content: str) -> None:
        """Store `content` at `file_path`, making the directories on the way."""
        path = self._store_path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _store_path(self, file_path: str, root_dir: Optional[str] = None) -> Path:
        """`file_path` read under the store root (or another root of the same
        kind): the parts are taken as written, so a path never escapes it."""
        relative = Path(*[part for part in str(file_path).split("/") if part])
        return Path(root_dir or self.store_root_dir) / relative


class EnvironmentCompute:
    """The implementations: `get_func(module_path, func_name) -> callable|None`."""

    __slots__ = ("get_func",)

    def __init__(self, get_func):
        self.get_func = get_func


class Environment:
    """Storage, compute and the module search path, and the children under it."""

    __slots__ = ("storage", "compute", "viba_path")

    def __init__(self, storage: EnvironmentStorage, compute: EnvironmentCompute,
                 viba_path=None):
        self.storage = storage
        self.compute = compute
        # Where modules are looked up, like PYTHONPATH: a string of directories,
        # or one path. A module runs under the environment it was handed, so its
        # own imports are looked up where that environment says — which is how a
        # sub-environment keeps the parent's search path along with its compute.
        self.viba_path = viba_path

    def sub_env(self, name) -> "Environment":
        """A child environment: its own storage, the parent's compute.

        `name` is what the viba side wrote: a material node lands as the leaf
        it carries, so `environ.sub_env << "add_demo"` names the module.
        The same name is the same child, handed back again.
        """
        if isinstance(name, VibaNode):
            name = name.value
        return Environment(self.storage.sub(str(name)), self.compute, self.viba_path)

    def tmp_sub_env(self, ignored=None) -> "Environment":
        """A child environment under a name of its own, fresh every time:
        `environ.tmp_sub_env << ()` — no name to pick, and no two calls share a
        storage path. The written argument is ignored; it is there because a
        call gives one, and `()` is the way to write "nothing"."""
        return Environment(self.storage.tmp(), self.compute, self.viba_path)


# ----------------------------------------------------------------------
# Snapshots: what makes a run replayable
# ----------------------------------------------------------------------


def snapshot_path(environ: Environment, name: str = SNAPSHOT_NAME) -> str:
    """Where this call's snapshot lives: `<cur_storage_path>/<name>.viba`,
    read under the storage's `store_root_dir`."""
    path = _storage(environ).cur_storage_path
    return "/".join(part for part in (path, name + SNAPSHOT_SUFFIX) if part)


def read_snapshot(environ: Environment, name: str = SNAPSHOT_NAME):
    """The value stored for this call, or None when nothing is stored yet.

    A snapshot is serialized viba data: it is parsed and rooted again, so it
    comes back as the material it was. What cannot be read raises — a host
    function's exception is the `VibaProgramErr` the caller sees.
    """
    text = _storage(environ).read_text(snapshot_path(environ, name))
    if text is None:
        return None
    module = custom_module(text)
    stored = _definition(module, SNAPSHOT_NAME)
    if stored is None:
        raise RuntimeError(f"the snapshot has no {SNAPSHOT_NAME}: {text!r}")
    node = stored.body
    return VibaNode(reflect_access, descriptor_of(AstNodeType(node, module)), node)


def write_snapshot(environ: Environment, value, name: str = SNAPSHOT_NAME) -> None:
    """Store `value` as this call's snapshot: serialized viba data, so it can
    be read back and played again."""
    written = serialize.serialize(SNAPSHOT_NAME, material(value))
    if isinstance(written, VibaProgramErr):
        raise RuntimeError(f"cannot snapshot this value: {written.err_msg}")
    _storage(environ).write_text(snapshot_path(environ, name), written.ok_value)


def replayed(environ: Environment, compute, name: str = SNAPSHOT_NAME):
    """The snapshot of this call, or `compute()` — and then the snapshot.

    A host function that is not pure answers through this, so running the same
    call again answers the value of the first run: the call is idempotent.
    """
    stored = read_snapshot(environ, name)
    if stored is not None:
        return stored
    value = compute()
    write_snapshot(environ, value, name)
    return value


def _storage(environ: Environment) -> EnvironmentStorage:
    storage = getattr(environ, "storage", None)
    if storage is None:
        raise RuntimeError("this environment has no storage to keep a snapshot in")
    return storage


def material(value) -> VibaNode:
    """A host value as material.

    A `VibaNode` as it is; an AST piece a host built (a product of tags and
    literals, say) gets the design written on it; a scalar becomes its leaf.
    """
    if isinstance(value, VibaNode):
        return value
    if isinstance(value, viba_ast.AST):
        return VibaNode(reflect_access,
                        descriptor_of(AstNodeType(value, _NO_MODULE)), value)
    if value is not None and not isinstance(value, (bool, int, float, str)):
        raise RuntimeError(f"cannot take {type(value).__name__} as material: "
                           f"only a VibaNode, an AST piece, a scalar, or None")
    node = viba_ast.Nil() if value is None else viba_ast.Constant(value)
    return VibaNode(reflect_access, descriptor_of(AstNodeType(node, _NO_MODULE)), node)


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
    if isinstance(answer, VibaProgramErr):
        raise RuntimeError(answer.err_msg)
    return answer.ok_value


def _stopped(result) -> bool:
    """True when a step carried no value on: a `VibaProgramErr`, or the deferral.

    `VibaProgramErr` says this run failed; `NotMyDutyException` says this run is asking for
    an implementation it does not have. Both stop the chain, and both travel
    back to the caller as they are.
    """
    return not isinstance(result, Ok)


def _no_implementation(step: Step, call) -> NotMyDutyException:
    """The deferral a host answers with: no implementation for that call.

    The step, the material it was given and why are all in it, so the side that
    answers next can write the work order without reading the run again.
    """
    return NotMyDutyException(step, call, REASON_NO_IMPLEMENTATION)


def _refused(deferred: NotMyDutyException, step: Step, call) -> NotMyDutyException:
    """What a `get_func` that raised the deferral is completed into.

    A host refusing a call need not know where it stands: the run fills in the
    step and the call it was about, and keeps whatever the host did say.
    """
    return NotMyDutyException(deferred.step or step,
                              deferred.call if deferred.call is not None else call,
                              deferred.reason or REASON_REFUSED)


# ----------------------------------------------------------------------
# interpret
# ----------------------------------------------------------------------


def interpret(viba_main_file: str, environ: Environment, get_file=None) -> InterpretResult:
    """Run `viba_main_file` with `environ`; its `__ret__` is the `Ok` value.

    What it answers is `Result[VibaNode]` with two more branches, both naming the
    step that stopped: `$not_my_duty_exception Duty`, when the compute side does
    not implement that step (a deferral, not a failure — the caller hands it on,
    and the duty carries the step, the material it was given and why), and
    `$underlying_viba_op_failed Failure`, when that step's implementation broke.
    `$viba_program_err str` is the rest: a program or an environment that cannot
    run at all.

    Where modules are looked up is the environment's business
    (`Environment.viba_path`): the directories are searched in order for
    `<name>.viba` (a dotted name as a path), and the directory of the file that
    wrote the import is searched first. A child environment keeps the parent's
    search path, so a module's own imports are looked up where the run says.

    `get_file` is where the source of a file comes from:
    `Optional[str <- $file_path str]`, the file's text for a path, `None` (or
    a `FileNotFoundError`) when that path has no file. Left out, the
    filesystem is read; given, nothing else is — a host can serve the whole
    run out of memory, a database, or anything else.
    """
    if not isinstance(environ, Environment):
        return VibaProgramErr("interpret needs an Environment")
    if environ.viba_path is not None and not isinstance(environ.viba_path,
                                                       (str, os.PathLike)):
        return VibaProgramErr(f"viba_path is a string of directories (or one path), "
                   f"not {type(environ.viba_path).__name__}")
    if get_file is not None and not callable(get_file):
        return VibaProgramErr(f"get_file is a function (or None), not {type(get_file).__name__}")
    return _Runner(environ.viba_path, get_file).run_file(viba_main_file, environ)


class _Runner:
    """One run: the files it has loaded, and where it looks for more."""

    def __init__(self, viba_path=None, get_file=None):
        text = "" if viba_path is None else os.fspath(viba_path)
        self.paths = [Path(p) for p in text.split(":") if p]
        self.get_file = get_file
        self.by_path: dict = {}        # normalized path -> module
        self.by_name: dict = {}        # module name -> module
        self.path_of: dict = {}        # module name -> file it was loaded from
        self.running: list = []        # module activations, for cycle refusal
        self.used_paths: set = set()   # storage paths module calls have run under

    def run_file(self, file: str, environ: Environment) -> InterpretResult:
        path = Path(file)
        source, problem = self._source(path)
        if problem is not None:
            return VibaProgramErr(problem)
        if source is None:
            return VibaProgramErr(f"no such file: {file}")
        module = self._file_of(path, path.stem, source)
        if _stopped(module):
            return module
        return _run_module(self, module.ok_value, environ, path.stem, str(path))

    # ---- where the source comes from ----

    def _source(self, path: Path):
        """(source, problem): the file's text, or why it cannot be read.

        `source` is None when that path has no file at all — the caller then
        looks somewhere else — and `problem` says the path was there and
        could not be used (unreadable, or the host broke).
        """
        if self.get_file is not None:
            try:
                source = self.get_file(str(path))
            except FileNotFoundError:
                return None, None
            except Exception as exc:            # the host's business, reported
                return None, f"get_file({path}) raised {exc!r}"
            if source is None:
                return None, None
            if not isinstance(source, str):
                return None, (f"get_file({path}) answered "
                              f"{type(source).__name__}, not the file's text")
            return source, None
        try:
            return path.read_text(), None
        except FileNotFoundError:
            return None, None
        except OSError as exc:
            return None, f"cannot read {path}: {exc}"

    def _key(self, path: Path) -> str:
        """What tells one file from another: the path, normalized. Not
        resolved — a host's file need not be on this filesystem at all."""
        return os.path.normpath(str(path))

    def _file_of(self, path: Path, name: str, source: str):
        """Parse one source, remember it under its path, bind it to `name`."""
        key = self._key(path)
        if key not in self.by_path:
            try:
                self.by_path[key] = custom_module(source)
            except SyntaxError as exc:
                return VibaProgramErr(f"cannot parse {path}: {exc}")
        return self._bind(self.by_path[key], path, name)

    def _bind(self, module, path: Path, name: str):
        self.by_name[name] = module
        self.path_of[name] = str(path)
        return Ok(module)

    def imported(self, name: str, near: Optional[str]):
        """The module `name`: loaded, or found next to `near`, or on the path."""
        if name in self.by_name:
            return Ok(self.by_name[name])
        for place in self._places(name, near):
            cached = self.by_path.get(self._key(place))
            if cached is not None:
                return self._bind(cached, place, name)
            source, problem = self._source(place)
            if problem is not None:
                return VibaProgramErr(problem)
            if source is None:
                continue                    # no file here: the next place
            return self._file_of(place, name, source)
        return VibaProgramErr(f"module {name!r} not found (next to {near} and on VIBA_PATH)")

    def _places(self, name: str, near: Optional[str]) -> list:
        """Where `name` may be, in the order it is looked for: next to the
        file that wrote the import first, then VIBA_PATH in order. A dotted
        name is a path, and also one file named with the dots (`pkg.inner.viba`).
        The same place twice is asked once."""
        rel = Path(*name.split(".")).with_suffix(".viba")
        places = []
        if near:
            places += [Path(near).parent / rel, Path(near).parent / f"{name}.viba"]
        for base in self.paths:
            places += [base / rel, base / f"{name}.viba"]
        out = []
        for place in places:
            if place not in out:
                out.append(place)
        return out


def _run_module(runner: _Runner, module: ModuleType, environ: Environment,
               name: str, file: Optional[str]) -> InterpretResult:
    """The module as a function: `environ` in, `__ret__` out.

    The storage path a module runs under is its identity: it is what the host
    is handed as `module_path`, so two activations under one path cannot be
    told apart. No two module calls may share one — the caller gives each call
    a sub-environment of its own.
    """
    if file is None:
        file = runner.path_of.get(name)     # a module called through an import
    ret = _definition(module, RET_NAME)
    if ret is None:
        return VibaProgramErr(f"module {name!r} has no {RET_NAME}: it is design, not a program")
    if name in runner.running:
        return VibaProgramErr(f"module {name!r} is already running: a module call cycle")
    path = _storage_path(environ)
    if path in runner.used_paths:
        return VibaProgramErr(f"module {name!r} was handed the storage path {path!r}, which "
                   f"another module call already used: give each module call a "
                   f"sub-environment of its own (environ.sub_env << ...)")
    runner.used_paths.add(path)
    runner.running.append(name)
    try:
        value = _Activation(runner, module, environ, name, file).evaluate(ret.body)
    finally:
        runner.running.pop()
    if _stopped(value):
        return value
    if not isinstance(value.ok_value, (_Material, _Host)):
        return VibaProgramErr(f"{name}.{RET_NAME} is a function still waiting for arguments")
    return Ok(_argument_value(value.ok_value))


def _storage_path(environ: Environment) -> str:
    """The path the host is handed for this environment: a module's identity."""
    storage = getattr(environ, "storage", None)
    return getattr(storage, "cur_storage_path", "") if storage else ""


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
        """Result: the value this piece writes — or why the chain stopped: an
        `VibaProgramErr`, or the deferral of a step nobody here implements.

        A piece of data written where a value goes is material as it stands:
        a literal or a unit, a tuple, and a product of tags and literals —
        `$victim ($x 0 * $y 0) * $at "12:30"` is a witness, the same spelling
        its type would have. Its members are data, not calls, so nothing in
        them is evaluated. A sum is not: which branch would it be.
        """
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never,
                             viba_ast.Any, viba_ast.Tuple, viba_ast.Tagged,
                             viba_ast.Product, viba_ast.ProductChain)):
            return Ok(_Material(VibaNode(reflect_access, self._descriptor(node), node)))
        if isinstance(node, viba_ast.Partial):
            return self._apply_chain(node)
        if isinstance(node, viba_ast.TypeRef):
            return self._resolve(node.name)
        if isinstance(node, viba_ast.CodeBlock):
            return VibaProgramErr("a code block is documentation: it is not a value")
        return VibaProgramErr(f"cannot compute {type(node).__name__}")

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
            if _stopped(imported):
                return imported
            if rest:
                return self._member_of(imported.ok_value, module_name, rest)
            return Ok(_ModuleFunc(self.runner, imported.ok_value, module_name))
        return VibaProgramErr(f"no definition named {name!r} in module {self.name!r}")

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
            return VibaProgramErr(f"module {module_name!r} has no {rest!r}")
        other = _Activation(self.runner, module, self.environ, module_name,
                            self.runner.path_of.get(module_name))
        return other._defined(rest, definition)

    def _environ_member(self, rest: str):
        member = getattr(self.environ, rest, None)
        if not callable(member):
            return VibaProgramErr(f"the environment has no {rest!r}")
        return Ok(_HostFunction(rest, member, slots=1,
                                module_path=_storage_path(self.environ)))

    # ---- calls ----

    def _apply_chain(self, node):
        """Give the written arguments to the function, in written order.

        The chain nests left: `((f << a) << b) << c` is read by walking the
        spine, which meets c first. The function is evaluated, then the
        arguments the other way round, so a call runs left to right — that is
        the order the host sees them in, and the order side effects happen in.
        """
        written = []
        while isinstance(node, viba_ast.Partial):
            written.append(node.argument)
            node = node.function
        function = self.evaluate(node)
        if _stopped(function):
            return function
        current = function.ok_value
        for argument in reversed(written):
            value = self._argument(argument)
            if _stopped(value):
                return value
            if value.ok_value is None:
                continue                        # documentation is no argument
            current = _give(current, value.ok_value)
            if _stopped(current):
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
        if _stopped(value):
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
        return VibaProgramErr(f"{function.node!r} is not a function: it is a value")
    if isinstance(function, _Host):
        return VibaProgramErr(f"{type(function.obj).__name__} is not a function")
    return VibaProgramErr(f"{type(function).__name__} is not a function")


class _Callable:
    """The viba values a host may call: one `as_callable` between them."""

    def as_callable(self):
        """This function as a host callable: the host hands over the values it
        wants given, in the function's own order, and gets the answer back.

        A call that runs into a missing implementation crosses as the deferral
        itself, so a host can tell "not mine" from "broke" the same way the run
        does; a `VibaProgramErr` raises, since there is nothing to carry on with.
        """
        def call(*values):
            current = self
            for value in values:
                given = _give(current, _Given(None, _given_value(value)))
                if isinstance(given, NotMyDutyException):
                    raise given
                if isinstance(given, UnderlyingVibaOpFailed):
                    raise given
                if isinstance(given, VibaProgramErr):
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
                return VibaProgramErr(f"{self.name} takes no {item.tag} argument")
            given[slots.index(item.tag)] = item.value
        else:
            # An argument written without a tag is the next slot that is free:
            # the caller did not say which one, and the order is the design's.
            free = [index for index in range(len(slots)) if index not in given]
            if not free:
                return VibaProgramErr(f"{self.name} takes no more arguments")
            given[free[0]] = item.value
        if all(index in given for index in range(len(slots))):
            return self.call(given)
        return Ok(_VibaFunc(self.activation, self.name, self.chain, given))

    def call(self, given):
        """Every slot is filled: the environment's compute side implements it."""
        problem = self._environ_problem(given)
        if problem is not None:
            return VibaProgramErr(problem)
        environ = given[self.slots.index(ENVIRON_TAG)].obj
        compute = getattr(environ, "compute", None)
        if compute is None:
            return VibaProgramErr(f"{self.name}: the environment carries no compute side")
        module_path = _storage_path(environ)
        step = Step(module_path, self.name)
        try:
            host = compute.get_func(module_path, self.name)
        except NotMyDutyException as deferred:   # the host refuses this call
            return _refused(deferred, step, self._call_material(given))
        except Exception as exc:            # the host is the host's business
            return UnderlyingVibaOpFailed(f"get_func({module_path!r}, {self.name!r}) raised {exc!r}",
                          step, REASON_GET_FUNC_RAISED)
        if host is None:
            return _no_implementation(step, self._call_material(given))
        args = [_argument_value(given[index]) for index, _ in self._ordered(given)]
        try:
            answer = host(*args)
        except NotMyDutyException as deferred:   # a viba call inside deferred
            return deferred
        except UnderlyingVibaOpFailed as failure:                # ... or failed inside
            return failure
        except Exception as exc:
            return UnderlyingVibaOpFailed(f"{self.name} raised {exc!r}", step, REASON_RAISED)
        return _answer(self.name, answer, step)

    def _call_material(self, given):
        """The material this call was given, as it was written.

        A host value — the environment above all — is no material and does not
        travel: the side that answers makes its own. One material argument is
        that argument itself (no tag is needed to tell it from the others),
        which is the `$call` a Prepare of such a call fixes; several make a
        product, keeping the tags as written. None when the call was given no
        material at all.
        """
        material_given = [(self.slots[index], value.node.data)
                          for index, value in self._ordered(given)
                          if isinstance(value, _Material)]
        if not material_given:
            return None
        if len(material_given) == 1:
            return material(material_given[0][1])
        written = [viba_ast.Tagged(tag, piece) if tag else piece
                   for tag, piece in material_given]
        return material(viba_ast.ProductChain(written))

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

    def __init__(self, name: str, func, slots: int = 1, given=None,
                 module_path: str = ""):
        self.name = name
        self.func = func
        self.slots = slots
        self.given = list(given or [])
        self.module_path = module_path

    def give(self, item):
        values = self.given + [_argument_value(item.value)]
        step = Step(self.module_path, f"environ.{self.name}")
        if len(values) < self.slots:
            return Ok(_HostFunction(self.name, self.func, self.slots, values,
                                    self.module_path))
        try:
            answer = self.func(*values)
        except NotMyDutyException as deferred:
            return _refused(deferred, step, None)
        except UnderlyingVibaOpFailed as failure:
            return failure
        except Exception as exc:
            return UnderlyingVibaOpFailed(f"environ.{self.name} raised {exc!r}", step, REASON_RAISED)
        return _answer(f"environ.{self.name}", answer, step)


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
            return VibaProgramErr(f"module {self.name!r} needs an Environment")
        answer = _run_module(self.runner, self.module, environ, self.name, None)
        if _stopped(answer):
            return answer
        # `interpret` hands the node out; inside a run a module's answer is a
        # value like any other, so it goes back into the value model.
        node = answer.ok_value
        return Ok(_Material(node) if isinstance(node, VibaNode) else _Host(node))


def _answer(name, answer, step: Step = None):
    """Result: what a host function answered, as a value.

    A `VibaNode` is taken as it is, an `Environment` stays a host value, and
    `None` is `nil` the way it is in the builder. A plain Python value lands
    as a leaf — but only a scalar one: a list, a dict, a callable or any other
    object has no leaf to be, and guessing one would put a piece into the
    material that no design asked for. Given a `step`, that refusal is a
    failure of it; without one — a value a host is handing back into a call —
    it is a plain `VibaProgramErr`.
    """
    if isinstance(answer, VibaNode):
        return Ok(_Material(answer))
    if isinstance(answer, Environment):
        return Ok(_Host(answer))
    if answer is not None and not isinstance(answer, (bool, int, float, str)):
        msg = (f"{name} answered {type(answer).__name__}, "
               f"which is no leaf: answer a VibaNode, a scalar, or None")
        if step is not None:
            return UnderlyingVibaOpFailed(msg, step, REASON_NO_LEAF)
        return VibaProgramErr(msg)
    node = viba_ast.Nil() if answer is None else viba_ast.Constant(answer)
    return Ok(_Material(VibaNode(reflect_access,
                                 descriptor_of(AstNodeType(node, _NO_MODULE)), node)))


def _elements(node):
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _elements(node.result) + [node.argument]
    return [node]


__all__ = ["interpret", "Environment", "EnvironmentStorage", "EnvironmentCompute",
           "material", "snapshot_path", "read_snapshot", "write_snapshot", "replayed"]
