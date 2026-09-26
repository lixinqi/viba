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
from viba.partial import marked_function, product_elements
from viba.reflect import VibaNode, access as reflect_access
from viba.type import (CustomModuleType, REASON_GET_FUNC_RAISED, REASON_NO_IMPLEMENTATION, REASON_NO_LEAF,
                       REASON_RAISED, REASON_REFUSED, AstNodeType, VibaProgramErr, UnderlyingVibaOpFailed,
                       InterpretResult, ModuleType, NotMyDutyException, Ok, Step,
                       BUILTIN_MODULE, NilType, NeverType, custom_module)
from viba.viba_type_descriptor import descriptor_of

# A scalar a host answers belongs to no file: its leaf gets an empty module.
# Parsed once, not once per answer.
_NO_MODULE = custom_module("")

TMP_PREFIX = "tmp_"
RET_NAME = "__ret__"
ARGS_NAME = "__args__"
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


def _is_never(value) -> bool:
    """Whether `value` is the additive unit `never`."""
    return isinstance(value, _Material) and isinstance(value.node.data, viba_ast.Never)


def _is_nil(value) -> bool:
    """Whether `value` is the multiplicative unit `nil`."""
    return isinstance(value, _Material) and isinstance(value.node.data, viba_ast.Nil)


def _one_line(node) -> str:
    """A piece as one line: error messages read better without the layout."""
    return " ".join(viba_ast.unparse_type(node).split())


def _slot_type(element):
    """The declared type a written argument slot asks for."""
    return element.type if isinstance(element, viba_ast.Tagged) else element


def _value_as_type(value):
    """The type this value already is, or None when it has none written here.

    Material is the design it was made of and a viba function is its chain, so
    the judgment can read them; the environment is `Environment`. A host value
    with no viba type — a host function, a host object — has none, and nothing
    is judged for it.
    """
    if isinstance(value, _Material):
        return value.node.data
    if isinstance(value, _Material):
        return value.node.data
    if isinstance(value, _Host) and isinstance(value.obj, Environment):
        return viba_ast.TypeRef(ENVIRON_TYPE)
    return None


def _fits_slot(value, element, module, owner: str):
    """Why this argument does not fit the slot it was given to, or None.

    The design says what a slot is (`$a int`), and a value that is written out
    already has a type, so the judgment that reads the design can refuse the
    argument before any host sees it: a string in an int slot is a program
    error, not a step whose implementation broke. Only what can be read both
    ways is judged — a value whose type the judgment cannot settle is left to
    whoever implements the step, the way it always was.
    """
    given = _value_as_type(value)
    if given is None:
        return None
    written = _slot_type(element)
    from viba.is_sub_type import is_sub_type
    judged = is_sub_type(AstNodeType(given, module), AstNodeType(written, module))
    if not isinstance(judged, Ok) or judged.ok_value is True:
        return None
    return (f"{owner}: {_one_line(given)} does not fit {_one_line(element)}: "
            f"{_one_line(given)} <: {_one_line(written)} does not hold")


def _builtin_unit(node):
    """The builtin unit type written by `node`, if it names one."""
    if isinstance(node, viba_ast.Nil):
        return NilType
    if isinstance(node, viba_ast.Never):
        return NeverType
    if not isinstance(node, viba_ast.TypeRef):
        return None
    builtin = BUILTIN_MODULE.lookup(node.name)
    if isinstance(builtin, Ok) and isinstance(builtin.ok_value, (NilType, NeverType)):
        return type(builtin.ok_value)
    return None


def _never_material():
    """The `never` value as material."""
    node = viba_ast.Never()
    return _Material(VibaNode(reflect_access,
                              descriptor_of(AstNodeType(node, _NO_MODULE)), node))


def _sum_branches(node):
    """The branches of a written sum, flattened from the left-nested `|` tree."""
    if isinstance(node, viba_ast.Sum):
        return _sum_branches(node.left) + _sum_branches(node.right)
    if isinstance(node, viba_ast.SumChain):
        return list(node.elements)
    return [node]


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


class _Raised(Exception):
    """A `Result` a getter has to cross a host call with.

    The host boundary speaks exceptions, so a getter that stops carries the
    whole answer — the program error, the deferral, the failure — and the call
    hands that answer back unchanged: what stopped is the argument that was
    asked for, not the host function that asked.
    """

    def __init__(self, result):
        super().__init__(repr(result))
        self.result = result


class _Getter:
    """One written argument of a lazy call, computed only when it is wanted.

    A marked function (`ParametersLazyEvaluated[F]`) is handed these instead of
    values: the host calls the ones it needs, and the argument expressions it
    does not call are never evaluated at all. Calling one answers what the host
    would have been handed eagerly — material as its `VibaNode`, the
    environment as itself — or raises `_Raised` with whatever the run stopped
    with.

    **At most once**: the answer, value or stop, is worked out on the first call
    and handed back on every later one. An argument that is asked for twice is
    still one argument — `get_v()` twice is as eager as writing `v` twice, which
    is what the host would have been handed. Asking again cannot make a side
    effect happen twice.
    """

    __slots__ = ("activation", "node", "scope", "answer", "slot", "module", "owner")

    def __init__(self, activation, node, scope=()):
        self.activation = activation
        self.node = node
        self.scope = scope                  # the block the argument was written in
        self.answer = None                  # the Result, once it is worked out
        self.slot = None                    # what the call asked for, once it is taken
        self.module = None
        self.owner = ""

    def watch(self, element, module, owner: str):
        """The call that took this argument says which slot it fills.

        A getter is handed over before the value exists, so the type the slot
        asks for can only be checked when the host asks for it. `_Pending.give`
        knows the slot; this is where it tells the getter.
        """
        self.slot = element
        self.module = module
        self.owner = owner

    def __call__(self):
        if self.answer is None:
            answer = self.activation.evaluate(self.node, self.scope)
            if isinstance(answer, Ok) and self.slot is not None:
                problem = _fits_slot(answer.ok_value, self.slot, self.module, self.owner)
                if problem is not None:
                    answer = VibaProgramErr(problem)
            self.answer = answer
        if not isinstance(self.answer, Ok):
            raise _Raised(self.answer)
        return _argument_value(self.answer.ok_value)


def _in_scope(scope, name):
    """The value `name` is bound to in this scope, or None (a value is never
    None: `nil` is material)."""
    for frame in reversed(scope):
        if name in frame:
            return frame[name]
    return None


def _let_inside(node):
    """The first binding block written inside a piece of material, or None."""
    for part in viba_ast.walk(node):
        if isinstance(part, viba_ast.Let):
            return part
    return None


def _addressed(node):
    """(tag, inner) of a written argument: the tag it is addressed by, if any."""
    if isinstance(node, viba_ast.Tagged):
        return node.tag, node.type
    return None, node


def _getter(value) -> "_Getter":
    """`value` as a getter: a `_Getter` as it is, anything else answered back."""
    if isinstance(value, _Getter):
        return value

    def get():
        return _argument_value(value)

    return get


def _argument_value(value):
    """What a host function is handed.

    Material arrives as its node, a viba function as a Python callable (the
    host calls it like any other function: the values it is called with are
    nodes or host objects, and the answer is a node or a host object), and
    anything else as itself — the environment among them.
    """
    if isinstance(value, _Material):
        return value.node
    if isinstance(value, _HostFunction):
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
        self.computing: list = []      # (module, definition) being computed, innermost last

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
                tree = viba_ast.parse(source)
            except SyntaxError as exc:
                return VibaProgramErr(f"cannot parse {path}: {exc}")
            imports = {stmt.alias or stmt.module: stmt.module
                       for stmt in tree.body if isinstance(stmt, viba_ast.Import)}
            # 这份模块自己知道怎么找它的 import：描述符/判定层要用（它们不经过
            # `_Runner.imported`），而按文件找模块这件事仍然由 run 说了算。
            module = custom_module(source)
            module.module_environment = lambda asked, near=str(path): self.imported(asked, near)
            module.imports = imports
            self.by_path[key] = module
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
               name: str, file: Optional[str], args=None) -> InterpretResult:
    """The module as a function: `environ` in, `__ret__` out.

    A module that declares `__args__` is called with its members as well; `args`
    is the product they were given as, and inside the module that product is what
    `__args__` (and every name that aliases it) means (viba-interpreter.md).

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
        value = _Activation(runner, module, environ, name, file, args).evaluate(ret.body)
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


def _module_arg_slots(module: ModuleType):
    """(slots, problem) for a module's `__args__`, or (None, None) for none.

    `__args__` is a product type and its members are the call's arguments, in
    written order: `__args__ = Object * $a int * $b int` is a module called with
    two. `__args__ = Object` declares a module that still takes none, and a
    `__args__` that is no product at all is a program error, reported at the
    call.
    """
    definition = _definition(module, ARGS_NAME)
    if definition is None:
        return None, None
    body = definition.body
    if _builtin_unit(body) is NilType:
        return [], None
    if not isinstance(body, (viba_ast.Product, viba_ast.ProductChain)):
        return None, (f"{ARGS_NAME} is not a product: {viba_ast.unparse_type(body)}")
    slots = []
    for factor in product_elements(body):
        if _builtin_unit(factor) is NilType:
            continue                        # the head, not a member
        if isinstance(factor, viba_ast.Tagged):
            slots.append((factor.tag, factor.type))
        else:
            slots.append((None, factor))    # a member with no tag: position only
    return slots, None


def _material_factors(node: VibaNode):
    """The factors of a product material, in written order."""
    data = node.data
    if isinstance(data, (viba_ast.Product, viba_ast.ProductChain)):
        return product_elements(data)
    return [data]


def _definition(module: ModuleType, name: str):
    for node in module.module.body:
        if getattr(node, "name", None) == name:
            return node
    return None


class _Activation:
    """One module running with one environment."""

    def __init__(self, runner: _Runner, module: ModuleType, environ: Environment,
                 name: str, file: Optional[str], args=None):
        self.runner = runner
        self.module = module
        self.environ = environ
        self.name = name
        self.file = file
        self.args = args                     # the `__args__` product, if any
        self.defined: dict = {}

    # ---- expressions ----

    def evaluate(self, node, scope=()):
        """Result: the value this piece writes — or why the chain stopped: an
        `VibaProgramErr`, or the deferral of a step nobody here implements.

        A piece of data written where a value goes is material as it stands:
        a literal or a unit, a tuple, and a product of tags and literals —
        `$victim ($x 0 * $y 0) * $at "12:30"` is a witness, the same spelling
        its type would have. Its members are data, not calls, so nothing in
        them is evaluated. A sum is not: which branch would it be.

        `scope` is what a binding block put around this piece: the frames of the
        blocks it is written inside, innermost last. A definition's own body
        runs with the empty scope, so a binding never reaches past its block.
        """
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never,
                             viba_ast.Any, viba_ast.Tuple, viba_ast.Tagged)):
            inside = _let_inside(node)
            if inside is not None:
                return VibaProgramErr(
                    f"a binding belongs in a value, not inside material: "
                    f"{viba_ast.unparse_type(inside)}")
            return Ok(_Material(VibaNode(reflect_access, self._descriptor(node), node)))
        if isinstance(node, (viba_ast.Product, viba_ast.ProductChain)):
            return self._product(node, scope)
        if isinstance(node, (viba_ast.Sum, viba_ast.SumChain)):
            return self._sum(node, scope)
        if isinstance(node, viba_ast.Let):
            return self._let(node, scope)
        if isinstance(node, viba_ast.Partial):
            return self._apply_chain(node, scope)
        if isinstance(node, viba_ast.TypeRef):
            return self._resolve(node.name, scope)
        if isinstance(node, viba_ast.CodeBlock):
            return VibaProgramErr("a code block is documentation: it is not a value")
        return VibaProgramErr(f"cannot compute {type(node).__name__}")

    def _descriptor(self, node):
        return descriptor_of(AstNodeType(node, self.module))

    def _product(self, node, scope=()):
        """Evaluate a product: never absorbs, while nil disappears."""
        kept = []
        for factor in product_elements(node):
            if _builtin_unit(factor) is NilType:
                continue
            value = self.evaluate(factor, scope)
            if _stopped(value):
                return value
            answered = value.ok_value
            if _is_never(answered):
                return Ok(_never_material())
            if _is_nil(answered):
                continue
            kept.append(answered)
        if not kept:
            return Ok(_Material(material(None)))
        if len(kept) == 1:
            return Ok(kept[0])
        chain = viba_ast.ProductChain([factor.node.data for factor in kept])
        return Ok(_Material(VibaNode(reflect_access, self._descriptor(chain), chain)))

    def _sum(self, node, scope=()):
        """Evaluate a written sum, dropping the branches that answered never.

        A sum here is `if/else`: each branch answers either its value or never.
        The branches that answered never are the ones not taken, so they are
        dropped; what is left is the taken branch. None left means every branch
        dropped — the whole sum is never.
        """
        kept = []
        for branch in _sum_branches(node):
            if _builtin_unit(branch) is NeverType:
                continue                        # the chain head, not a branch
            value = self.evaluate(branch, scope)
            if _stopped(value):
                return value
            answered = value.ok_value
            if _is_never(answered):
                continue
            kept.append(answered)
        if not kept:
            return Ok(_never_material())
        if len(kept) == 1:
            return Ok(kept[0])
        elements = [branch.node.data for branch in kept]
        chain = viba_ast.SumChain(elements)
        return Ok(_Material(VibaNode(reflect_access, self._descriptor(chain), chain)))

    def _imports(self) -> dict:
        """The file's import table: what each import binds, and the module it
        names (`import a.b as c` binds c, `import a.b` binds a.b)."""
        return {stmt.alias or stmt.module: stmt.module
                for stmt in self.module.module.body
                if isinstance(stmt, viba_ast.Import)}

    def _let(self, node, scope):
        """A binding block: the bindings in written order, then the result.

        The names live in a frame of their own, so the result sees them — and
        so does anything the block hands out, a lazy argument's getter above
        all — while the module around the block does not. The bindings are
        computed as they are written, used or not; a name bound twice in one
        block is the later binding.
        """
        frame = {}
        inner = scope + (frame,)
        for binding in node.bindings:
            value = self.evaluate(binding.value, inner)
            if _stopped(value):
                return value
            frame[binding.name] = value.ok_value
        return self.evaluate(node.body, inner)

    def _resolve(self, name: str, scope=()):
        bound = _in_scope(scope, name)
        if bound is not None:
            return Ok(bound)
        if name == ENVIRON_NAME:
            return Ok(_Host(self.environ))
        if name == ARGS_NAME and self.args is not None:
            # Read as a type `__args__` is the product of the call's arguments;
            # read as a value inside the module it is that product (the one the
            # call was given), which is why a name aliasing it works too.
            return Ok(self.args)
        if name.startswith(ENVIRON_NAME + "."):
            return self._environ_member(name[len(ENVIRON_NAME) + 1:])
        definition = _definition(self.module, name)
        if definition is not None:
            return self._value_of_definition(name, definition, self.module)
        bound = self._imported_name(name)
        if bound is not None:
            module_name, rest = bound
            imported = self.runner.imported(module_name, self.file)
            if _stopped(imported):
                return imported
            if rest:
                return self._member_of(imported.ok_value, module_name, rest, name)
            node = viba_ast.TypeRef(name)
            return Ok(_Material(VibaNode(
                reflect_access, descriptor_of(AstNodeType(node, self.module)), node)))
        member = self._tagged_member(name, scope)
        if member is not None:
            return member
        return VibaProgramErr(f"no definition named {name!r} in module {self.name!r}")

    def _tagged_member(self, name: str, scope=()):
        """`args.a`: the `$a` member of a product this module names.

        A dotted name is a member of an imported module (`demo.print`), which is
        settled before this; when the head is not an import, it is the tag of a
        member of the product the head stands for — the arguments of the call
        among them (`args.a`, `__args__.a`). None when the head is no such
        value, so the caller reports the name the way it always did.
        """
        head, dot, tag = name.rpartition(".")
        if not dot or not head:
            return None
        value = self._resolve(head, scope)
        if _stopped(value) or not isinstance(value.ok_value, _Material):
            return None
        wanted = "$" + tag
        for factor in _material_factors(value.ok_value.node):
            if isinstance(factor, viba_ast.Tagged) and factor.tag == wanted:
                inner = factor.type          # the member's value, not its address
                return Ok(_Material(VibaNode(
                    reflect_access, descriptor_of(AstNodeType(inner, self.module)), inner)))
        return VibaProgramErr(f"{head!r} has no member tagged {wanted!r}")

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
        """The value this definition stands for, computed once.

        A definition that is asked for while it is still being computed is a
        definition that goes round: one file's definitions may not form a cycle,
        so this is a program error — recursion is what two files do, and it is
        the module call that carries it (viba-interpreter.md).
        """
        if name in self.defined:
            return self.defined[name]
        mine = (self.name, name)
        if mine in self.runner.computing:
            return VibaProgramErr(self._goes_round(name))
        self.runner.computing.append(mine)
        try:
            value = self._compute(name, definition)
        finally:
            self.runner.computing.pop()
        self.defined[name] = value
        return value

    def _goes_round(self, name: str) -> str:
        """What went round, as the path that came back to `name`.

        Inside one file this is the rule: a file's own definitions may not form
        a cycle, which is why a file alone cannot recurse. Across files the
        design is left alone — two files may call each other — but a run that
        comes back to a definition still being computed has nowhere to stop, so
        it is reported rather than left to the interpreter's own stack.
        """
        mine = (self.name, name)
        path = self.runner.computing[self.runner.computing.index(mine):] + [mine]
        if len({module for module, _defined in path}) == 1:
            written = " -> ".join(defined for _module, defined in path)
            return (f"{written}: one file's definitions may not go round — "
                    f"recursion takes two files")
        written = " -> ".join(f"{module}.{defined.split('.')[-1]}"
                              for module, defined in path)
        return f"{written}: the run came back to where it started"

    def _compute(self, name: str, definition):
        # 函数名不经过这里：它当一个值读时是它代表的那个闭包（见
        # `_value_of_definition`），当链头读时是一次调用（见 `_pending_of`）。
        return self.evaluate(definition.body)

    def _member_of(self, module, module_name: str, rest: str, written: str = ""):
        """`demo.print`: a definition of an imported module, read as a value.

        The value is computed in the module the definition belongs to (that is
        where its own names resolve), while a closure made of it is written the
        way the caller wrote it.
        """
        definition = _definition(module, rest)
        if definition is None:
            return VibaProgramErr(f"module {module_name!r} has no {rest!r}")
        other = _Activation(self.runner, module, self.environ, module_name,
                            self.runner.path_of.get(module_name))
        return other._value_of_definition(rest, definition, module,
                                          home=self.module, written=written or rest)

    def _value_of_definition(self, name, definition, owner_module, home=None,
                             written=None):
        """A name read as a value: a function is the closure it stands for, which
        is written as the name itself. Everything else is its body, computed.

        `name` is the local definition (the name this module knows it by, which
        is also how the cycle guard counts it); `written` is how the caller wrote
        it, which is what a closure made of it says.
        """
        home = home or self.module
        body = definition.body
        if isinstance(body, (viba_ast.Exponent, viba_ast.ExponentChain)) or \
                marked_function(body, owner_module) is not None:
            node = viba_ast.TypeRef(written or name)
            return Ok(_Material(VibaNode(
                reflect_access, descriptor_of(AstNodeType(node, home)), node)))
        return self._defined(name, definition)

    def _environ_member(self, rest: str):
        member = getattr(self.environ, rest, None)
        if not callable(member):
            return VibaProgramErr(f"the environment has no {rest!r}")
        return Ok(_HostFunction(rest, member, slots=1,
                                module_path=_storage_path(self.environ)))

    # ---- calls ----

    def _apply_chain(self, node, scope=()):
        """Give the written arguments to what stands at the head of the chain.

        The chain nests left: `((f << a) << b) << c` is read by walking the
        spine, which meets c first. It is read apart into (head, arguments in
        written order), the head is resolved, and then each argument is given in
        that order — the order the host sees them in, and the order side effects
        happen in. A closure stored as material is that same written chain, so
        applying more arguments to one is reading it apart again and carrying on.
        """
        target, written = self._target_and_arguments(node, scope)
        if written is None:
            return target
        current = target.ok_value
        for argument in written:
            lazy = isinstance(current, _Pending) and current.lazy \
                and not current.is_environment_slot(argument)
            value = (self._lazy_argument(argument, scope) if lazy
                     else self._argument(argument, scope))
            if _stopped(value):
                return value
            if value.ok_value is None:
                continue                        # documentation is no argument
            if isinstance(current, _Pending):
                current = current.give(value.ok_value.tag, value.ok_value.value)
            else:
                current = _host_give(current, value.ok_value)
            if _stopped(current):
                return current
            current = current.ok_value
        if isinstance(current, _Pending):
            return self._finish(current)
        return Ok(current)

    def _finish(self, pending):
        """The chain ended: run it when the environment is in, store it when not.

        Giving the environment is what execution is, so a pending call that has
        one is a call being made — its arguments have to be complete. A pending
        call without one is a value: a closure, written down and serializable.
        """
        if pending.environ is None:
            return pending.materialize()
        missing = pending.missing()
        if missing is not None:
            return VibaProgramErr(missing)
        return pending.fire()

    def _target_and_arguments(self, node, scope):
        """(what the chain calls, its written arguments) — or (a Result, None).

        A name at the head is read as the definition it names, not as the value
        it stands for: `add` at the head is that call, while `add` in an argument
        is the closure. A closure stored as material is the chain it was made
        from, so it is read apart here, its arguments coming first.
        """
        head, arguments = _call_parts(node)
        while True:
            if isinstance(head, viba_ast.TypeRef):
                # None 是"这个名字不是一次调用"，不是"停下了"：停下只有 Result 能表达。
                target = self._call_target(head, scope)
                if target is not None:
                    if _stopped(target):
                        return target, None
                    return target, arguments
            value = self.evaluate(head, scope)
            if _stopped(value):
                return value, None
            got = value.ok_value
            if isinstance(got, _Material) and isinstance(got.node.data, viba_ast.Partial):
                head, stored = _call_parts(got.node.data)
                arguments = stored + arguments
                continue
            if isinstance(got, _Material) and isinstance(got.node.data, viba_ast.TypeRef):
                head = got.node.data           # a name kept as a value
                continue
            return Ok(got), arguments

    def _call_target(self, name_node, scope):
        """The call a written name stands for, or None when it is no call.

        A definition that is a function is the call; a bare import name is the
        module, whose environment runs it. Everything else is not a call here and
        the caller reads the name as a value instead.
        """
        name = name_node.name
        if name == ENVIRON_NAME or name.startswith(ENVIRON_NAME + "."):
            return None                        # 宿主那一侧按值读（另有安排）
        definition = _definition(self.module, name)
        if definition is not None:
            return self._pending_of(definition, name_node, name, self.module, name)
        bound = self._imported_name(name)
        if bound is None:
            return None
        module_name, rest = bound
        imported = self.runner.imported(module_name, self.file)
        if _stopped(imported):
            return imported
        if rest:
            return self._member_target(imported.ok_value, module_name, rest, name_node,
                                       name, home=self.module)
        return Ok(_Pending.module(self.runner, imported.ok_value, module_name,
                                  name_node, home=self.module))

    def _pending_of(self, definition, name_node, name, owner_module, written,
                    local_name="", home=None):
        """A definition read as the call it stands for, or None when it is no
        function (then the name is a plain value). `written` is how it was
        written, `local_name` how the host knows it."""
        body = definition.body
        if isinstance(body, (viba_ast.Exponent, viba_ast.ExponentChain)):
            return self._func_pending(name_node, written, owner_module, body,
                                      local_name or name, home=home)
        chain = marked_function(body, owner_module)
        if chain is None:
            return None
        if not isinstance(chain, (viba_ast.Exponent, viba_ast.ExponentChain)):
            return VibaProgramErr(
                f"{name}: ParametersLazyEvaluated marks a function, not "
                f"{viba_ast.unparse_type(chain)}")
        return self._func_pending(name_node, written, owner_module, chain,
                                  local_name or name, lazy=True, home=home)

    def _func_pending(self, name_node, written, owner_module, chain, name,
                      lazy=False, home=None):
        """The call a function stands for. Every executable function depends on
        the environment, so a chain without that slot can never run: saying so
        here is what keeps such a call from becoming a closure that never runs."""
        slots = _slots_of(chain)
        if not any(isinstance(one, viba_ast.Tagged) and one.tag == ENVIRON_TAG
                   for one in slots):
            return VibaProgramErr(
                f"{written} takes no {ENVIRON_TAG} {ENVIRON_TYPE} argument: "
                f"every executable function depends on the environment")
        return Ok(_Pending.func(self, name_node, written, owner_module, slots,
                                lazy=lazy, name=name, home=home))

    def _member_target(self, module, module_name, rest, name_node, written,
                       home=None):
        """`demo.print` read as the call it stands for."""
        definition = _definition(module, rest)
        if definition is None:
            return VibaProgramErr(f"module {module_name!r} has no {rest!r}")
        other = _Activation(self.runner, module, self.environ, module_name,
                            self.runner.path_of.get(module_name))
        return other._pending_of(definition, name_node, rest, module, written,
                                 home=home)

    def _argument(self, node, scope=()):
        """Result: the `_Given` this argument is, or Ok(None) for documentation."""
        if isinstance(node, viba_ast.CodeBlock):
            return Ok(None)
        tag, inner = _addressed(node)
        value = self.evaluate(inner, scope)
        if _stopped(value):
            return value
        return Ok(_Given(tag, value.ok_value))

    def _lazy_argument(self, node, scope=()):
        """Result: the `_Given` this argument is, with its value not computed.

        What travels is a getter; the host calls it if and when it wants that
        argument (see `ParametersLazyEvaluated`).
        """
        if isinstance(node, viba_ast.CodeBlock):
            return Ok(None)
        tag, inner = _addressed(node)
        return Ok(_Given(tag, _Getter(self, inner, scope)))


def _slots_of(chain):
    """A written function's argument slots: documentation is no argument."""
    return [element for element in _elements(chain)[1:]
            if not isinstance(element, viba_ast.CodeBlock)]


def _call_parts(node):
    """A written call as (head, arguments in written order).

    A closure made of material is this same shape, which is why applying more
    arguments to a stored closure is only reading it apart again.
    """
    written = []
    while isinstance(node, viba_ast.Partial):
        written.append(node.argument)
        node = node.function
    return node, list(reversed(written))


def _host_give(function, item):
    """Give one argument to what the environment handed over, or say why not.

    A member of the environment counts its own slots and takes the values in
    written order. Anything else that was handed an argument is named by its
    kind, and the message stays the same between runs — an object's repr would
    carry an address that changes every time.
    """
    if isinstance(function, _HostFunction):
        return function.give(item)
    if isinstance(function, _Material):
        return VibaProgramErr(f"{function.node!r} is not a function: it is a value")
    if isinstance(function, _Host):
        return VibaProgramErr(f"{type(function.obj).__name__} is not a function")
    return VibaProgramErr(f"{type(function).__name__} is not a function")


def _is_environ_value(value) -> bool:
    """Whether this value is an environment — giving one is what execution is."""
    return isinstance(value, _Host) and isinstance(value.obj, Environment)


def _is_empty_product(value) -> bool:
    """Whether this value is the written empty product, `()`."""
    return (isinstance(value, _Material)
            and isinstance(value.node.data, viba_ast.Tuple)
            and not value.node.data.elements)


class _Pending:
    """A call being prepared: a function or a module, and what it has been given.

    It lives inside one chain. When the chain ends it either runs (the
    environment is in, so the arguments must be complete) or becomes material —
    the function's name and the arguments already computed, which is a closure.
    That is why a half-given call cannot be stored, and why the state of having
    no environment is itself a serializable value.
    """

    def __init__(self, kind, activation=None, head=None, written="", name="",
                 module=None, home=None, elements=(), lazy=False, runner=None,
                 module_name=None):
        self.kind = kind                     # "func" | "module"
        self.activation = activation         # 函数：定义在哪个模块里
        self.head = head                     # 闭包写成什么：函数名或模块名那个节点
        self.written = written               # 写出来的样子：给人和闭包用
        self.name = name or written          # 定义名：宿主按这个名字找实现
        self.module = module                 # 格子声明在哪个模块：核对类型用
        self.home = home or module           # 这次调用写在哪个模块：写回材料用
        self.elements = list(elements)       # 各格，按书写顺序
        self.lazy = lazy
        self.runner = runner                 # 模块：谁来跑它
        self.module_name = module_name
        self.environ = None                  # 环境；给了就是执行
        self.given = {}                      # 格号 -> 值
        self.empty = False                   # 模块：那份空实参被显式写出来了

    @classmethod
    def func(cls, activation, head, written, owner_module, elements, lazy=False,
             name="", home=None):
        return cls("func", activation=activation, head=head, written=written,
                   name=name, module=owner_module, home=home, elements=elements,
                   lazy=lazy)

    @classmethod
    def module(cls, runner, module, module_name, head, home=None):
        return cls("module", runner=runner, head=head, written=module_name,
                   module=module, home=home, module_name=module_name)

    # ---- what this call is ----

    @property
    def slots(self):
        """Each slot's tag, or None for a position."""
        return [element.tag if isinstance(element, viba_ast.Tagged) else None
                for element in self.elements]

    def slot_tags(self):
        """The tags of the slots this pending call fills, in order — a module's
        slots come from its `__args__`, a function's from its chain."""
        if self.kind == "module":
            return [tag for tag, _written in (_module_arg_slots(self.module)[0] or [])]
        return self.slots

    def environ_slot(self):
        """Where the environment goes among a function's slots; a module has
        none — there it is not an argument but the execution itself."""
        if self.kind != "func":
            return None
        for index, element in enumerate(self.elements):
            if isinstance(element, viba_ast.Tagged) and element.tag == ENVIRON_TAG:
                return index
        return None

    def is_environment_slot(self, node):
        """Whether this written argument is the environment's slot.

        Only a marked function asks: its arguments are not computed, and the
        environment is never one of those — it is what an execution runs on.
        """
        index = self.environ_slot()
        if index is None:
            return False
        tag, _ = _addressed(node)
        if tag is not None:
            return tag == ENVIRON_TAG
        free = [one for one in range(len(self.elements)) if one not in self.given]
        return bool(free) and free[0] == index

    def ready(self) -> bool:
        if self.environ is None:
            return False
        if self.kind == "module":
            return self.empty or len(self.given) == len(self.elements)
        return len(self.given) == len(self.elements)

    def missing(self):
        """Why this call cannot run, or None when it can."""
        if self.kind == "module":
            return self._module_missing()
        left = [_slot_name(index, self.slots[index])
                for index in range(len(self.elements)) if index not in self.given]
        if not left:
            return None
        return (f"{self.written}: was given {len(self.given)} of its "
                f"{len(self.elements)} arguments: {', '.join(left)} missing")

    def _module_missing(self):
        slots, problem = _module_arg_slots(self.module)
        if problem is not None:
            return f"module {self.module_name!r}: {problem}"
        if not slots:
            return None
        left = [_slot_name(index, tag)
                for index, (tag, _written) in enumerate(slots) if index not in self.given]
        if not left:
            return None
        return (f"module {self.module_name!r} was given {len(self.given)} of its "
                f"{len(slots)} {ARGS_NAME}: {', '.join(left)} missing")

    # ---- taking arguments ----

    def give(self, tag, value):
        """Put one computed argument in the slot it goes to."""
        if self.kind == "module":
            return self._give_module(tag, value)
        return self._give_func(tag, value)

    def _give_func(self, tag, value):
        slots = self.slots
        if len(self.given) == len(slots):
            # 每一格都填过了：再来一个实参就是这个调用写多了（重复的 tag 后写的算，
            # 见下）。
            return VibaProgramErr(f"{self.written} takes no more arguments")
        if tag is not None:
            if tag not in slots:
                return VibaProgramErr(f"{self.written} takes no {tag} argument")
            index = slots.index(tag)
        else:
            free = [one for one in range(len(slots)) if one not in self.given]
            if not free:
                return VibaProgramErr(f"{self.written} takes no more arguments")
            index = free[0]
        element = self.elements[index]
        if self.lazy and self.environ is None and index != self.environ_slot():
            # 标记过的函数：环境要先给。给之前，一个实参是环境还是惰性实参分不出来。
            return VibaProgramErr(
                f"{self.written}: a marked function is given its environment "
                f"({ENVIRON_TAG}) before its arguments")
        if index == self.environ_slot():
            if not _is_environ_value(value):
                return VibaProgramErr(f"{self.written} was not given an {ENVIRON_TYPE}")
            self.given[index] = value
            self.environ = value.obj
            return Ok(self)
        if isinstance(value, _Getter):
            value.watch(element, self.module, self.written)
        else:
            problem = _fits_slot(value, element, self.module, self.written)
            if problem is not None:
                return VibaProgramErr(problem)
        self.given[index] = value
        return Ok(self)

    def _give_module(self, tag, value):
        slots, problem = _module_arg_slots(self.module)
        if problem is not None:
            return VibaProgramErr(f"module {self.module_name!r}: {problem}")
        # 环境不是 __args__ 的成员：它就是执行这一步，按值认，或者按 $env 这个 tag 认。
        if self.environ is None and (tag == ENVIRON_TAG or _is_environ_value(value)):
            if not _is_environ_value(value):
                return VibaProgramErr(f"module {self.module_name!r} needs an Environment")
            self.environ = value.obj
            return Ok(self)
        if not slots:
            if self.empty:
                return VibaProgramErr(
                    f"module {self.module_name!r} takes no more arguments: its "
                    f"{ARGS_NAME} is empty, written ()")
            if tag is not None or not _is_empty_product(value):
                return VibaProgramErr(
                    f"module {self.module_name!r} needs an Environment here "
                    f"(its {ARGS_NAME} is empty, written ())")
            self.empty = True                  # 那份空实参，是被写出来的
            return Ok(self)
        tags = [one for one, _written in slots]
        if tag is not None:
            if tag not in tags:
                return VibaProgramErr(
                    f"module {self.module_name!r} takes no {tag} argument: its "
                    f"{ARGS_NAME} are {_written_slots(slots)}")
            index = tags.index(tag)
            if index in self.given:
                return VibaProgramErr(
                    f"module {self.module_name!r} was given {tag} twice")
        else:
            free = [one for one in range(len(slots)) if one not in self.given]
            if not free:
                return VibaProgramErr(
                    f"module {self.module_name!r} takes no more arguments: its "
                    f"{ARGS_NAME} are all given")
            index = free[0]
        slot_tag, declared = slots[index]
        problem = _fits_slot(value,
                             viba_ast.Tagged(slot_tag, declared) if slot_tag else declared,
                             self.module, f"module {self.module_name!r}")
        if problem is not None:
            return VibaProgramErr(problem)
        self.given[index] = value
        return Ok(self)

    # ---- finishing ----

    def fire(self):
        """Run it: the environment is in and every argument is given."""
        if self.kind == "module":
            return self._run_module_call()
        return self._call_host()

    def materialize(self):
        """Without an environment this pending call is material — a closure.

        The function's name and the arguments already computed, written back as
        the chain they came from. Only material goes in: that is what makes the
        closure serializable, and it is the same thing every function and module
        answers with.
        """
        if self.lazy and self.given:
            return VibaProgramErr(
                f"{self.written}: a marked function is not stored; it is executed "
                f"in one chain")
        node = self.head
        tags = self.slot_tags()
        for index in sorted(self.given):
            value = self.given[index]
            if not isinstance(value, _Material):
                return VibaProgramErr(
                    f"{self.written}: a closure holds material only; the "
                    f"{_slot_name(index, tags[index])} argument is not")
            tag = tags[index]
            piece = value.node.data
            node = viba_ast.Partial(node, viba_ast.Tagged(tag, piece) if tag else piece)
        return Ok(_Material(VibaNode(reflect_access, self.descriptor(node), node)))

    def descriptor(self, node):
        """What the design calls this piece.

        For a closure that is the written call itself — the head name — and not
        the type it would have once executed: the node is the call as it stands,
        and the spelling of a value comes from its own piece.
        """
        if isinstance(node, viba_ast.Partial):
            return descriptor_of(AstNodeType(self.head, self.home))
        return descriptor_of(AstNodeType(node, self.home))

    def call_material(self):
        """The material this call was given, as it was written.

        A host value — the environment above all — is no material and does not
        travel: the side that answers makes its own. One material argument is
        that argument itself (no tag is needed to tell it from the others),
        which is the `$call` a Prepare of such a call fixes; several make a
        product, keeping the tags as written. None when nothing material was
        given.
        """
        tags = self.slot_tags()
        material_given = [(tags[index], value.node.data)
                          for index, value in sorted(self.given.items())
                          if isinstance(value, _Material)]
        if not material_given:
            return None
        if len(material_given) == 1:
            return material(material_given[0][1])
        written = [viba_ast.Tagged(tag, piece) if tag else piece
                   for tag, piece in material_given]
        return material(viba_ast.ProductChain(written))

    def _call_host(self):
        """Every slot is filled: the environment's compute side implements it."""
        environ = self.environ
        compute = getattr(environ, "compute", None)
        if compute is None:
            return VibaProgramErr(
                f"{self.written}: the environment carries no compute side")
        module_path = _storage_path(environ)
        step = Step(module_path, self.name)
        try:
            host = compute.get_func(module_path, self.name)
        except NotMyDutyException as deferred:   # the host refuses this call
            return _refused(deferred, step, self.call_material())
        except Exception as exc:            # the host is the host's business
            return UnderlyingVibaOpFailed(
                f"get_func({module_path!r}, {self.name!r}) raised {exc!r}",
                step, REASON_GET_FUNC_RAISED)
        if host is None:
            return _no_implementation(step, self.call_material())
        handed = [_getter(self.given[index]) if self.lazy
                  else _argument_value(self.given[index])
                  for index in range(len(self.elements))]
        try:
            answer = host(*handed)
        except NotMyDutyException as deferred:   # a viba call inside deferred
            return deferred
        except UnderlyingVibaOpFailed as failure:                # ... or failed inside
            return failure
        except _Raised as raised:                # ... or stopped inside a getter
            return raised.result
        except Exception as exc:
            return UnderlyingVibaOpFailed(f"{self.name} raised {exc!r}", step,
                                          REASON_RAISED)
        return _answer(self.name, answer, step)

    def _run_module_call(self):
        """Every `__args__` member is in: hand them to the module as one product."""
        slots, _problem = _module_arg_slots(self.module)
        nodes = []
        for index, (tag, _written) in enumerate(slots or []):
            value = self.given.get(index)
            if value is None:
                return VibaProgramErr(
                    f"module {self.module_name!r} is not ready to run")
            if not isinstance(value, _Material):
                return VibaProgramErr(
                    f"module {self.module_name!r}: the {_slot_name(index, tag)} argument "
                    f"is not material, so it cannot be part of {ARGS_NAME}")
            nodes.append(viba_ast.Tagged(tag, value.node.data) if tag else value.node.data)
        if not nodes:
            args = _Material(material(None))
        elif len(nodes) == 1:
            args = _Material(VibaNode(reflect_access, self.descriptor(nodes[0]), nodes[0]))
        else:
            chain = viba_ast.ProductChain(nodes)
            args = _Material(VibaNode(reflect_access, self.descriptor(chain), chain))
        answer = _run_module(self.runner, self.module, self.environ, self.module_name,
                             None, args)
        if _stopped(answer):
            return answer
        # `interpret` hands the node out; inside a run a module's answer is a
        # value like any other, so it goes back into the value model.
        node = answer.ok_value
        return Ok(_Material(node) if isinstance(node, VibaNode) else _Host(node))


class _HostFunction:
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
        except _Raised as raised:
            return raised.result
        except Exception as exc:
            return UnderlyingVibaOpFailed(f"environ.{self.name} raised {exc!r}", step,
                                          REASON_RAISED)
        return _answer(f"environ.{self.name}", answer, step)


def _slot_name(index: int, tag) -> str:
    return tag if tag else f"#{index + 1}"


def _written_slots(slots) -> str:
    return ", ".join(_slot_name(index, tag) for index, (tag, _written) in enumerate(slots))


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
