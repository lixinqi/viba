"""viba.interpret — run a viba module.

A module is a file, and a file is also a function: `__def__` is that function
(the environment among its parameters), and its output is `__ret__`. Type
inference reads the same file as a type (viba.is_sub_type); computation runs it
(here). A file that wants to be runnable defines `__ret__`; a file that does not
is design only. Inside the module, `args = __get_args__ << __def__` is what the
call handed over — `args.env` above all.

    from viba.interpret import interpret

    interpret("add_demo.viba", environ)  # -> Ok(VibaNode) | VibaProgramErr(str) | a stop
    interpret("main.viba", environ, get_file=files.get)   # sources from anywhere

A function the compute side does not implement is not a failure: the run stops
and answers `NotMyDutyException` — `$not_my_duty_exception Duty` — the deferral
that says this host is not the one to finish it, and carries the step, the
viba data it was given and why (`roadmap.md`). A step whose implementation broke
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
        the three together, and `sub_env(name)` / `tmp_env()` for a child —
        which keeps the parent's compute and its module search path.

A host function is called with the arguments already evaluated, in the order
they are written: a piece of viba data arrives as a `viba.reflect.VibaNode`,
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

import inspect
import os
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from viba import serialize, viba_ast
from viba.partial import (file_environment_result_problem, parameters_of,
                          product_elements, names_the_environment)
from viba.reflect import VibaNode, access as reflect_access
from viba.type import (CustomModuleType, REASON_GET_FUNC_RAISED, REASON_NO_IMPLEMENTATION, REASON_NO_LEAF,
                       REASON_RAISED, REASON_REFUSED, AstNodeType, VibaProgramErr, UnderlyingVibaOpFailed,
                       InterpretResult, ModuleType, NotMyDutyException, Ok, Step,
                       BUILTIN_MODULE, NilType, NeverType, custom_module)
from viba.viba_type_descriptor import (descriptor_of, descriptor_of_tagged,
                                        descriptor_of_values)

# A scalar a host answers belongs to no file: its leaf gets an empty module.
# Parsed once, not once per answer.
_NO_MODULE = custom_module("")

TMP_PREFIX = "tmp_"
RET_NAME = "__ret__"
DEF_NAME = "__def__"
GET_ARGS_NAME = "__get_args__"
ENVIRON_TAG = "$env"
ENVIRON_TYPE = "Environment"
ENV_TYPE = "Env"

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

    __slots__ = ("storage", "compute", "viba_path", "sub_env", "tmp_env")

    def __init__(self, storage: EnvironmentStorage, compute: EnvironmentCompute,
                 viba_path=None):
        self.storage = storage
        self.compute = compute
        # Where modules are looked up, like PYTHONPATH: a string of directories,
        # or one path. A module runs under the environment it was handed, so its
        # own imports are looked up where that environment says — which is how a
        # sub-environment keeps the parent's search path along with its compute.
        self.viba_path = viba_path
        # The members a design hangs off the environment are plain functions of
        # the environment itself: nothing is bound to the one they were read
        # from, so `args.env.sub_env << args.env << "child"` and
        # `$sub_env << args.env << "child"` are the same call (viba-interpreter.md).
        self.sub_env = sub_env
        self.tmp_env = tmp_env


def sub_env(environ: "Environment", name) -> "Environment":
    """A child environment: its own storage, the parent's compute.

    `name` is what the viba side wrote: a viba data node lands as the leaf it
    carries, so `args.env.sub_env << args.env << "add_demo"` names the module. The
    same name is the same child, handed back again.
    """
    if isinstance(name, VibaNode):
        name = name.value
    return Environment(environ.storage.sub(str(name)), environ.compute, environ.viba_path)


def tmp_env(environ: "Environment") -> "Environment":
    """A child environment under a name of its own, fresh every time: no name to
    pick, and no two calls share a storage path."""
    return Environment(environ.storage.tmp(), environ.compute, environ.viba_path)


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
    comes back as the viba data it was. What cannot be read raises — a host
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
    written = serialize.serialize(SNAPSHOT_NAME, viba_data(value))
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


def viba_data(value) -> VibaNode:
    """A host value as viba data.

    A `VibaNode` as it is; an AST piece a host built (a product of tags and
    literals, say) gets the design written on it; a scalar becomes its leaf.
    """
    if isinstance(value, VibaNode):
        return value
    if isinstance(value, viba_ast.AST):
        return VibaNode(reflect_access,
                        descriptor_of(AstNodeType(value, _NO_MODULE)), value)
    if value is not None and not isinstance(value, (bool, int, float, str)):
        raise RuntimeError(f"cannot take {type(value).__name__} as viba data: "
                           f"only a VibaNode, an AST piece, a scalar, or None")
    node = viba_ast.Nil() if value is None else viba_ast.Constant(value)
    return VibaNode(reflect_access, descriptor_of(AstNodeType(node, _NO_MODULE)), node)


def _is_never(value) -> bool:
    """Whether `value` is the additive unit `never`."""
    return isinstance(value, _VibaData) and isinstance(value.node.data, viba_ast.Never)


def _is_nil(value) -> bool:
    """Whether `value` is the multiplicative unit `nil`."""
    return isinstance(value, _VibaData) and isinstance(value.node.data, viba_ast.Nil)


def _one_line(node) -> str:
    """A piece as one line: error messages read better without the layout."""
    return " ".join(viba_ast.unparse_type(node).split())


def _slot_type(element):
    """The declared type a written argument slot asks for."""
    return element.type if isinstance(element, viba_ast.Tagged) else element


def _value_as_type(value):
    """The type this value already is, or None when it has none written here.

    Viba data is the design it was made of and a viba function is its chain, so
    the judgment can read them; the environment is `Environment`. A half-given
    call is the function type it still owes. A host value with no viba type — a
    host function, a host object — has none, and nothing is judged for it.
    """
    if isinstance(value, _VibaData):
        return value.node.data
    if isinstance(value, _Pending):
        return _pending_type(value)
    if isinstance(value, _Host) and isinstance(value.obj, Environment):
        return viba_ast.TypeRef(ENVIRON_TYPE)
    return None


def _pending_type(value):
    """The function type a half-given call still is, or None.

    Its arguments are the design's, so the environment does not show up here: a
    call that still owes the environment is the same function as one that has it
    — giving it is what runs the call, not an argument of it.
    """
    if value.head is None:
        return None
    keep = [index for index, element in enumerate(value.elements)
            if not _is_the_environment_element(element)]
    rest = [value.elements[index] for index in keep if index not in value.given]
    if not rest:
        return value.head
    return viba_ast.ExponentChain([value.head] + rest)


def _is_the_environment_element(element) -> bool:
    """Whether this written slot is the environment — the call's rule, not an
    argument the design takes."""
    if isinstance(element, viba_ast.Tagged):
        return element.tag == ENVIRON_TAG
    return False


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
    # 这个名字写在哪个模块，就在哪个模块里读 —— 跟着值走，不跟着收它的那一方走。
    judged = is_sub_type(AstNodeType(given, _writing_module(value) or module),
                         AstNodeType(written, module))
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


def _never_viba_data():
    """The `never` value as viba data."""
    node = viba_ast.Never()
    return _VibaData(VibaNode(reflect_access,
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


def _written_in_of(node):
    """The module a node's written names resolve in, read off its descriptor.

    A name means what it means in the module that wrote it (`F` is one file's
    import and nothing in another), so every piece carries that module. It sits
    on the descriptor's resolvable type: a descriptor is built *from* one, so it
    holds the module rather than showing it as an attribute of its own.
    """
    descriptor = getattr(node, "descriptor", None)
    written_in = getattr(descriptor, "container_module", None)
    if written_in is None:
        resolvable = getattr(descriptor, "resolvable_type", None)
        written_in = getattr(resolvable, "container_module", None)
    return written_in


class _VibaData:
    """A piece of viba data with its design: a `VibaNode`.

    It also carries **the module that wrote it** and, for a call stored
    as a value, **the values that call was given**. Both are what lets a piece
    travel: a name resolves in the module it was written in, and a call's
    arguments are the values it received rather than their text re-read
    somewhere else.
    """

    __slots__ = ("node", "written_in", "given")

    def __init__(self, node: VibaNode, written_in=None, given=None):
        self.node = node
        self.written_in = written_in if written_in is not None else _written_in_of(node)
        self.given = given                 # [(tag, _VibaData | None)], a kept call's arguments


class _Host:
    """Anything the host handed over: the environment, a host value."""

    __slots__ = ("obj",)

    def __init__(self, obj):
        self.obj = obj


class _Member:
    """`$tag` at the head of a chain: the member `$tag` of the value given first.

    The chain gives that value first (`$sub_env << args.env << "c"`), and giving
    it is what takes the member: the first argument is never an argument of the
    member itself. A tag is not a value, so this only ever stands at the head of
    a chain and is gone as soon as the chain has run (viba-interpreter.md).
    """

    __slots__ = ("tag",)

    def __init__(self, tag):
        self.tag = tag


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
    """One written argument of a deferred call, not computed where it is written.

    A slot written as a function type — `$get_v (T <- $env Environment)` — is
    handed one of these instead of a value: the argument is written down, and the
    host runs it when and where it wants it. A host that never calls it never
    computes it at all.

    **At most once**: the answer, value or stop, is worked out on the first call
    and handed back on every later one, so asking twice cannot make a side effect
    happen twice.
    """

    __slots__ = ("activation", "node", "scope", "answer", "slot", "module", "owner",
                 "file", "written_in", "name", "origin")

    def __init__(self, activation, node, scope=()):
        self.activation = activation
        self.node = node
        self.scope = scope                  # the block the argument was written in
        self.answer = None                  # the Result, once it is worked out
        self.slot = None                    # what the call asked for, once it is taken
        self.module = None
        self.owner = ""
        self.file = activation.file         # the module that wrote the argument
        self.written_in = activation.module       # where the writing module's names resolve
        self.origin = activation            # the activation that wrote it: its args are there
        self.name = activation.name

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
    None: `nil` is viba data)."""
    for frame in reversed(scope):
        if name in frame:
            return frame[name]
    return None


def _addressed(node):
    """(tag, inner) of a written argument: the tag it is addressed by, if any."""
    if isinstance(node, viba_ast.Tagged):
        return node.tag, node.type
    return None, node


def _argument_value(value):
    """What a host function is handed.

    Viba data arrives as its node, a viba function as a Python callable (the
    host calls it like any other function: the values it is called with are
    nodes or host objects, and the answer is a node or a host object), and
    anything else as itself — the environment among them.
    """
    if isinstance(value, _VibaData):
        return value.node
    if isinstance(value, _HostFunction):
        return value.as_callable()
    if isinstance(value, _Host):
        return value.obj
    return value


def _given_value(value):
    """A value a host handed back into a call: a node, a scalar, or an object."""
    if isinstance(value, _HostArgument):
        return _VibaData(VibaNode(reflect_access,
                                  descriptor_of(AstNodeType(value.node, value.module)),
                                  value.node))
    if isinstance(value, _VibaData):
        return value
    answer = _answer("a host argument", value)
    if isinstance(answer, VibaProgramErr):
        raise RuntimeError(answer.err_msg)
    return answer.ok_value


def _as_given_value(value):
    """A value a step answered: a `Result` that was carried through as a value,
    or the value itself. Nothing else can reach a host."""
    if isinstance(value, (Ok, VibaProgramErr)):
        return value.ok_value if isinstance(value, Ok) else value
    return value


def _stopped(result) -> bool:
    """True when a step carried no value on: a `VibaProgramErr`, or the deferral.

    `VibaProgramErr` says this run failed; `NotMyDutyException` says this run is asking for
    an implementation it does not have. Both stop the chain, and both travel
    back to the caller as they are.
    """
    return not isinstance(result, Ok)


def _no_implementation(step: Step, call) -> NotMyDutyException:
    """The deferral a host answers with: no implementation for that call.

    The step, the viba data it was given and why are all in it, so the side that
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
    and the duty carries the step, the viba data it was given and why), and
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
        # 一次调用的身份是它的 storage 路径：路径正在跑就是环；同一条路径答过了就把那份
        # 答案回放出来。另外一条：同一个模块 + 同一份实参还在跑，是"没有进展"，也算环 ——
        # 同样输入的一次计算正在里面进行，它停不下来。
        self.running: list = []        # (path, module name, arguments) 正在跑的调用
        self.done: dict = {}           # 路径 -> (module name, answer) 答过的调用
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
            problem = file_environment_result_problem(tree)
            if problem is not None:
                return VibaProgramErr(f"{path}: {problem}")
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
               name: str, file: Optional[str], args=None, members=None) -> InterpretResult:
    """The module as a function: the environment in, `__ret__` out.

    A module that declares `__def__` is called with its parameters as well; `args`
    is the product they were given as — the environment among its members, which
    is what the module reads as `args.env` (`args = __get_args__ << __def__`).

    The module the host runs has no caller to write its arguments, so the
    environment it is handed is the only member of that product.

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
    params, problem = _module_params(module)
    if problem is not None:
        return VibaProgramErr(f"module {name!r}: {problem}")
    if members is None:
        members = [(ENVIRON_TAG, _Host(environ))]
    if args is None:
        args = _VibaData(viba_data(None))
    path = _storage_path(environ)
    signature = _call_signature(members)
    for seen, running_name, running_signature in runner.running:
        if seen == path:
            return VibaProgramErr(
                f"the storage path {path!r} is already running a call, so {name!r} "
                f"cannot run there too: give each module call a sub-environment of "
                f"its own (args.env.sub_env << args.env << ...)")
        if running_name == name and running_signature == signature:
            return VibaProgramErr(
                f"module {name!r} is already running with the same arguments: "
                f"a module call cycle")
    if path in runner.done:
        answered_as, answered = runner.done[path]
        if answered_as == name:
            # 同一条地址、同一个模块：这是同一个子计算被问了第二次，把它答过的那份交回去。
            return Ok(answered)
        return VibaProgramErr(f"module {name!r} was handed the storage path {path!r}, which "
                   f"another module call already used: give each module call a "
                   f"sub-environment of its own (args.env.sub_env << args.env << ...)")
    runner.running.append((path, name, signature))
    try:
        value = _Activation(runner, module, environ, name, file, args,
                            members=members).evaluate(ret.body)
    finally:
        runner.running.pop()
    if _stopped(value):
        return value
    if not isinstance(value.ok_value, (_VibaData, _Host)):
        return VibaProgramErr(f"{name}.{RET_NAME} is a function still waiting for arguments")
    answer = _argument_value(value.ok_value)
    runner.done[path] = (name, answer)
    return Ok(answer)


def _call_signature(members) -> tuple:
    """What a call's arguments are, as one comparable thing.

    The path is a call's identity, so the same path twice is the same call; this
    is what tells a call that made no progress from one that did — a module
    running with the same arguments again has the same inputs it already has.
    The environment is not one of those inputs: it is the address the call runs
    at, which the path guard already answers, so two calls that differ only in
    the environment they were handed are the same call with the same arguments.
    """
    if not members:
        return ()
    return tuple(_signature_of(value) for _tag, value in members
                 if not _is_environ_value(value))


def _signature_of(value) -> str:
    """One argument, written the same way every time."""
    if isinstance(value, _VibaData):
        return viba_ast.unparse_type(value.node.data)
    return type(value).__name__


def _storage_path(environ: Environment) -> str:
    """The path the host is handed for this environment: a module's identity."""
    storage = getattr(environ, "storage", None)
    return getattr(storage, "cur_storage_path", "") if storage else ""


def _module_params(module: ModuleType):
    """(params, problem) for a module's `__def__`, or (None, None) for none.

    `__def__` is the module read as a function: the result first, then the
    call's parameters in written order — `__def__ = int <- $env Env <- $a int`
    is a module answering an int, asking for an environment and an `$a`. Each
    parameter comes back as (tag, written type, is the environment, argument
    slot); the slot counts arguments only, so the environment has none: giving
    it is what runs the call, it is not an argument.

    A module that runs declares exactly one environment parameter, tagged
    `$env`: a module that takes the environment is a module that can be run. Its
    answer is never the environment itself — a file that writes such a function is
    refused where it is read (`file_environment_result_problem`).
    """
    definition = _definition(module, DEF_NAME)
    if definition is None:
        return None, None
    body = definition.body
    if not isinstance(body, (viba_ast.Exponent, viba_ast.ExponentChain)):
        return None, (f"{DEF_NAME} is not a function type: {viba_ast.unparse_type(body)}")
    params = parameters_of(body)
    envs = [one for one in params if one[2]]
    if _definition(module, RET_NAME) is not None and not envs:
        return None, (f"{DEF_NAME} has no {ENVIRON_TAG} {ENV_TYPE} parameter: "
                      f"every module that runs depends on the environment")
    if len(envs) > 1:
        return None, (f"{DEF_NAME} has {len(envs)} {ENVIRON_TAG} {ENV_TYPE} parameters: "
                      f"a module that runs takes exactly one")
    for _tag, written, _is_env, _slot in envs:
        if not names_the_environment(written):
            return None, (f"the {ENVIRON_TAG} parameter of {DEF_NAME} must be "
                          f"{ENV_TYPE}, not {viba_ast.unparse_type(written)}")
    return params, None


def _module_arg_slots(module: ModuleType):
    """(slots, problem) for a module's arguments: its `__def__` less the
    environment parameter, which is the call's rule rather than an argument."""
    params, problem = _module_params(module)
    if problem is not None:
        return None, problem
    if params is None:
        if _definition(module, RET_NAME) is None:
            return [], None                  # design only: it never runs
        return None, (f"module has no {DEF_NAME}: a module that runs declares its "
                      f"parameters, its {ENVIRON_TAG} {ENV_TYPE} among them")
    return ([(tag, written) for tag, written, is_env, _slot in params if not is_env],
            None)


def _viba_data_factors(node: VibaNode):
    """The factors of a product viba data, in written order."""
    data = node.data
    if isinstance(data, (viba_ast.Product, viba_ast.ProductChain)):
        return product_elements(data)
    return [data]


def _definition(module: ModuleType, name: str):
    """The definition this name stands for: the last one written under it."""
    found = None
    for node in module.module.body:
        if getattr(node, "name", None) == name:
            found = node
    return found


class _Activation:
    """One module running with one environment."""

    def __init__(self, runner: _Runner, module: ModuleType, environ: Environment,
                 name: str, file: Optional[str], args=None, members=None):
        self.runner = runner
        self.module = module
        self.environ = environ
        self.name = name
        self.file = file
        self.args = args                     # 这次调用收到的实参积（`__get_args__` 答的那份）
        self.path = _storage_path(environ)   # 这次调用跑在哪个地址上
        self.signature = _call_signature(members)   # 这次调用收到的是什么
        # 这次调用收到的每个成员，按它给进来时的样子留着（各自记着写它的那个模块）：`args.a`
        # 是"这次调用收到的那份值"，不是把值压成节点之后重新读出来的东西。
        #
        # 没有调用方写实参的那一次激活（读另一个模块的定义、宿主递回来的那段文本），收到的
        # 只有环境：那份积里就只有 `env` 这一个成员，和一次模块调用收到的一样。
        self.members = members if members is not None else [(ENVIRON_TAG, _Host(environ))]
        if args is None:
            args = _VibaData(viba_data(None))
        self.args = args
        self.defined: dict = {}

    # ---- expressions ----

    def evaluate(self, node, scope=()):
        """Result: the value this piece writes — or why the chain stopped: an
        `VibaProgramErr`, or the deferral of a step nobody here implements.

        A piece of data written where a value goes is viba data as it stands:
        a literal or a unit, a tuple, and a product of tags and literals —
        `$victim ($x 0 * $y 0) * $at "12:30"` is a witness, the same spelling
        its type would have. Its members are data, not calls, so nothing in
        them is evaluated. A sum is not: which branch would it be.

        `scope` is the frames written around this piece; nothing binds a name
        inside an expression, so a definition's own body runs with the empty
        scope.
        """
        if isinstance(node, (viba_ast.Constant, viba_ast.Nil, viba_ast.Never,
                             viba_ast.Any, viba_ast.Tuple, viba_ast.Tagged)):
            return Ok(_VibaData(VibaNode(reflect_access, self._descriptor(node), node)))
        if isinstance(node, (viba_ast.Product, viba_ast.ProductChain)):
            return self._product(node, scope)
        if isinstance(node, (viba_ast.Sum, viba_ast.SumChain)):
            return self._sum(node, scope)
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
                return Ok(_never_viba_data())
            if _is_nil(answered):
                continue
            kept.append(answered)
        if not kept:
            return Ok(_VibaData(viba_data(None)))
        if len(kept) == 1:
            return Ok(kept[0])
        chain = viba_ast.ProductChain([factor.node.data for factor in kept])
        return Ok(_VibaData(VibaNode(reflect_access, self._descriptor(chain), chain)))

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
            return Ok(_never_viba_data())
        if len(kept) == 1:
            return Ok(kept[0])
        elements = [branch.node.data for branch in kept]
        chain = viba_ast.SumChain(elements)
        return Ok(_VibaData(VibaNode(reflect_access, self._descriptor(chain), chain)))

    def _imports(self) -> dict:
        """The file's import table: what each import binds, and the module it
        names (`import a.b as c` binds c, `import a.b` binds a.b)."""
        return {stmt.alias or stmt.module: stmt.module
                for stmt in self.module.module.body
                if isinstance(stmt, viba_ast.Import)}

    def _resolve(self, name: str, scope=()):
        bound = _in_scope(scope, name)
        if bound is not None:
            return Ok(bound)
        if name == GET_ARGS_NAME:
            # 这次调用收到的那份实参：给一个环境就答它。推导层读同一个名字，读出来的
            # 是那次调用的参数的积类型（viba-interpreter.md）。
            return Ok(_GetArgs(self))
        if name == DEF_NAME:
            # Read as a type `__def__` is the module's own function chain; read as
            # a value it is that same written chain, which is what `__get_args__`
            # is given.
            definition = _definition(self.module, DEF_NAME)
            if definition is not None:
                node = definition.body
                return Ok(_VibaData(VibaNode(
                    reflect_access,
                    descriptor_of(AstNodeType(node, self.module)), node)))
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
            return Ok(_VibaData(VibaNode(
                reflect_access, descriptor_of(AstNodeType(node, self.module)), node)))
        member = self._tagged_member(name, scope)
        if member is not None:
            return member
        builtin = self._builtin_member(name)
        if builtin is not None:
            return builtin
        member = self._environment_member(name, scope)
        if member is not None:
            return member
        return VibaProgramErr(f"no definition named {name!r} in module {self.name!r}")

    def _tagged_member(self, name: str, scope=()):
        """`args.a`: the `$a` member of a product this module names.

        A dotted name is a member of an imported module (`demo.print`), which is
        settled before this; when the head is not an import, it is the tag of a
        member of the product the head stands for — the arguments of the call
        among them (`args.a`). None when the head is no such value, so the
        caller reports the name the way it always did.
        """
        head, dot, tag = name.rpartition(".")
        if not dot or not head:
            return None
        value = self._resolve(head, scope)
        if _stopped(value) or not isinstance(value.ok_value, _VibaData):
            return None
        wanted = "$" + tag
        if value.ok_value is self.args:
            # 这次调用的实参积：成员就是当初给进来的那份值，原样交回去 —— 它自己
            # 记得名字写在哪个模块，重新包一遍会把那个记住丢掉。
            for member_tag, member in self.members:
                if member_tag == wanted:
                    return Ok(member)
        for factor in _viba_data_factors(value.ok_value.node):
            if isinstance(factor, viba_ast.Tagged) and factor.tag == wanted:
                inner = factor.type          # the member's value, not its address
                return Ok(_VibaData(VibaNode(
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
        mine = (self.name, name, self.signature)
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
        mine = (self.name, name, self.signature)
        path = self.runner.computing[self.runner.computing.index(mine):] + [mine]
        # 同一个模块、同一份实参还在算，就是没有进展：一份文件的定义不许成环。两个
        # **不同**的模块互相读才是"绕回起点"；同名不同实参的两次调用是两次调用，不是环
        # （固定点就是这样展开的）。
        if len({module for module, _defined, _signature in path}) == 1:
            written = " -> ".join(defined for _module, defined, _signature in path)
            return (f"{written}: one file's definitions may not go round — "
                    f"recursion takes two files")
        written = " -> ".join(f"{module}.{defined.split('.')[-1]}"
                              for module, defined, _signature in path)
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
                                          written_in=self.module, written=written or rest)

    def _value_of_definition(self, name, definition, owner_module, written_in=None,
                             written=None):
        """A name read as a value: a function is the closure it stands for, which
        is written as the name itself. Everything else is its body, computed.

        `name` is the local definition (the name this module knows it by, which
        is also how the cycle guard counts it); `written` is how the caller wrote
        it, which is what a closure made of it says.
        """
        written_in = written_in or self.module
        body = definition.body
        if isinstance(body, (viba_ast.Exponent, viba_ast.ExponentChain)):
            node = viba_ast.TypeRef(written or name)
            return Ok(_VibaData(VibaNode(
                reflect_access, descriptor_of(AstNodeType(node, written_in)), node)))
        return self._defined(name, definition)

    def _take_member(self, member, value):
        """The member `$tag` of the value the chain gave first.

        An environment hands over the function it hangs off itself — the same
        thing `args.env.sub_env` reads, only reached from the value the chain was
        given. A product hands over the piece its tag addresses. Anything else
        keeps no members here.
        """
        tag = member.tag
        name = tag[1:]
        if _is_environ_value(value):
            attributed = getattr(value.obj, name, None)
            if not callable(attributed):
                return VibaProgramErr(f"the environment has no {name!r}")
            # The value the member is taken from is also what the member is
            # given first: `$sub_env << args.env << "child"` is
            # `args.env.sub_env << args.env << "child"`.
            return Ok(_HostFunction(name, attributed,
                                    slots=_required_arguments(attributed),
                                    given=[value.obj],
                                    module_path=_storage_path(value.obj)))
        if isinstance(value, _VibaData):
            for factor in _viba_data_factors(value.node):
                if isinstance(factor, viba_ast.Tagged) and factor.tag == tag:
                    inner = factor.type       # the member's value, not its address
                    return Ok(_VibaData(VibaNode(
                        reflect_access,
                        descriptor_of(AstNodeType(inner, self.module)), inner)))
            return VibaProgramErr(f"no member tagged {tag!r} to take from it")
        return VibaProgramErr(f"{type(value).__name__} has no member tagged {tag!r}")

    def _environment_member(self, name: str, scope=()):
        """`args.env.sub_env`: a member of the environment this module names.

        The environment is a host value, and what a design hangs off it are
        plain functions of the environment itself, so the value the member is
        read from is also what it is given first:
        `args.env.sub_env << args.env << "child"` is the same call as
        `$sub_env << args.env << "child"`. None when the head is no environment,
        so the caller reports the name the way it always did.
        """
        head, dot, rest = name.rpartition(".")
        if not dot or not head:
            return None
        value = self._resolve(head, scope)
        if _stopped(value) or not _is_environ_value(value.ok_value):
            return None
        environ = value.ok_value.obj
        member = getattr(environ, rest, None)
        if not callable(member):
            return VibaProgramErr(f"the environment has no {rest!r}")
        return Ok(_HostFunction(rest, member, slots=_required_arguments(member),
                                module_path=_storage_path(environ)))

    # ---- calls ----

    def _apply_chain(self, node, scope=()):
        """Give the written arguments to what stands at the head of the chain.

        The chain nests left: `((f << a) << b) << c` is read by walking the
        spine, which meets c first. It is read apart into (head, arguments in
        written order), the head is resolved, and then each argument is given in
        that order — the order the host sees them in, and the order side effects
        happen in. A closure stored as viba data is that same written chain, so
        applying more arguments to one is reading it apart again and carrying on.
        """
        target, written = self._target_and_arguments(node, scope)
        if written is None:
            return target
        return self._give_all(target.ok_value, written, scope)

    def _give_all(self, current, written, scope, finish=True):
        """Give a list of written arguments to what the chain calls, in order.

        An entry is either a written piece — computed here, in the module that
        wrote it — or an argument a call already kept (`_Given`), which is the
        value it was given, and the module it was written in.

        `finish` says what to do with a call the chain leaves incomplete: the end
        of a written chain finishes it (it runs when the environment is in, and
        is stored as a closure when it is not), while a host that hands over an
        environment of its own calls this with `finish=False` and gives that
        environment itself.
        """
        for index, argument in enumerate(written):
            if _is_a_written_call(current):
                # A value that is a call written down — a member read out of the
                # design, a closure kept as a value — takes the rest of the chain
                # the way it would at the head: read the chain it is and keep
                # giving. Nothing is computed twice: the arguments still to be
                # given have not been touched yet.
                target = self._read_stored_call(current, scope, finish=finish)
                if target is not None:
                    if _stopped(target):
                        return target
                    return self._give_all(target.ok_value, written[index:], scope,
                                          finish=finish)
                node = current.node.data
                for later in written[index:]:
                    node = viba_ast.Partial(node, later)
                written_in = _writing_module(current)
                other = self._in_module(written_in) if written_in is not None else None
                if other is None or written_in is self.module:
                    return self._apply_chain(node, scope)
                # 存下来的那一段写在 writing_module 那个模块里：只把**它**读成一次调用，
                # 后面的实参写在这里，仍在这里算。
                stored = self._read_stored_call(current, scope, whole=True, finish=finish)
                if stored is None:
                    return self._apply_chain(node, scope)
                if _stopped(stored):
                    return stored
                return stored
            if isinstance(argument, _Given):
                value = Ok(argument)
            else:
                value = (self._lazy_argument(argument, scope)
                         if self._argument_is_deferred(current, argument)
                         else self._argument(argument, scope))
            if _stopped(value):
                return value
            if value.ok_value is None:
                continue                        # documentation is no argument
            if isinstance(current, _Member):
                taken = self._take_member(current, value.ok_value.value)
                if _stopped(taken):
                    return taken
                current = taken.ok_value
                if _is_a_written_call(current):
                    # The value the member came from is given to the member too,
                    # and the rest of the chain goes on from there.
                    node = current.node.data
                    for later in written[index:]:
                        node = viba_ast.Partial(node, later)
                    return self._apply_chain(node, scope)
                if isinstance(current, _HostFunction) and current.filled():
                    run = current.run()
                    if _stopped(run):
                        return run
                    current = run.ok_value
                continue
            if isinstance(current, _Pending):
                current = current.give(value.ok_value.tag, value.ok_value.value)
            else:
                current = _host_give(current, value.ok_value)
            if _stopped(current):
                return current
            current = current.ok_value
        if isinstance(current, _Pending):
            return self._finish(current) if finish else Ok(current)
        if isinstance(current, _HostFunction):
            if current.filled():
                return current.run()
            return VibaProgramErr(
                f"the environment's {current.name} was given {len(current.given)} "
                f"of its {current.slots} arguments")
        if isinstance(current, _GetArgs):
            return VibaProgramErr(
                f"{GET_ARGS_NAME} asks for the module's {DEF_NAME}, and none was given")
        return Ok(current)

    def _in_module(self, module):
        """An activation for another module — the one a piece's text was written
        in. None when this run never bound that module to a name."""
        for name, known in self.runner.by_name.items():
            if known is module:
                return _Activation(self.runner, module, self.environ, name,
                                   self.runner.path_of.get(name))
        return None

    def _read_stored_call(self, value, scope, whole=False, finish=True):
        """Read a call kept as a value: (what it calls, its kept arguments).

        The head resolves in the module the call was written in — that is what
        its name means — while its arguments are the values it was given, each
        carrying the module it was written in. None when there is nothing stored to read
        (then the caller reads the chain the old way). `whole` additionally asks
        for the head itself to be resolved here, not only the arguments.
        """
        if value.given is None:
            return None
        head, passed = _stored_arguments(value)
        written_in = _writing_module(value)
        if whole or (written_in is not None and written_in is not self.module):
            other = self._in_module(written_in) if written_in is not None else None
            if other is not None and isinstance(head, viba_ast.TypeRef):
                target = other._call_target(head, scope)
                if target is None:
                    return None
                if _stopped(target):
                    return target
                return self._give_all(target.ok_value, passed, scope, finish=finish)
        return None

    def _apply_without_environ(self, node, scope, environ):
        """Run the call `node` writes with the environment the host handed over.

        A function-typed slot names the argument as the call it stands for
        (`tick`, or `poison << $x 1`). The environment is what runs a call and it
        is never an argument of the written chain, so the chain is given first
        and the host's environment lands in the `$env` slot it still owes: a
        chain that wrote one of its own (`branch.echo_or_never << $env ...`) is
        left alone, and one that did not is run under the environment the host
        chose. `finish=False` is what keeps the call from being stored as a
        closure before that environment is in.
        """
        target, written = self._target_and_arguments(node, scope)
        if written is None:
            return target
        given = self._give_all(target.ok_value, written, scope, finish=False)
        if _stopped(given):
            return given
        current = given.ok_value
        if isinstance(current, _Pending):
            if current.environ is None:
                handed = environ if _is_environ_value(environ) else _Host(environ)
                current = current.give(ENVIRON_TAG, handed)
                if _stopped(current):
                    return current
                current = current.ok_value
            return self._finish(current)
        return Ok(current)

    def _is_the_environment(self, argument) -> bool:
        """Whether this written argument says the environment, by its tag.

        The environment is what `args.env` answers; a written argument is that
        environment when the call gives it the `$env` tag the runtime function
        declares. The value itself says it too — an environment handed to a call
        is what runs it (`_Pending.give`) — which is what the layers that hold an
        argument, rather than its text, read.
        """
        if isinstance(argument, viba_ast.Tagged):
            return argument.tag == ENVIRON_TAG
        return False

    def _argument_is_deferred(self, current, argument):
        """Whether this written argument lands on a function-typed slot — then it
        is not computed here at all: the host runs it with an environment of its
        own. The environment is never deferred: giving it is what runs the call.

        A module's arguments are values it receives as one product, so nothing is
        deferred there: what a module's slot asks for is computed here, and a
        function-typed one is the closure it writes.
        """
        if not isinstance(current, _Pending) or current.kind != "func":
            return False
        if self._is_the_environment(argument):
            return False
        index = current.index_for(argument)
        if index is None:
            return False
        return _function_slot(current, index) is not None

    def _finish(self, pending):
        """The chain ended: run it when the environment is in, store it when not.

        Giving the environment is what execution is, so a pending call that has
        one is a call being made — its arguments have to be complete. A pending
        call without one is a value: a closure, written down and serializable.
        """
        if pending.environ is None:
            return pending.as_viba_data()
        missing = pending.missing()
        if missing is not None:
            return VibaProgramErr(missing)
        return pending.fire()

    def _target_and_arguments(self, node, scope):
        """(what the chain calls, its written arguments) — or (a Result, None).

        A name at the head is read as the definition it names, not as the value
        it stands for: `add` at the head is that call, while `add` in an argument
        is the closure. A closure stored as viba data is the chain it was made
        from, so it is read apart here, its arguments coming first.
        """
        head, arguments = _call_parts(node)
        while True:
            if isinstance(head, viba_ast.Member):
                return Ok(_Member(head.tag)), arguments
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
            if isinstance(got, _VibaData) and got.given is not None:
                written_in = _writing_module(got)
                stored_head, passed = _stored_arguments(got)
                if written_in is not None and written_in is not self.module:
                    other = self._in_module(written_in)
                    target = (other._call_target(stored_head, scope)
                              if other is not None
                              and isinstance(stored_head, viba_ast.TypeRef) else None)
                    if target is not None:
                        if _stopped(target):
                            return target, None
                        return target, passed + arguments
                head = stored_head
                arguments = passed + arguments
                continue
            if isinstance(got, _VibaData) and isinstance(got.node.data, viba_ast.Partial):
                head, stored = _call_parts(got.node.data)
                arguments = stored + arguments
                continue
            if isinstance(got, _VibaData) and isinstance(got.node.data, viba_ast.TypeRef):
                written_in = _writing_module(got)
                if written_in is not None and written_in is not self.module:
                    other = self._in_module(written_in)
                    target = (other._call_target(got.node.data, scope)
                              if other is not None else None)
                    if target is not None:
                        if _stopped(target):
                            return target, None
                        return target, arguments
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
        definition = _definition(self.module, name)
        if definition is not None:
            return self._pending_of(definition, name_node, name, self.module, name)
        chain = self._member_function(name)
        if chain is not None:
            return self._func_pending(name_node, name, self.module, chain, name)
        bound = self._imported_name(name)
        if bound is None:
            return None
        module_name, rest = bound
        imported = self.runner.imported(module_name, self.file)
        if _stopped(imported):
            return imported
        if rest:
            return self._member_target(imported.ok_value, module_name, rest, name_node,
                                       name, written_in=self.module)
        return Ok(_Pending.module(self.runner, imported.ok_value, module_name,
                                  name_node, written_in=self.module))

    def _builtin_member(self, name):
        """`builtin.echo`：内建库里那个概念的一个成员。"""
        head, dot, tag = name.rpartition(".")
        if not dot:
            return None
        found = BUILTIN_MODULE.lookup(head)
        if not isinstance(found, Ok) or not isinstance(found.ok_value, AstNodeType):
            return None
        body = found.ok_value.ast_node
        body = body.body if isinstance(body, viba_ast.TypeDefinition) else body
        for factor in product_elements(body):
            if isinstance(factor, viba_ast.Tagged) and factor.tag == "$" + tag:
                inner = factor.type
                return Ok(_VibaData(VibaNode(
                    reflect_access,
                    descriptor_of(AstNodeType(inner, BUILTIN_MODULE)), inner)))
        return VibaProgramErr(f"{head!r} has no member tagged {'$' + tag!r}")

    def _member_function(self, name):
        """`a.b` 写在链头、而 `b` 是 `a` 的一个函数成员时，这一步的函数体。

        点分名字定义的是父概念的一个成员（viba-style.md），所以 `a.b << …` 调的
        就是那一步，而名字仍然是写下来的整串 —— 宿主拿到的 `func_name` 就是它。
        不是函数成员、或者父概念下没有这个成员时，答 None，交给别的读法。
        """
        if "." not in name:
            return None
        member = self._tagged_member(name)
        if member is None or not isinstance(member, Ok):
            member = self._builtin_member(name)
        if not isinstance(member, Ok) or not isinstance(member.ok_value, _VibaData):
            return None
        piece = member.ok_value.node.data
        if isinstance(piece, (viba_ast.Exponent, viba_ast.ExponentChain)):
            return piece
        return None

    def _pending_of(self, definition, name_node, name, owner_module, written,
                    local_name="", written_in=None):
        """A definition read as the call it stands for, or None when it is no
        function (then the name is a plain value). `written` is how it was
        written, `local_name` how the host knows it."""
        body = definition.body
        if isinstance(body, (viba_ast.Exponent, viba_ast.ExponentChain)):
            return self._func_pending(name_node, written, owner_module, body,
                                      local_name or name, written_in=written_in)
        return None

    def _func_pending(self, name_node, written, owner_module, chain, name,
                      written_in=None):
        """The call a function stands for. Every executable function depends on
        the environment, so a chain without that slot can never run: saying so
        here is what keeps such a call from becoming a closure that never runs."""
        slots = _slots_of(chain)
        envs = [one for one in slots
                if isinstance(one, viba_ast.Tagged) and one.tag == ENVIRON_TAG]
        if not envs:
            return VibaProgramErr(
                f"{written} takes no {ENVIRON_TAG} {ENV_TYPE} parameter: "
                f"every function that runs depends on the environment")
        if not names_the_environment(envs[0].type):
            return VibaProgramErr(
                f"{written}: the {ENVIRON_TAG} parameter must be {ENV_TYPE}, "
                f"not {viba_ast.unparse_type(envs[0].type)}")
        return Ok(_Pending.func(self, name_node, written, owner_module, slots,
                                name=name, written_in=written_in))

    def _member_target(self, module, module_name, rest, name_node, written,
                       written_in=None):
        """`demo.print` read as the call it stands for."""
        definition = _definition(module, rest)
        if definition is None:
            return VibaProgramErr(f"module {module_name!r} has no {rest!r}")
        other = _Activation(self.runner, module, self.environ, module_name,
                            self.runner.path_of.get(module_name))
        return other._pending_of(definition, name_node, rest, module, written,
                                 written_in=written_in)

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
        argument, in the environment it hands over.
        """
        if isinstance(node, viba_ast.CodeBlock):
            return Ok(None)
        tag, inner = _addressed(node)
        return Ok(_Given(tag, _Getter(self, inner, scope)))


def _slots_of(chain):
    """A written function's argument slots.

    The first element is the result type and the last one is the body: a written
    call the chain ends on, or documentation, or nothing at all when the chain is
    only a function type. Everything between them is an argument.
    """
    elements = _elements(chain)[1:]
    if elements and isinstance(elements[-1], (viba_ast.CodeBlock, viba_ast.Partial)):
        elements = elements[:-1]
    return [element for element in elements
            if not isinstance(element, viba_ast.CodeBlock)]


def _call_parts(node):
    """A written call as (head, arguments in written order).

    A closure kept as a value is that same written call, so applying more
    arguments to one is only reading it apart again.
    """
    written = []
    while isinstance(node, viba_ast.Partial):
        written.append(node.argument)
        node = node.function
    return node, list(reversed(written))


class _HostArgument:
    """A written argument of a function-typed slot, while the host holds it.

    A slot written as a function type — `$get_v (T <- $env Environment)` — asks
    for a function of the environment, so the argument written there is not
    computed here: this carries the written expression and the call it belongs
    to, and the host runs it with an environment of its choosing. A host that
    does not call it never computes it.

    **At most once**: what the expression answered for the first environment is
    handed back for every later one, so a value the host asks for twice is
    computed once — an argument is one argument.
    """

    def __init__(self, node, scope, pending, slot, module, owner, file=None,
                 written_in=None, name="", origin=None):
        self.node = node
        self.scope = scope              # the block the argument was written in
        self.pending = pending
        self.slot = slot                # the declared type inside the function
        self.module = module
        self.owner = owner
        self.file = file                # the module the argument was written in
        self.written_in = written_in                # where the written names resolve
        self.origin = origin            # the activation that wrote it, if it is known
        self.name = name                # the module's identity, for messages
        self.answer = None              # the Result, once it is worked out
        self.environ = None             # the environment it was called with
        self.closure = False            # the value a name stands for, once asked for

    def closure_of(self):
        """The value this argument stands for without an environment, or None.

        A name that stands for a closure (`add << $a 40`) is that closure already:
        evaluating it answers the written call, and no environment is needed. A
        call still owing an argument (`f << $x`) is not: it has to run, and
        evaluating it here would compute it. The answer is kept, and asking is
        only what the host does when it hands the argument straight back.
        """
        if self.closure is not False:
            return self.closure
        self.closure = None
        activation = self.pending.activation
        if activation is not None:
            answer = activation.evaluate(self.node, self.scope)
            if isinstance(answer, Ok) and _is_a_written_call(answer.ok_value):
                self.closure = answer.ok_value.node
        return self.closure

    def __call__(self, environ):
        if self.answer is None:
            self.answer = self._compute(environ)
        if not isinstance(self.answer, Ok):
            raise _Raised(self.answer)  # 停下就照原样报回去，不变成一份值
        value = self.answer.ok_value
        if (isinstance(self.node, viba_ast.TypeRef)
                and isinstance(value, _VibaData)
                and isinstance(value.node.data, viba_ast.Partial)):
            # 给的名字本身就是一份闭包：宿主拿到的就是那份闭包，不是这个名字。
            return value.node
        return _argument_value(_as_given_value(value))

    def _compute(self, environ):
        pending = self.pending
        source = pending.activation
        written_in = pending.written_in or pending.module
        given = _as_environ_value(environ)
        if self.environ is None:
            self.environ = given
        # The host hands an environment over, so the argument runs in it — in a
        # child of the module it was written in, that is where the written names
        # resolve. The declaration that named the slot is `self.module`.
        #
        # When that module is the one that wrote the call, the activation carries
        # **its own arguments**: reading a definition of it may read `args`, and
        # the arguments a module was called with are not a property of its text.
        writing_module = self.written_in or written_in
        origin = getattr(self, "origin", None)
        if origin is not None and origin.module is writing_module:
            wrote_args, wrote_members = origin.args, origin.members
        elif writing_module is source.module:
            wrote_args, wrote_members = source.args, source.members
        else:
            wrote_args, wrote_members = None, None
        caller = _Activation(source.runner, writing_module,
                             given.obj if _is_environ_value(given) else None,
                             self.name, self.file, wrote_args,
                             members=wrote_members)
        if isinstance(self.node, viba_ast.Partial):
            head, _written = _call_parts(self.node)
            if isinstance(head, viba_ast.Partial):
                # 写下来的是一份闭包（存下来的那次调用）：它代表的那次调用还没
                # 给环境，交回去的就是它。
                answer = caller.evaluate(self.node, self.scope)
                if not isinstance(answer, Ok):
                    return answer
                problem = _fits_slot(answer.ok_value, self.slot, self.module, self.owner)
                if problem is not None:
                    return VibaProgramErr(problem)
                return answer
        answer = caller._apply_without_environ(self.node, self.scope, given)
        if not isinstance(answer, Ok):
            return answer
        if self.slot is None:
            return answer
        problem = _fits_slot(answer.ok_value, self.slot, self.module, self.owner)
        if problem is not None:
            return VibaProgramErr(problem)
        return answer


def _function_slot(pending, index):
    """The `(T <- $env Environment)` a slot asks for, or None.

    A function type is `T <- $env Environment`: the result is the left of the
    exponent and the environment is what it is called with. The result is what
    the host's answer has to fit, so that is what comes back.
    """
    if index >= len(pending.elements):
        return None
    written = pending.elements[index]
    inner = written.type if isinstance(written, viba_ast.Tagged) else written
    if isinstance(inner, viba_ast.Exponent):
        return inner.result
    if isinstance(inner, viba_ast.ExponentChain):
        parts = _elements(inner)
        return parts[0] if parts else None
    return None


def _handed_to_host(pending, index, value):
    if isinstance(value, _HostArgument):
        return value
    if _function_slot(pending, index) is not None and _is_a_written_call(value):
        return _HostArgument(value.node.data, pending, _function_slot(pending, index),
                             pending.module, pending.written,
                             written_in=pending.activation.module if pending.activation else None,
                             name=pending.activation.name if pending.activation else "",
                             origin=pending.activation)
    return _argument_value(value)


def _stored_arguments(value):
    """(head, arguments) for a call kept as a value, in written order.

    What the call stored is the **values** it was given, each with the module it was
    written in; a function-typed one is stored as the call it stands for and is read
    from the chain instead. Both land in one list, in the order they were written.
    """
    head, stored = _call_parts(value.node.data)
    stored_values = value.given or []
    arguments = []
    for index, piece in enumerate(stored):
        tag, remembered = (stored_values[index] if index < len(stored_values)
                           else (None, None))
        arguments.append(_Given(tag, remembered) if remembered is not None else piece)
    return head, arguments


def _writing_module(value):
    """The module whose text this piece is — where its names resolve, or None.

    A piece that travelled — a closure stored as a value, an argument handed to
    another module — is read again in the module it came from, so that `F` means
    there what it meant where it was written. A host value has none.
    """
    if not isinstance(value, _VibaData):
        return None
    return value.written_in


def _closure_module(value):
    """The module a stored closure was written in, or None when it says."""
    return _writing_module(value)


def _is_a_written_call(value) -> bool:
    """Whether this value is a call written down rather than an answer.

    A closure kept as a value is the chain it was made from (`f << $a 1`), and a
    name kept as a value is the call it stands for, so both take arguments by
    being read on. Viba data that is an answer — a literal, a product — does not.
    """
    return (isinstance(value, _VibaData)
            and isinstance(value.node.data, (viba_ast.Partial, viba_ast.TypeRef)))


def _host_give(function, item):
    """Give one argument to what the environment handed over, or say why not.

    A member of the environment counts its own slots and takes the values in
    written order. Anything else that was handed an argument is named by its
    kind, and the message stays the same between runs — an object's repr would
    carry an address that changes every time.
    """
    if isinstance(function, _HostFunction):
        return function.give(item)
    if isinstance(function, _GetArgs):
        return function.give(item)
    if isinstance(function, _VibaData):
        return VibaProgramErr(f"{function.node!r} is not a function: it is a value")
    if isinstance(function, _Host):
        return VibaProgramErr(f"{type(function.obj).__name__} is not a function")
    return VibaProgramErr(f"{type(function).__name__} is not a function")


def _required_arguments(func) -> int:
    """How many arguments a host callable insists on.

    A member whose parameter is the empty product is called with no argument at
    all; a callable that cannot be read is taken to insist on one, the way every
    environment member used to.
    """
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return 1
    return sum(1 for parameter in parameters
               if parameter.default is inspect.Parameter.empty
               and parameter.kind in (parameter.POSITIONAL_ONLY,
                                      parameter.POSITIONAL_OR_KEYWORD))


def _is_environ_value(value) -> bool:
    """Whether this value is an environment — giving one is what execution is."""
    return isinstance(value, _Host) and isinstance(value.obj, Environment)


def _as_environ_value(value):
    """An environment a host handed over, as a value this run can give on.

    The host side speaks plain objects and `VibaNode`s, while inside a run an
    environment is the `_Host` that carries it; a node that writes the
    environment is viba data and stays one. That is the same pair `_answer`
    answers with, so this only says which of the two it is.
    """
    if isinstance(value, _HostArgument):
        return value
    if _is_environ_value(value):
        return value
    if isinstance(value, Environment):
        return _Host(value)
    answer = _given_value(value)
    if _is_environ_value(answer) or isinstance(answer, _VibaData):
        return answer
    return None


class _Pending:
    """A call being prepared: a function or a module, and what it has been given.

    It lives inside one chain. When the chain ends it either runs (the
    environment is in, so the arguments must be complete) or becomes viba_data —
    the function's name and the arguments already computed, which is a closure.
    That is why a half-given call cannot be stored, and why the state of having
    no environment is itself a serializable value.
    """

    def __init__(self, kind, activation=None, head=None, written="", name="",
                 module=None, written_in=None, elements=(), runner=None,
                 module_name=None):
        self.kind = kind                     # "func" | "module"
        self.activation = activation         # 函数：定义在哪个模块里
        self.head = head                     # 闭包写成什么：函数名或模块名那个节点
        self.written = written               # 写出来的样子：给人和闭包用
        self.name = name or written          # 定义名：宿主按这个名字找实现
        self.module = module                 # 参数声明在哪个模块：核对类型用
        self.written_in = written_in or module           # 这次调用写在哪个模块：写回可序列化数据用
        self.elements = list(elements)       # 各参数，按书写顺序
        self.runner = runner                 # 模块：谁来跑它
        self.module_name = module_name
        self.environ = None                  # 环境；给了就是执行
        self.given = {}                      # 参数序号 -> 值

    @classmethod
    def func(cls, activation, head, written, owner_module, elements, name="",
             written_in=None):
        return cls("func", activation=activation, head=head, written=written,
                   name=name, module=owner_module, written_in=written_in, elements=elements)

    @classmethod
    def module(cls, runner, module, module_name, head, written_in=None):
        return cls("module", runner=runner, head=head, written=module_name,
                   module=module, written_in=written_in, module_name=module_name)

    # ---- what this call is ----

    @property
    def slots(self):
        """Each slot's tag, or None for a position."""
        return [element.tag if isinstance(element, viba_ast.Tagged) else None
                for element in self.elements]

    def slot_tags(self):
        """The tags of the slots this pending call fills, in order — a module's
        slots come from its `__def__`, a function's from its chain."""
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

    def index_for(self, node):
        """Which slot this written argument goes to, or None when there is none
        (then `give` reports it)."""
        tag, _ = _addressed(node)
        index, _problem = self.slot_for(tag)
        return index

    def slot_for(self, tag):
        """(index, problem) for a written argument: by its tag, or the next free
        slot when it carries none."""
        slots = self.slots
        if len(self.given) == len(slots):
            return None, f"{self.written} takes no more arguments"
        if tag is not None:
            if tag not in slots:
                return None, f"{self.written} takes no {tag} argument"
            return slots.index(tag), None
        free = [one for one in range(len(slots)) if one not in self.given]
        if not free:
            return None, f"{self.written} takes no more arguments"
        return free[0], None

    def ready(self) -> bool:
        """Whether the call can run: the environment is in and every argument."""
        if self.environ is None:
            return False
        if self.kind == "module":
            slots, _problem = _module_arg_slots(self.module)
            return len(self.given) == len(slots or [])
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
                f"{len(slots)} parameters: {', '.join(left)} missing")

    # ---- taking arguments ----

    def give(self, tag, value):
        """Put one computed argument in the slot it goes to."""
        if self.kind == "module":
            return self._give_module(tag, value)
        return self._give_func(tag, value)

    def _give_func(self, tag, value):
        index, problem = self.slot_for(tag)
        if problem is not None:
            return VibaProgramErr(problem)
        element = self.elements[index]
        if index == self.environ_slot():
            if not _is_environ_value(value):
                return VibaProgramErr(f"{self.written} was not given an {ENVIRON_TYPE}")
            self.given[index] = value
            self.environ = value.obj
            return Ok(self)
        wanted = _function_slot(self, index)
        if wanted is not None and isinstance(value, _Getter):
            # 函数类型的槽收的是写下来的那次调用：这里不算它，宿主叫它的时候才算，
            # 在宿主给的那个环境里算（`_HostArgument`）。
            self.given[index] = _HostArgument(
                value.node, value.scope, self, wanted, self.module, self.written,
                value.file, written_in=value.written_in, name=value.name,
                origin=getattr(value, "origin", None))
            return Ok(self)
        problem = _fits_slot(value, element, self.module, self.written)
        if problem is not None:
            return VibaProgramErr(problem)
        self.given[index] = value
        return Ok(self)

    def _give_module(self, tag, value):
        slots, problem = _module_arg_slots(self.module)
        if problem is not None:
            return VibaProgramErr(f"module {self.module_name!r}: {problem}")
        # 环境不是实参：它就是执行这一步，按值认，或者按 $env 这个 tag 认。
        if self.environ is None and (tag == ENVIRON_TAG or _is_environ_value(value)):
            if not _is_environ_value(value):
                return VibaProgramErr(
                    f"module {self.module_name!r} was not given an {ENVIRON_TYPE}: "
                    f"its {ENVIRON_TAG} parameter asks for {ENV_TYPE}")
            self.environ = value.obj
            return Ok(self)
        if not slots:
            # 这个模块只收环境：环境之外的实参一个都没有，所以给什么都不对。
            return VibaProgramErr(
                f"module {self.module_name!r} needs an {ENVIRON_TYPE}: its "
                f"{DEF_NAME} takes no other parameters")
        tags = [one for one, _written in slots]
        if tag is not None:
            if tag not in tags:
                return VibaProgramErr(
                    f"module {self.module_name!r} takes no {tag} argument: its "
                    f"{DEF_NAME} parameters are {_written_slots(slots)}")
            index = tags.index(tag)
            if index in self.given:
                return VibaProgramErr(
                    f"module {self.module_name!r} was given {tag} twice")
        else:
            free = [one for one in range(len(slots)) if one not in self.given]
            if not free:
                return VibaProgramErr(
                    f"module {self.module_name!r} takes no more arguments: its "
                    f"{DEF_NAME} parameters are all given")
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

    def as_viba_data(self):
        """Without an environment this pending call is viba data — a closure.

        The function's name and the arguments already computed, written back as
        the chain they came from. Only viba_data goes in: that is what makes the
        closure serializable, and it is the same thing every function and module
        answers with.
        """
        node = self.head
        tags = self.slot_tags()
        kept = []
        for index in sorted(self.given):
            value = self.given[index]
            if isinstance(value, _Getter):
                return VibaProgramErr(
                    f"{self.written}: the {_slot_name(index, tags[index])} argument "
                    f"is computed only when it is wanted, so it cannot be stored: "
                    f"give it in the chain that runs")
            if isinstance(value, _HostArgument):
                # A function-typed slot was handed the written call itself, which
                # is viba_data: the closure is that call, waiting for an
                # environment.
                piece = value.node
            elif isinstance(value, _VibaData):
                piece = value.node.data
            else:
                return VibaProgramErr(
                    f"{self.written}: a closure holds viba_data only; the "
                    f"{_slot_name(index, tags[index])} argument is not")
            tag = tags[index]
            kept.append((tag, value if isinstance(value, _VibaData) else None))
            node = viba_ast.Partial(node, viba_ast.Tagged(tag, piece) if tag else piece)
        return Ok(_VibaData(VibaNode(reflect_access, self.descriptor(node), node),
                            written_in=self.written_in, given=kept))

    def descriptor(self, node):
        """What the design calls this piece.

        For a closure that is the written call itself — the head name — and not
        the type it would have once executed: the node is the call as it stands,
        and the spelling of a value comes from its own piece.
        """
        if isinstance(node, viba_ast.Partial):
            return descriptor_of(AstNodeType(self.head, self.written_in))
        return descriptor_of(AstNodeType(node, self.written_in))

    def call_viba_data(self):
        """The viba_data this call was given, as it was written.

        A host value — the environment above all — is no viba_data and does not
        travel: the side that answers makes its own. One viba_data argument is
        that argument itself (no tag is needed to tell it from the others),
        which is the `$call` a Prepare of such a call fixes; several make a
        product, keeping the tags as written. None when nothing viba_data was
        given.
        """
        tags = self.slot_tags()
        viba_data_given = [(tags[index], value.node.data)
                          for index, value in sorted(self.given.items())
                          if isinstance(value, _VibaData)]
        if not viba_data_given:
            return None
        if len(viba_data_given) == 1:
            return viba_data(viba_data_given[0][1])
        written = [viba_ast.Tagged(tag, piece) if tag else piece
                   for tag, piece in viba_data_given]
        return viba_data(viba_ast.ProductChain(written))

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
            return _refused(deferred, step, self.call_viba_data())
        except Exception as exc:            # the host is the host's business
            return UnderlyingVibaOpFailed(
                f"get_func({module_path!r}, {self.name!r}) raised {exc!r}",
                step, REASON_GET_FUNC_RAISED)
        if host is None:
            return _no_implementation(step, self.call_viba_data())
        handed = [_handed_to_host(self, index, self.given[index])
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
        """Every parameter of `__def__` is in: hand them to the module, as one
        product, the environment among its members (`args.env`)."""
        params, problem = _module_params(self.module)
        if problem is not None:
            return VibaProgramErr(f"module {self.module_name!r}: {problem}")
        nodes = []
        kept = []
        members = []
        for tag, _written, is_env, index in params or []:
            if is_env:
                members.append((tag, _Host(self.environ)))
                continue
            value = self.given.get(index)
            if value is None:
                return VibaProgramErr(
                    f"module {self.module_name!r} is not ready to run")
            if not isinstance(value, _VibaData):
                return VibaProgramErr(
                    f"module {self.module_name!r}: the {_slot_name(index, tag)} argument "
                    f"is not viba data, so it cannot be part of {DEF_NAME}")
            nodes.append(viba_ast.Tagged(tag, value.node.data) if tag else value.node.data)
            kept.append(value)
            members.append((tag, value))
        if not nodes:
            # 只有环境：这次调用收到的是那份环境，没有别的成员（`args.env` 走的就是它）。
            args = _VibaData(viba_data(None))
        elif len(nodes) == 1:
            descriptor = kept[0].node.descriptor
            if isinstance(nodes[0], viba_ast.Tagged):
                # 一个成员也是按 tag 寻址的：那一位是值的读数，tag 另加在外面。
                descriptor = descriptor_of_tagged(nodes[0].tag, descriptor,
                                                  nodes[0], self.written_in)
            args = _VibaData(VibaNode(reflect_access, descriptor, nodes[0]))
        else:
            chain = viba_ast.ProductChain(nodes)
            args = _VibaData(VibaNode(
                reflect_access,
                descriptor_of_values(chain, self.written_in, kept), chain))
        answer = _run_module(self.runner, self.module, self.environ, self.module_name,
                             None, args, members=members)
        if _stopped(answer):
            return answer
        # `interpret` hands the node out; inside a run a module's answer is a
        # value like any other, so it goes back into the value model.
        node = answer.ok_value
        return Ok(_VibaData(node) if isinstance(node, VibaNode) else _Host(node))


class _GetArgs:
    """`__get_args__ << __def__`: the arguments this call was handed.

    Read as a type, the same call answers the product of `__def__`'s parameters
    (viba.is_sub_type reads it that way); run, it answers the product the module
    received, the environment among its members — which is how a module reads the
    environment (`args = __get_args__ << __def__`, then `args.env`). What it is
    given is the module's own `__def__`: what a call received is not a property
    of the text it was written with.
    """

    __slots__ = ("activation",)

    def __init__(self, activation):
        self.activation = activation

    def give(self, item):
        """One argument — the module's `__def__` — and this call's arguments."""
        if self.activation.args is None:
            return VibaProgramErr(
                f"{GET_ARGS_NAME} is the arguments of a module call, and no module "
                f"call is running here")
        return Ok(self.activation.args)


class _HostFunction:
    """A function the host hangs off the environment: `args.env.sub_env`."""

    def __init__(self, name: str, func, slots: int = 1, given=None,
                 module_path: str = ""):
        self.name = name
        self.func = func
        self.slots = slots
        self.given = list(given or [])
        self.module_path = module_path

    def filled(self) -> bool:
        """Whether every argument it asked for is in."""
        return len(self.given) >= self.slots

    def run(self):
        """Run it: a chain that ends with every argument in makes the call."""
        return self._call(self.given)

    def give(self, item):
        values = self.given + [_argument_value(item.value)]
        if len(values) < self.slots:
            return Ok(_HostFunction(self.name, self.func, self.slots, values,
                                    self.module_path))
        return self._call(values)

    def _call(self, values):
        step = Step(self.module_path, f"environ.{self.name}")
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
    if isinstance(answer, _HostArgument):
        # 宿主把那个写下来的调用原样递回来：它代表的那份值就是这次调用。名字代
        # 表的那份闭包由 `__call__` 算出来，所以这里先问它一次。
        # 宿主把那个写下来的调用原样递回来：它代表的那份值就是这次调用。
        closure = answer.closure_of()
        if closure is not None:
            return Ok(_VibaData(closure))
        return Ok(_VibaData(VibaNode(
            reflect_access, descriptor_of(AstNodeType(answer.node, answer.module)),
            answer.node)))
    """Result: what a host function answered, as a value.

    A `VibaNode` is taken as it is, an `Environment` stays a host value, and
    `None` is `nil` the way it is in the builder. A plain Python value lands
    as a leaf — but only a scalar one: a list, a dict, a callable or any other
    object has no leaf to be, and guessing one would put a piece into the
    viba_data that no design asked for. Given a `step`, that refusal is a
    failure of it; without one — a value a host is handing back into a call —
    it is a plain `VibaProgramErr`.
    """
    if isinstance(answer, VibaNode):
        return Ok(_VibaData(answer))
    if isinstance(answer, Environment):
        return Ok(_Host(answer))
    if answer is not None and not isinstance(answer, (bool, int, float, str)):
        msg = (f"{name} answered {type(answer).__name__}, "
               f"which is no leaf: answer a VibaNode, a scalar, or None")
        if step is not None:
            return UnderlyingVibaOpFailed(msg, step, REASON_NO_LEAF)
        return VibaProgramErr(msg)
    node = viba_ast.Nil() if answer is None else viba_ast.Constant(answer)
    return Ok(_VibaData(VibaNode(reflect_access,
                                 descriptor_of(AstNodeType(node, _NO_MODULE)), node)))


def _elements(node):
    if isinstance(node, viba_ast.ExponentChain):
        return list(node.elements)
    if isinstance(node, viba_ast.Exponent):
        return _elements(node.result) + [node.argument]
    return [node]


__all__ = ["interpret", "Environment", "EnvironmentStorage", "EnvironmentCompute",
           "viba_data", "snapshot_path", "read_snapshot", "write_snapshot", "replayed"]
