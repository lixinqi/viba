"""viba.rule.reflect — 把一份设计当图，去材料里按图索骥。

这是 ``viba-reflect.md`` 的访问侧，落在 viba.rule 上：地图（描述符）由描述符侧
从设计给出，数据（Data）是一份 witness 的类型表达式，访问就是按地图上的坐标
一步一步在 witness 里取。名字照 ``viba_type_descriptor.py`` 对
``viba_type_descriptor.viba`` 的老规矩来：协议里的类名原样保留，协议里的函数
落 snake_case，本层自己的辅助加下划线。

    协议（viba-reflect.md）            Python
    ------------------------------    ------------------------------------
    VibaStep                          VibaStep
    VibaStep 的四支                    by_tag / by_field_index / at_index / at_key
    VibaPath                          VibaPath（= list[VibaStep]）
    VibaNode[Data]                    VibaNode
    VibaAccess[Data]                  VibaAccess
    VibaRoot / VibaHas / VibaGet      VibaAccess.root / has / get / leaf /
    VibaLeaf / VibaLength / VibaKeys  length / keys
    VibaResolve / VibaGetByPath       viba_resolve / viba_get_by_path /
    VibaListFields                    viba_list_fields

协议里没有名字的，是本层对它的绑定与辅助，不算协议概念：

    Data 这个形参         -> Witness：一份呈证材料（viba-rule.md 的 Witness）
    第 5.4 节"直接抛异常"  -> VibaReflectError
    读图与遍历            -> VibaAccess 上的下划线方法、以及本模块的下划线函数

两条协议里的规矩：

* ``root`` 只把 (描述符, 数据) 配成对。版本协议不管：一份材料对着哪一版设计
  做的，是上层自己的事。
* ``has`` 只答真假——设计里没这个坐标、数据里没这一段，都是 ``False``。
  ``get`` 里"数据里没有"是 ``Ok(nil)``，只有"图上压根没这个坐标"才是 ``Err``；
  ``leaf`` / ``length`` / ``keys`` 取不到就是 ``Err``。

第 5.4 节的 Python 落点在 VibaNode 上：``get_{name}()`` / ``has_{name}()`` /
``in`` / ``try_get_{name}()`` / ``node[i]`` / ``node.value`` / ``len(node)`` /
迭代 / ``keys()`` / ``values()`` / ``items()``。取值的那一类抛异常，问有没有的
那一类永不抛。
"""

from __future__ import annotations

import copy
from typing import Iterable, List, Optional, Sequence

from viba import viba_ast
from viba.type import AstNodeType, Err, Ok, Result, module_get_type
from viba.viba_type_descriptor import (
    CODE_BLOCK,
    ELLIPSIS,
    EXPONENT,
    LITERAL,
    NEVER,
    NIL,
    PRODUCT,
    SUM,
    TAGGED,
    TUPLE,
    TYPE_APP,
    TYPE_REF,
    VibaChainDescriptor,
    VibaConstantValue,
    VibaDefinitionDescriptor,
    VibaMemberDescriptor,
    VibaPool,
    VibaTaggedDescriptor,
    VibaTupleDescriptor,
    VibaTypeAppDescriptor,
    VibaTypeDescriptor,
    definition_members,
    member_type_name,
)

# 单位元链头。描述符侧只认 Object / nil / Oneof / never；规则标记是规则层的词汇，
# 由规则层自己认（所以这里比描述符侧多两个名字）。
UNIT_HEADS = ("Object", "nil", "RuleObject", "Oneof", "never", "OneofRule")

# 三种内建容器，以及实现里容器的字面量形状。
CONTAINERS = ("list", "set", "dict")
LITERAL_CTORS = ("ListLiteral", "SetLiteral", "DictLiteral")


class VibaReflectError(Exception):
    """取值那一路失败时抛的异常，带的是协议那句 ``Err`` 的话。

    第 5.4 节只说"直接抛异常"，没给异常起名字，这个名字是本层的。
    """


# ----------------------------------------------------------------------
# VibaStep 与 VibaPath
# ----------------------------------------------------------------------


class VibaStep:
    """地址的一步：四支之一。"""

    __slots__ = ("kind", "value")

    def __init__(self, kind: str, value):
        self.kind = kind
        self.value = value

    def __repr__(self):
        return f"{self.kind}({self.value!r})"

    def __eq__(self, other):
        return isinstance(other, VibaStep) and (self.kind, self.value) == (other.kind, other.value)

    def __hash__(self):
        return hash((self.kind, self.value))


def by_tag(name: str) -> VibaStep:
    return VibaStep("by_tag", _tag_of(name))


def by_field_index(index: int) -> VibaStep:
    return VibaStep("by_field_index", index)


def at_index(index: int) -> VibaStep:
    return VibaStep("at_index", index)


def at_key(key: str) -> VibaStep:
    return VibaStep("at_key", key)


VibaPath = List[VibaStep]


def _tag_of(name: str) -> str:
    """链式写法里名字不带 $；带上也认。"""
    return name if name.startswith("$") else "$" + name


# ----------------------------------------------------------------------
# Data 的绑定
# ----------------------------------------------------------------------


class Witness:
    """本层对协议里 Data 形参的绑定：一份呈证材料的类型表达式。

    协议没规定 Data 长什么样——它是实现方绑的。viba.rule 这边，"材料"就是
    viba-rule.md 说的呈证材料；版本之类的信息由上层自己带，协议不管。
    """

    __slots__ = ("node",)

    def __init__(self, node):
        if isinstance(node, AstNodeType):
            node = node.ast_node
        self.node = node


# ----------------------------------------------------------------------
# VibaNode[Data]
# ----------------------------------------------------------------------


class VibaNode:
    """一个节点：设计侧的一段描述符 + witness 侧的一段数据。

    ``path`` 是从起点走到这里的 VibaPath（协议的节点上没有这一栏，是本层为了让
    地址可存、可打印、可比较而留的痕）。
    """

    __slots__ = ("_access", "descriptor", "data", "path")

    def __init__(self, access: "VibaAccess", descriptor: VibaTypeDescriptor, data,
                 path: Sequence[VibaStep] = ()):
        self._access = access
        self.descriptor = descriptor
        self.data = data
        self.path = tuple(path)

    def __repr__(self):
        where = ".".join(repr(step) for step in self.path) or "root"
        return f"VibaNode({where})"

    # ---- 第 5.4 节的节点访问器：取值那一路抛异常 ----

    def by_tag(self, name: str) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, by_tag(name)))

    def by_field_index(self, index: int) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, by_field_index(index)))

    def at_index(self, index: int) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, at_index(index)))

    def at_key(self, key: str) -> "VibaNode":
        return self._access._unwrap(self._access.get(self, at_key(key)))

    @property
    def leaf(self) -> VibaConstantValue:
        return self._access._unwrap(self._access.leaf(self))

    @property
    def value(self) -> VibaConstantValue:
        """第 5.4 节的 Python 落点：``leaf`` 落到 ``node.value``。"""
        return self.leaf

    # ---- 形状：只看写出来的链头，不展开名字 ----

    @property
    def is_list(self) -> bool:
        return self._access._container_kind(self.descriptor) == "list"

    @property
    def is_set(self) -> bool:
        return self._access._container_kind(self.descriptor) == "set"

    @property
    def is_dict(self) -> bool:
        return self._access._container_kind(self.descriptor) == "dict"

    def __len__(self) -> int:
        return self._access._unwrap(self._access.length(self))

    def keys(self) -> List[str]:
        return list(self._access._unwrap(self._access.keys(self)))

    def values(self):
        return [self[key] for key in self.keys()]

    def items(self):
        return [(key, self[key]) for key in self.keys()]

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]

    def __getitem__(self, key):
        if isinstance(key, int) and not isinstance(key, bool):
            return self._access._unwrap(self._access.get(self, at_index(key)))
        return self._access._unwrap(self._access.get(self, at_key(str(key))))

    # ---- Python 落点的魔术方法 ----

    def _has_tag(self, name: str) -> bool:
        given = self._access.has(self, by_tag(name))
        return bool(given.value) if isinstance(given, Ok) else False

    def _has_field(self, index: int) -> bool:
        given = self._access.has(self, by_field_index(index))
        return bool(given.value) if isinstance(given, Ok) else False

    def __contains__(self, name: str) -> bool:
        return self._has_tag(name)

    def __getattr__(self, name: str):
        if name.startswith("get_field_"):
            return lambda: self.by_field_index(int(name[len("get_field_"):]))
        if name.startswith("has_field_"):
            return lambda: self._has_field(int(name[len("has_field_"):]))
        if name.startswith("try_get_field_"):
            return lambda: self._access.get(
                self, by_field_index(int(name[len("try_get_field_"):])))
        if name.startswith("get_"):
            return lambda: self.by_tag(name[len("get_"):])
        if name.startswith("has_"):
            return lambda: self._has_tag(name[len("has_"):])
        if name.startswith("try_get_"):
            return lambda: self._access.get(self, by_tag(name[len("try_get_"):]))
        raise AttributeError(name)

    def __dir__(self):
        names = ["value", "leaf", "is_list", "is_set", "is_dict",
                 "keys", "values", "items", "path", "descriptor", "data"]
        positional = 0
        for tag, _ in self._access._members(self) or []:
            if tag:
                short = tag[1:]
                names += [f"get_{short}", f"has_{short}", f"try_get_{short}"]
            else:
                names += [f"get_field_{positional}", f"has_field_{positional}",
                          f"try_get_field_{positional}"]
                positional += 1
        return sorted(names)


# ----------------------------------------------------------------------
# VibaAccess[Data]
# ----------------------------------------------------------------------


class VibaAccess:
    """绑好一份设计的访问器：第 5.2 节的六格，加本层的读图辅助。"""

    def __init__(self, definition: VibaDefinitionDescriptor):
        self.definition = definition
        self.pool = definition.pool

    def __repr__(self):
        return f"VibaAccess({self.definition.full_name!r})"

    # ---- 协议六格 ----

    def root(self, data: Witness) -> Result:
        """起点：给出 (描述符, 数据) 这一对。"""
        return Ok(VibaNode(self, self.definition.body, data.node))

    def has(self, node: VibaNode, step: VibaStep) -> Result:
        """VibaHas：这一步有没有。设计里没这个坐标、数据里没这一段，都是 false。"""
        if not self._knows(node, step, self._members(node)):
            return Ok(False)
        return Ok(self._value_at(node.data, step, node.descriptor) is not None)

    def get(self, node: VibaNode, step: VibaStep) -> Result:
        """VibaGet：往里走一步。数据里没有是 Ok(nil)，图上没这个坐标才是 Err。"""
        slots = self._members(node)
        if not self._knows(node, step, slots):
            return Err(f"规则里没有这个坐标：{step}（{node!r}）")
        descriptor = self._target_of(node, step, slots)
        if descriptor is None:
            return Err(f"规则里没有这个坐标：{step}（{node!r}）")
        piece = self._value_at(node.data, step, node.descriptor)
        if piece is None:
            return Ok(None)
        return Ok(VibaNode(self, descriptor, piece, tuple(node.path) + (step,)))

    def leaf(self, node: VibaNode) -> Result:
        data = node.data
        if isinstance(data, viba_ast.Constant):
            return Ok(_constant_value(data.value))
        if isinstance(data, viba_ast.Nil):
            return Ok(VibaConstantValue("nil", None))
        return Err(f"这一段不是叶子：{node!r}")

    def length(self, node: VibaNode) -> Result:
        elements = self._elements_of(node.data)
        if elements is None:
            return Err(f"这一段不是容器：{node!r}")
        return Ok(len(elements))

    def keys(self, node: VibaNode) -> Result:
        if not isinstance(node.data, viba_ast.TypeApp) or node.data.constructor != "DictLiteral":
            return Err(f"这一段不是 dict：{node!r}")
        out = []
        for pair in node.data.args:
            key = self._key_of(pair)
            if key is None:
                return Err(f"dict 的键不是字面量：{node!r}")
            out.append(_key_text(key))
        return Ok(out)

    # ---- 私有：取值那一路（Err 与 Ok(nil) 都抛） ----

    def _unwrap(self, given: Result):
        if isinstance(given, Err):
            raise VibaReflectError(given.message)
        if given.value is None:
            raise VibaReflectError("这一段没有值")
        return given.value

    # ---- 私有：地图（把描述符读成可寻址的形状） ----

    def _unfold(self, descriptor: VibaTypeDescriptor,
               seen: Optional[set] = None) -> VibaTypeDescriptor:
        """把写成名字的那一段展开到能寻址的形状。

        两件事，都按池子里的定义来：名字展开成定义体；泛型应用代入实参，
        也落到定义体上。和/积/指数一个规矩，没有别的例外。
        """
        seen = seen or set()
        if id(descriptor) in seen:
            return descriptor
        seen = seen | {id(descriptor)}
        if descriptor.kind == TYPE_REF:
            resolved = self._resolve_ref(descriptor)
            if resolved is None:
                return descriptor
            return self._unfold(resolved, seen)
        if descriptor.kind == TYPE_APP:
            applied = self._apply_generic(descriptor)
            if applied is not None:
                return self._unfold(applied, seen)
        return descriptor

    def _apply_generic(self, descriptor: VibaTypeDescriptor) -> Optional[VibaTypeDescriptor]:
        """构造子指向一个泛型定义时，给出代入实参后的定义体描述符。"""
        payload = descriptor.payload
        if payload.resolvable_type is None:
            return None
        resolved = module_get_type(payload.resolvable_type.container_module,
                                   payload.constructor_name)
        if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
            return None
        target = resolved.value
        if not isinstance(target.ast_node, viba_ast.GenericDefinition):
            return None
        params = list(target.ast_node.generic_params or [])
        if len(params) != len(payload.args):
            return None
        # 描述符侧还没有公开的"给一段类型表达式做描述符"的入口，先用它的构建器。
        # 内建库是按源码原样装的（没有规范化过），这里先过一遍规范化。
        from viba.viba_type_descriptor import _build_type

        body_node = viba_ast.convert_to_chain_style(target.ast_node.body)
        body = _build_type(self.pool, target.container_module, body_node)
        return self._substitute(body, dict(zip(params, payload.args)))


    def _substitute(self, descriptor: VibaTypeDescriptor,
                    bindings: dict) -> VibaTypeDescriptor:
        """按名字把形参换成实参描述符；只碰写名字的那一支。"""
        payload = descriptor.payload
        if descriptor.kind == TYPE_REF:
            return bindings.get(payload.type_name, descriptor)
        if descriptor.kind in (PRODUCT, SUM, EXPONENT):
            return VibaTypeDescriptor(descriptor.kind, VibaChainDescriptor(
                payload.pool, payload.resolvable_type,
                [self._substitute(e, bindings) for e in payload.elements]))
        if descriptor.kind == TYPE_APP:
            return VibaTypeDescriptor(TYPE_APP, VibaTypeAppDescriptor(
                payload.pool, payload.resolvable_type, payload.constructor_name,
                [self._substitute(a, bindings) for a in payload.args]))
        if descriptor.kind == TUPLE:
            return VibaTypeDescriptor(TUPLE, VibaTupleDescriptor(
                payload.pool, payload.resolvable_type,
                [self._substitute(e, bindings) for e in payload.elements]))
        if descriptor.kind == TAGGED:
            return VibaTypeDescriptor(TAGGED, VibaTaggedDescriptor(
                payload.pool, payload.resolvable_type, payload.tag,
                self._substitute(payload.tagged_type, bindings)))
        return descriptor

    def _resolve_ref(self, descriptor: VibaTypeDescriptor) -> Optional[VibaTypeDescriptor]:
        """一个名字指向一个透明定义时，给出那个定义体的描述符。"""
        resolvable = descriptor.payload.resolvable_type
        if resolvable is None:
            return None
        resolved = module_get_type(resolvable.container_module, descriptor.payload.type_name)
        if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
            return None  # 内建叶子、泛型形参之类：到此为止
        target = resolved.value
        if not isinstance(target.ast_node, viba_ast.TypeDefinition):
            return None  # 泛型定义要带实参才有体，裸名字到此为止
        # 描述符侧还没有公开的"给一段类型表达式做描述符"的入口，先用它的构建器。
        from viba.viba_type_descriptor import _build_type

        return _build_type(self.pool, target.container_module, target.ast_node.body)

    def _container_kind(self, descriptor: VibaTypeDescriptor) -> Optional[str]:
        """list / set / dict 三种内建容器；元组不算，它是按位置对位的积。"""
        if descriptor.kind == TYPE_APP and descriptor.payload.constructor_name in CONTAINERS:
            return descriptor.payload.constructor_name
        return None

    def _has_elements_by_index(self, descriptor: VibaTypeDescriptor) -> bool:
        """能按下标数、按下标取的：list / set / tuple。

        dict 不算：它的元素要靠 keys + at_key 取，按下标问就是图上没这个坐标。
        """
        return self._container_kind(descriptor) in ("list", "set") or descriptor.kind == TUPLE

    def _members(self, node: VibaNode) -> Optional[List[tuple]]:
        """这一段有哪些成员：[(tag 或 None, 描述符), ...]；没有成员就给 None。

        和链、积链、指数链一个规矩：成员就是 $elements；链头是单位元就不算。
        支链（$elements 里嵌套的那条链）算一个成员，与别的元素同等。
        """
        descriptor = self._unfold(node.descriptor)
        elements = None
        if descriptor.kind in (PRODUCT, SUM, EXPONENT):
            elements = list(descriptor.payload.elements)
        elif descriptor.kind == TAGGED:
            return [(descriptor.payload.tag, descriptor.payload.tagged_type)]
        if elements is None:
            return None
        if elements and _is_unit_descriptor(elements[0]):
            elements = elements[1:]
        slots = []
        for element in elements:
            if element.kind == TAGGED:
                slots.append((element.payload.tag, element.payload.tagged_type))
            else:
                slots.append((None, element))
        return slots

    def _knows(self, node: VibaNode, step: VibaStep, slots: Optional[List[tuple]]) -> bool:
        if step.kind == "by_tag":
            return any(tag == step.value for tag, _ in slots or [])
        if step.kind == "by_field_index":
            positional = [tag for tag, _ in slots or [] if tag is None]
            return 0 <= step.value < len(positional)
        if step.kind == "at_index":
            return self._has_elements_by_index(self._unfold(node.descriptor))
        if step.kind == "at_key":
            return self._container_kind(self._unfold(node.descriptor)) == "dict"
        return False

    def _target_of(self, node: VibaNode, step: VibaStep, slots: Optional[List[tuple]]):
        if step.kind == "by_tag":
            for tag, descriptor in slots or []:
                if tag == step.value:
                    return descriptor
            return None
        if step.kind == "by_field_index":
            positional = [descriptor for tag, descriptor in slots or [] if tag is None]
            if 0 <= step.value < len(positional):
                return _as_value(positional[step.value])
            return None
        if step.kind == "at_index":
            shape = self._unfold(node.descriptor)
            if not self._has_elements_by_index(shape):
                return None
            if shape.kind == TUPLE:
                if 0 <= step.value < len(shape.payload.elements):
                    return shape.payload.elements[step.value]
                return None
            return shape.payload.args[0]
        if step.kind == "at_key":
            shape = self._unfold(node.descriptor)
            if self._container_kind(shape) != "dict":
                return None
            return shape.payload.args[1]
        return None

    # ---- 私有：数据侧 ----

    def _value_at(self, data, step: VibaStep, design=None):
        """数据里对应的那一段；没有就是 None（``...`` 不算数据）。"""
        return _as_value(self._match(data, step, design))

    def _expand_data(self, data, design):
        """数据写成名字或泛型应用时展开：名字给定义体，应用代入实参。

        与设计侧一个规矩（那边是描述符，这边是语法树）；名字要在设计那一段
        所属的模块里解析，所以用 design 带下来的模块。
        """
        module = _design_module(design)
        if module is None:
            return data
        if isinstance(data, viba_ast.TypeRef):
            body = _definition_body(module, data.name)
            return data if body is None else body
        if isinstance(data, viba_ast.TypeApp):
            definition = _generic_definition(module, data.constructor)
            if definition is None:
                return data
            params = list(definition.generic_params or [])
            if len(params) != len(data.args):
                return data
            bindings = dict(zip(params, data.args))
            return _fill_params(definition.body, bindings)
        return data

    def _match(self, data, step: VibaStep, design=None):
        if step.kind in ("by_tag", "by_field_index"):
            slots = _data_members(self._expand_data(data, design))
            if slots is None:
                return None
            if step.kind == "by_tag":
                for tag, piece in slots:
                    if tag == step.value:
                        return piece
                return None
            positional = [piece for tag, piece in slots if tag is None]
            if 0 <= step.value < len(positional):
                return positional[step.value]
            return None
        if step.kind == "at_index":
            elements = self._elements_of(data)
            if elements is None or not (0 <= step.value < len(elements)):
                return None
            return elements[step.value]
        if step.kind == "at_key":
            if not isinstance(data, viba_ast.TypeApp) or data.constructor != "DictLiteral":
                return None
            for pair in data.args:
                key = self._key_of(pair)
                if key is not None and _key_text(key) == step.value:
                    value = pair.elements[1] if len(pair.elements) > 1 else None
                    return value
            return None
        return None

    def _key_of(self, pair):
        """dict 一项的键；不是字面量就是 None。"""
        if isinstance(pair, viba_ast.Tuple) and pair.elements:
            key = pair.elements[0]
            if isinstance(key, viba_ast.Constant):
                return key.value
        return None

    def _elements_of(self, data) -> Optional[List]:
        if isinstance(data, viba_ast.Tuple):
            return list(data.elements)
        if isinstance(data, viba_ast.TypeApp) and data.constructor in LITERAL_CTORS:
            return list(data.args)
        return None

    # ---- 私有：成员遍历（VibaListFields 与 walk 用） ----

    def _is_unit_member(self, member: VibaMemberDescriptor) -> bool:
        written = member_type_name(member)
        return isinstance(written, Ok) and written.value in UNIT_HEADS

    def _walk(self, node: VibaNode) -> List[VibaNode]:
        """把这张图上能走到的坐标全走一遍（体检表用）。"""
        out = [node]
        positional = 0
        for tag, _ in self._members(node) or []:
            if tag:
                step = by_tag(tag)
            else:
                step = by_field_index(positional)
                positional += 1
            given = self.get(node, step)
            if isinstance(given, Ok) and given.value is not None:
                out += self._walk(given.value)
        shape = self._unfold(node.descriptor)
        container = self._container_kind(shape)
        if container == "dict":
            given_keys = self.keys(node)
            if isinstance(given_keys, Ok):
                for key in given_keys.value:
                    given = self.get(node, at_key(key))
                    if isinstance(given, Ok) and given.value is not None:
                        out += self._walk(given.value)
        elif self._has_elements_by_index(shape):
            length = self.length(node)
            if isinstance(length, Ok):
                for index in range(length.value):
                    given = self.get(node, at_index(index))
                    if isinstance(given, Ok) and given.value is not None:
                        out += self._walk(given.value)
        return out


# ----------------------------------------------------------------------
# 第 5.3 节的便利函数
# ----------------------------------------------------------------------


def viba_resolve(node: VibaNode, path: Sequence[VibaStep]) -> Result:
    """VibaResolve：逐个 step 走 VibaGet。

    走到"数据里没有这一段"（``Ok(nil)``）就到此为止：后面还有 step 就是
    ``Err("这一步没有值")``，没有 step 了就把 ``Ok(nil)`` 交出去。
    """
    current = node
    for step in path:
        if current is None:
            return Err("这一步没有值")
        given = node._access.get(current, step)
        if isinstance(given, Err):
            return given
        current = given.value
    return Ok(current)


def viba_get_by_path(node: VibaNode, path: Sequence[VibaStep]) -> Result:
    """VibaGetByPath：先 VibaResolve 再 VibaLeaf。"""
    resolved = viba_resolve(node, path)
    if isinstance(resolved, Err):
        return resolved
    if resolved.value is None:
        return Err("这一步没有值")
    return node._access.leaf(resolved.value)


def viba_list_fields(node: VibaNode, definition: VibaDefinitionDescriptor) -> Result:
    """VibaListFields：按 DefinitionMembers 逐个 VibaGet，取到的收进表。"""
    accessor = node._access
    members = definition_members(definition)
    if isinstance(members, Err):
        return members
    out = []
    positional = 0
    for member in members.value:
        if member.tag:
            step = by_tag(member.tag)
        else:
            # 描述符侧不认识规则标记，会把 RuleObject / OneofRule 记成一个位置
            # 成员；规则层这边认得它，跳过，也不让它占位置编号。
            if accessor._is_unit_member(member):
                continue
            step = by_field_index(positional)
            positional += 1
        given = accessor.get(node, step)
        # 取到的收进表；缺的（含这一段压根不是积/和，问不出来的）不进表。
        if isinstance(given, Ok) and given.value is not None:
            out.append(given.value)
    return Ok(out)


def access(definition: VibaDefinitionDescriptor) -> VibaAccess:
    """把一个定义（描述符）当图，返回访问器。

    协议里 VibaAccess[Data] 是实现方交付的那个类型；绑哪个定义由调用方给，
    地图就是描述符侧按这个定义建出来的那个定义描述符。
    """
    return VibaAccess(definition)


# ----------------------------------------------------------------------
# 私有小工具
# ----------------------------------------------------------------------


def _as_value(piece):
    """``...`` 不是数据：这一段里没有可取的数。"""
    if piece is None or isinstance(piece, viba_ast.Ellipsis):
        return None
    return piece


def _is_unit_descriptor(descriptor) -> bool:
    """链头的单位元：Object / nil / Oneof / never（名字或字面形态）。"""
    if descriptor.kind in (NIL, NEVER):
        return True
    return (descriptor.kind == TYPE_REF
            and descriptor.payload.type_name in UNIT_HEADS)


def _flatten(node) -> Optional[List]:
    """积/和/指数读成元素表：主链摊平，支链算一个元素。

    读法与规范化一致（链就是主链，元素可以是支链）；没规范化过的二元树
    也按同一个规矩读，两边给出的元素表一样。
    """
    if isinstance(node, (viba_ast.ProductChain, viba_ast.SumChain, viba_ast.ExponentChain)):
        return list(node.elements)
    if isinstance(node, (viba_ast.Product, viba_ast.Sum, viba_ast.Exponent)):
        head = node.left if isinstance(node, (viba_ast.Product, viba_ast.Sum)) else node.result
        tail = node.right if isinstance(node, (viba_ast.Product, viba_ast.Sum)) else node.argument
        elements = _flatten(head)
        if elements is None:
            elements = [head]
        return elements + [tail]
    return None


def _is_unit_data(node) -> bool:
    """链头的单位元：Object / nil / Oneof / never（名字或字面形态）。"""
    if isinstance(node, (viba_ast.Nil, viba_ast.Never)):
        return True
    return isinstance(node, viba_ast.TypeRef) and node.name in UNIT_HEADS


def _data_members(data) -> Optional[List[tuple]]:
    """数据这一层的成员：[(tag 或 None, 那一段), ...]，三种链一个规矩。"""
    elements = _flatten(data)
    if elements is None:
        if isinstance(data, viba_ast.Tagged):
            return [(data.tag, data.type)]
        return None
    if elements and _is_unit_data(elements[0]):
        elements = elements[1:]
    slots = []
    for element in elements:
        if isinstance(element, viba_ast.Tagged):
            slots.append((element.tag, element.type))
        else:
            slots.append((None, element))
    return slots


def _design_module(descriptor):
    """设计那一段属于哪个模块（数据里的名字按它解析）。"""
    resolvable = getattr(descriptor, "resolvable_type", None)
    return getattr(resolvable, "container_module", None)


def _definition_body(module, name):
    """模块里那个名字指到的普通定义体；不是普通定义就没有。"""
    resolved = module_get_type(module, name)
    if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
        return None
    node = resolved.value.ast_node
    return node.body if isinstance(node, viba_ast.TypeDefinition) else None


def _generic_definition(module, name):
    """模块里那个名字指到的泛型定义；不是泛型就没有。"""
    resolved = module_get_type(module, name)
    if isinstance(resolved, Err) or not isinstance(resolved.value, AstNodeType):
        return None
    node = resolved.value.ast_node
    return node if isinstance(node, viba_ast.GenericDefinition) else None


class _ParamFiller(viba_ast.NodeTransformer):
    """把定义体里写成形参的名字换成实参节点。"""

    def __init__(self, bindings):
        self.bindings = bindings

    def visit_TypeRef(self, node):
        return self.bindings.get(node.name, node)


def _fill_params(body, bindings):
    return _ParamFiller(bindings).visit(copy.deepcopy(body))


def _key_text(key) -> Optional[str]:
    if key is None:
        return None
    if isinstance(key, bool):
        return "true" if key else "false"
    return str(key)


def _constant_value(value) -> VibaConstantValue:
    if isinstance(value, bool):
        return VibaConstantValue("bool", value)
    if isinstance(value, int):
        return VibaConstantValue("int", value)
    if isinstance(value, float):
        return VibaConstantValue("float", value)
    if isinstance(value, str):
        return VibaConstantValue("str", value)
    if value is None:
        return VibaConstantValue("nil", None)
    raise TypeError(f"no constant kind for {value!r}")


# 协议里的名字（viba-reflect.md 第 4、5 节），只列这些。
__all__ = [
    "access",
    "VibaAccess", "VibaNode", "VibaStep", "VibaPath",
    "by_tag", "by_field_index", "at_index", "at_key",
    "viba_resolve", "viba_get_by_path", "viba_list_fields",
]

# 本层绑定的，按名字直接 import 用，不算协议概念：
#   Witness          协议里的 Data 形参在这里绑成什么
#   VibaReflectError 第 5.4 节"取值直接抛异常"的那个异常
