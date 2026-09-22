"""viba.builder 的验收：上层写法 → 源码 → 解析回同一份 AST。

    python tests/test_builder.py

一个细节一个函数：源码逐字比，结构看建出来的树（不是你打印出来的文本）。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import builder, viba_ast
from viba.viba_ast import GenericDefinition, Import, TypeDefinition

tag = builder.tag

_DEMO = (Path(__file__).resolve().parent / "data" / "builder" / "report.viba")


# ----------------------------------------------------------------------
# 小工具
# ----------------------------------------------------------------------


def _sketch():
    """上层那种写法：带 import、泛型参数、tagged 字段与指数字段。"""
    vb = builder.Builder("import store_core as sc")
    vb.UserName = str
    vb.Optional[vb.T] = vb.T | None
    vb.List[vb.T] = (vb.Oneof
                     | vb.Object * tag.head(vb.T) * tag.tail(vb.List[vb.T])
                     | None)
    vb.latest_file[vb.Ctx] = (vb.sc.Result[vb.sc.FileState]
                              ** tag.ctx(vb.Ctx)
                              ** tag.file(vb.sc.FileId))
    return vb


def _text(vb) -> str:
    """写出来的源码，去掉收尾那一个换行。"""
    return str(vb).rstrip("\n")


def _one(vb, name):
    """那份定义建出来的节点。"""
    for node in vb._body:
        if getattr(node, "name", None) == name:
            return node
    raise AssertionError(f"没有定义 {name!r}")


def _shape(node) -> str:
    """树里这一段的形状。"""
    return type(node).__name__


def _writes(expr, name: str = "X") -> str:
    """把一段表达式放进一份定义，返回它的写法。"""
    vb = builder.Builder()
    setattr(vb, name, expr)
    return _text(vb)


def _raises(kind, body) -> None:
    """这一段必须抛 kind。"""
    try:
        body()
        raise AssertionError(f"本该抛 {kind.__name__}")
    except kind:
        pass


# ----------------------------------------------------------------------
# 上层写法
# ----------------------------------------------------------------------


def test_builder_sketch_case_001():
    """整份文件逐字。"""
    assert str(_sketch()) == """import store_core as sc

UserName =
  str

Optional[T] =
  T
  | nil

List[T] =
  Oneof
  | Object
    * $head T
    * $tail List[T]
  | nil

latest_file[Ctx] =
  sc.Result[sc.FileState]
  <- $ctx Ctx
  <- $file sc.FileId
"""


def test_builder_sketch_case_002():
    """header 原样在最前，收尾只有一个换行。"""
    source = str(_sketch())
    assert source.startswith("import store_core as sc\n\n")
    assert source.endswith("\n") and not source.endswith("\n\n")


def test_builder_sketch_case_003():
    """定义的名字与顺序。"""
    module = viba_ast.parse(str(_sketch()))
    names = [d.name for d in module.body
             if isinstance(d, (TypeDefinition, GenericDefinition))]
    assert names == ["UserName", "Optional", "List", "latest_file"]


def test_builder_sketch_case_004():
    """`vb.UserName = str` 是普通定义，体是名字 str。"""
    node = _one(_sketch(), "UserName")
    assert isinstance(node, TypeDefinition)
    assert _shape(node.body) == "TypeRef" and node.body.name == "str"


def test_builder_sketch_case_005():
    """`vb.Optional[vb.T] = …` 是带形参的定义。"""
    node = _one(_sketch(), "Optional")
    assert isinstance(node, GenericDefinition) and node.generic_params == ["T"]


def test_builder_sketch_case_006():
    """泛型体里的 `vb.T | None` 是和链，nil 是关键字节点。"""
    node = _one(_sketch(), "Optional")
    assert _shape(node.body) == "SumChain"
    assert [_shape(e) for e in node.body.elements] == ["TypeRef", "Nil"]
    assert node.body.elements[0].name == "T"


def test_builder_sketch_case_007():
    """`vb.Object * tag.head(vb.T) * …` 是积链，元素按写的顺序。"""
    branch = _one(_sketch(), "List").body.elements[1]
    assert _shape(branch) == "ProductChain"
    assert [_shape(e) for e in branch.elements] == ["TypeRef", "Tagged", "Tagged"]
    assert [e.tag for e in branch.elements[1:]] == ["$head", "$tail"]


def test_builder_sketch_case_008():
    """`vb.List[vb.T]` 是真应用：constructor 与实参各就各位。"""
    tail = _one(_sketch(), "List").body.elements[1].elements[2]
    assert _shape(tail.type) == "TypeApp"
    assert tail.type.constructor == "List"
    assert [_shape(a) for a in tail.type.args] == ["TypeRef"]


def test_builder_sketch_case_009():
    """点分名字：`vb.sc.Result[vb.sc.FileState]`。"""
    result = _one(_sketch(), "latest_file").body.elements[0]
    assert _shape(result) == "TypeApp" and result.constructor == "sc.Result"
    assert result.args[0].name == "sc.FileState"


def test_builder_sketch_case_010():
    """指数字段是平的 A <- B <- C，不是 A <- (B <- C)。"""
    body = _one(_sketch(), "latest_file").body
    assert _shape(body) == "ExponentChain"
    assert [_shape(e) for e in body.elements] == ["TypeApp", "Tagged", "Tagged"]
    assert [e.tag for e in body.elements[1:]] == ["$ctx", "$file"]


def test_builder_sketch_case_011():
    """整份文件能被解析器读回来。"""
    module = viba_ast.parse(str(_sketch()))
    assert len(module.body) == 5
    assert isinstance(module.body[0], Import)
    assert module.body[0].module == "store_core" and module.body[0].alias == "sc"


# ----------------------------------------------------------------------
# 和 / 积 / 指数
# ----------------------------------------------------------------------


def test_builder_sum_case_001():
    """`|` 写和链，按写的顺序。"""
    vb = builder.Builder()
    vb.X = vb.A | vb.B | vb.C
    assert _text(vb) == "X =\n  A\n  | B\n  | C"


def test_builder_sum_case_002():
    """`A | (B | C)` 的支链留着，不压进主链。"""
    vb = builder.Builder()
    vb.X = vb.A | (vb.B | vb.C)
    assert [_shape(e) for e in _one(vb, "X").body.elements] == ["TypeRef", "SumChain"]


def test_builder_sum_case_003():
    """和链里能放字面量与单位元。"""
    vb = builder.Builder()
    vb.X = 1 | vb.A | None
    assert _text(vb) == "X =\n  1\n  | A\n  | nil"


def test_builder_product_case_001():
    """`*` 写积链。"""
    vb = builder.Builder()
    vb.X = vb.A * vb.B * vb.C
    assert _text(vb) == "X =\n  A\n  * B\n  * C"


def test_builder_product_case_002():
    """积里放和要带括号，读回来还是积。"""
    vb = builder.Builder()
    vb.X = vb.A * (vb.B | vb.C)
    assert _text(vb) == "X =\n  A\n  * (B\n    | C)"
    assert _shape(_one(vb, "X").body.elements[1]) == "SumChain"


def test_builder_product_case_003():
    """积里的 tagged 字段照写。"""
    vb = builder.Builder()
    vb.X = vb.Object * tag.a(vb.A) * tag.b(vb.B)
    assert _text(vb) == "X =\n  Object\n  * $a A\n  * $b B"


def test_builder_exponent_case_001():
    """`**` 写指数字段。"""
    assert _writes(builder.Builder().A ** tag.p(builder.Builder().B)) == \
        "X =\n  A\n  <- $p B"


def test_builder_exponent_case_002():
    """连着写：链平着长。"""
    vb = builder.Builder()
    vb.X = vb.A ** tag.p(vb.B) ** tag.q(vb.C)
    assert [_shape(e) for e in _one(vb, "X").body.elements] == ["TypeRef", "Tagged", "Tagged"]
    assert _text(vb) == "X =\n  A\n  <- $p B\n  <- $q C"


def test_builder_exponent_case_003():
    """`tag(…)` 留住一个分组：A <- (B <- $p C)。"""
    vb = builder.Builder()
    vb.X = vb.A ** tag(vb.B ** tag.p(vb.C))
    assert _text(vb) == "X =\n  A\n  <- (B\n    <- $p C)"
    assert _shape(_one(vb, "X").body.elements[1]) == "ExponentChain"


def test_builder_exponent_case_004():
    """括号不分组：`**` 右边是链就摊平，要分组得写 tag(…)。"""
    vb = builder.Builder()
    vb.X = vb.A ** (vb.B ** tag.p(vb.C))
    assert _text(vb) == "X =\n  A\n  <- B\n  <- $p C"


def test_builder_group_case_001():
    """`tag(…)` 在和里就是括号：A | (B | C)。"""
    vb = builder.Builder()
    vb.X = vb.A | tag(vb.B | vb.C)
    assert _text(vb) == "X =\n  A\n  | (B\n    | C)"
    assert [_shape(e) for e in _one(vb, "X").body.elements] == ["TypeRef", "SumChain"]


def test_builder_group_case_002():
    """`tag(…)` 在积里也是括号：A * (B | C)、A * (B * C)。"""
    vb = builder.Builder()
    vb.X = vb.A * tag(vb.B | vb.C)
    vb.Y = vb.A * tag(vb.B * vb.C)
    text = _text(vb)
    assert "X =\n  A\n  * (B\n    | C)" in text
    assert "Y =\n  A\n  * (B\n    * C)" in text


def test_builder_group_case_003():
    """`tag(…)` 裹一个名字跟没裹一样。"""
    vb = builder.Builder()
    vb.X = vb.A | tag(vb.B)
    assert _text(vb) == "X =\n  A\n  | B"


def test_builder_exponent_case_005():
    """`tag(name, body)` 与 `tag.name(body)` 一样。"""
    vb = builder.Builder()
    vb.X = vb.A ** tag("p", vb.B)
    assert _text(vb) == "X =\n  A\n  <- $p B"


def test_builder_exponent_case_006():
    """指数比积紧：`A * B ** tag.p(C)` 是 A * (B <- $p C)。"""
    vb = builder.Builder()
    vb.X = vb.A * vb.B ** tag.p(vb.C)
    assert _text(vb) == "X =\n  A\n  * B\n    <- $p C"
    assert _shape(_one(vb, "X").body.elements[1]) == "ExponentChain"


def test_builder_exponent_case_007():
    """指数链的结果可以是应用。"""
    vb = builder.Builder()
    vb.X = vb.F[vb.A] ** tag.p(vb.B)
    assert _text(vb) == "X =\n  F[A]\n  <- $p B"


# ----------------------------------------------------------------------
# 单位元 / 字面量
# ----------------------------------------------------------------------


def test_builder_unit_case_001():
    """`None` 是 nil。"""
    assert _writes(None) == "X =\n  nil"


def test_builder_unit_case_002():
    """`vb.nil` 建出来的就是 Nil 节点，不只是打印成 nil。"""
    vb = builder.Builder()
    vb.X = vb.nil
    assert _shape(_one(vb, "X").body) == "Nil"


def test_builder_unit_case_003():
    """`vb.void` 是 nil 的别名。"""
    vb = builder.Builder()
    vb.X = vb.void
    assert _shape(_one(vb, "X").body) == "Nil"


def test_builder_unit_case_004():
    """`vb.never` 建出来的是 Never 节点。"""
    vb = builder.Builder()
    vb.X = vb.never
    assert _shape(_one(vb, "X").body) == "Never"


def test_builder_unit_case_005():
    """和里同时放 never 与 nil，树和文本都对。"""
    vb = builder.Builder()
    vb.X = vb.never | vb.nil
    assert [_shape(e) for e in _one(vb, "X").body.elements] == ["Never", "Nil"]
    assert _text(vb) == "X =\n  never\n  | nil"


def test_builder_unit_case_006():
    """`Object` / `Oneof` 是名字，不是关键字。"""
    vb = builder.Builder()
    vb.X = vb.Object | vb.Oneof
    assert [_shape(e) for e in _one(vb, "X").body.elements] == ["TypeRef", "TypeRef"]


def test_builder_unit_case_007():
    """`...` 是省略号。"""
    assert _writes(...) == "X =\n  ..."
    vb = builder.Builder()
    vb.X = ...
    assert _shape(_one(vb, "X").body) == "Ellipsis"


def test_builder_literal_case_001():
    """`True` / `False` 是 true / false，不掉成 1 / 0。"""
    vb = builder.Builder()
    vb.T = True
    vb.F = False
    assert _text(vb) == "T =\n  true\n\nF =\n  false"
    assert _one(vb, "T").body.value is True and _one(vb, "F").body.value is False


def test_builder_literal_case_002():
    """整数与浮点。"""
    vb = builder.Builder()
    vb.I = 42
    vb.F = 1.5
    assert _text(vb) == "I =\n  42\n\nF =\n  1.5"


def test_builder_literal_case_003():
    """字符串字面量。"""
    assert _writes("x") == 'X =\n  "x"'


def test_builder_literal_case_004():
    """`str` 这个类型（不是字符串实例）是名字。"""
    vb = builder.Builder()
    vb.X = str
    assert _shape(_one(vb, "X").body) == "TypeRef" and _one(vb, "X").body.name == "str"


def test_builder_literal_case_005():
    """内置类型名：int / float / bool / list / set / dict。"""
    vb = builder.Builder()
    vb.X = vb.int * vb.float * vb.bool * vb.list * vb.set * vb.dict
    assert _text(vb) == "X =\n  int\n  * float\n  * bool\n  * list\n  * set\n  * dict"


def test_builder_literal_case_008():
    """最左边是个普通 Python 值时，用 `builder.literal` 起头。"""
    vb = builder.Builder()
    vb.X = builder.literal("a") | 1
    assert _text(vb) == 'X =\n  "a"\n  | 1'


def test_builder_literal_case_009():
    """`literal` 起头的积、以及 nil 起头。"""
    vb = builder.Builder()
    vb.X = builder.literal(1) * "x"
    vb.Y = builder.literal(None) | vb.A
    assert _text(vb) == 'X =\n  1\n  * "x"\n\nY =\n  nil\n  | A'


def test_builder_literal_case_007():
    """Python 自己的 `int | str` 也是和（NoneType 就是 nil）。"""
    vb = builder.Builder()
    vb.X = int | str
    vb.Y = int | None
    assert _text(vb) == "X =\n  int\n  | str\n\nY =\n  int\n  | nil"


def test_builder_literal_case_006():
    """字面量也能进和链（最左边那一个得是 builder 出来的，Python 才认）。"""
    vb = builder.Builder()
    vb.X = vb.A | "a" | 1 | None
    assert _text(vb) == 'X =\n  A\n  | "a"\n  | 1\n  | nil'


# ----------------------------------------------------------------------
# 名字 / 应用 / 定义
# ----------------------------------------------------------------------


def test_builder_name_case_001():
    """点分名字。"""
    assert _writes(builder.Builder().some.deep.Name) == "X =\n  some.deep.Name"


def test_builder_name_case_002():
    """应用：constructor[实参]。"""
    vb = builder.Builder()
    vb.X = vb.F[vb.A]
    assert _shape(_one(vb, "X").body) == "TypeApp"
    assert _one(vb, "X").body.constructor == "F"


def test_builder_name_case_003():
    """多实参，逗号分开。"""
    vb = builder.Builder()
    vb.X = vb.Map[vb.K, vb.V]
    assert _text(vb) == "X =\n  Map[K, V]"


def test_builder_name_case_004():
    """空实参应用 `Name[]`：Python 没有空下标，写 `vb.F[()]`。"""
    vb = builder.Builder()
    vb.X = vb.F[()]
    assert _text(vb) == "X =\n  F[]"
    assert _shape(_one(vb, "X").body) == "TypeApp" and _one(vb, "X").body.args == []


def test_builder_name_case_005():
    """裸名与应用是两回事：`F` 是引用，`F[]` 是应用。"""
    vb = builder.Builder()
    vb.Bare = vb.F
    vb.Empty = vb.F[()]
    assert [_shape(_one(vb, n).body) for n in ("Bare", "Empty")] == ["TypeRef", "TypeApp"]


def test_builder_name_case_006():
    """实参可以是点分名字、应用、字面量、tagged。"""
    vb = builder.Builder()
    vb.X = vb.F[vb.a.b, vb.G[vb.C], 1, tag.p(vb.D)]
    assert _text(vb) == "X =\n  F[a.b, G[C], 1, $p D]"


def test_builder_applied_case_001():
    """Python 的下标类型 `list[vb.A]` 就是 `list[A]`。"""
    vb = builder.Builder()
    vb.X = list[vb.A]
    assert _text(vb) == "X =\n  list[A]"
    assert _one(vb, "X").body.constructor == "list"


def test_builder_applied_case_002():
    """三种容器都认：`set[vb.A]` / `list[vb.A]` / `dict[str, int]`。"""
    vb = builder.Builder()
    vb.S = set[vb.A]
    vb.L = list[vb.A]
    vb.D = dict[str, int]
    assert _text(vb) == "S =\n  set[A]\n\nL =\n  list[A]\n\nD =\n  dict[str, int]"


def test_builder_applied_case_003():
    """下标类型也能进和：`list[int] | None`。"""
    vb = builder.Builder()
    vb.X = list[int] | None
    assert _text(vb) == "X =\n  list[int]\n  | nil"


def test_builder_applied_case_004():
    """`typing.List[vb.A]` 落到语言的写法 `list[A]`。"""
    import typing

    assert _writes(typing.List[int]) == "X =\n  list[int]"


def test_builder_applied_case_005():
    """`typing.Optional[vb.A]` 是和：`A | nil`。"""
    import typing

    assert _writes(typing.Optional[int]) == "X =\n  int\n  | nil"


def test_builder_definition_case_001():
    """`vb.Name = body` 是普通定义。"""
    vb = builder.Builder()
    vb.X = vb.A
    assert isinstance(_one(vb, "X"), TypeDefinition)


def test_builder_definition_case_002():
    """`vb.Name[P] = body` 是带一个形参的定义。"""
    vb = builder.Builder()
    vb.X[vb.T] = vb.T
    assert isinstance(_one(vb, "X"), GenericDefinition)
    assert _one(vb, "X").generic_params == ["T"]


def test_builder_definition_case_003():
    """两个形参。"""
    vb = builder.Builder()
    vb.Map[vb.K, vb.V] = vb.V
    assert _one(vb, "Map").generic_params == ["K", "V"]
    assert _text(vb) == "Map[K, V] =\n  V"


def test_builder_definition_case_004():
    """同名定义各写各的，顺序就是写的顺序。"""
    vb = builder.Builder()
    vb.X = vb.A
    vb.X = vb.B
    assert [d.name for d in vb._body] == ["X", "X"]


# ----------------------------------------------------------------------
# 容器字面量
# ----------------------------------------------------------------------


def test_builder_list_case_001():
    """`[a, b]` 是 ListLiteral，顺序照写。"""
    assert _writes([1, "x"]) == 'X =\n  ListLiteral[1, "x"]'


def test_builder_list_case_002():
    """空列表写成 `ListLiteral[]`。"""
    assert _writes([]) == "X =\n  ListLiteral[]"


def test_builder_list_case_003():
    """列表元素可以是表达式与元组。"""
    vb = builder.Builder()
    vb.X = [vb.A, (vb.B, vb.C)]
    assert _text(vb) == "X =\n  ListLiteral[A, (B, C)]"


def test_builder_set_case_001():
    """`{a, b}` 是 SetLiteral。"""
    assert _writes({"a", "b"}) == 'X =\n  SetLiteral["a", "b"]'


def test_builder_set_case_002():
    """集合无序：按写出来的文本排序，同一个集合每次一样。"""
    assert _writes({"c", "a", "b"}) == _writes({"a", "b", "c"}) == \
        'X =\n  SetLiteral["a", "b", "c"]'


def test_builder_set_case_003():
    """集合元素是表达式也照排。"""
    vb = builder.Builder()
    vb.X = {vb.B, vb.A}
    assert _text(vb) == "X =\n  SetLiteral[A, B]"


def test_builder_dict_case_001():
    """`{k: v}` 是 DictLiteral，键值成对，顺序照写。"""
    assert _writes({"k": 1, "m": 2}) == 'X =\n  DictLiteral[("k", 1), ("m", 2)]'


def test_builder_dict_case_002():
    """字典的值可以是表达式。"""
    vb = builder.Builder()
    vb.X = {"k": vb.A}
    assert _text(vb) == 'X =\n  DictLiteral[("k", A)]'


def test_builder_dict_case_003():
    """空字典写成 `DictLiteral[]`。"""
    assert _writes({}) == "X =\n  DictLiteral[]"


def test_builder_container_case_001():
    """容器也能套容器。"""
    vb = builder.Builder()
    vb.X = [[1], {"k": [2]}]
    assert _text(vb) == ('X =\n  ListLiteral[ListLiteral[1], '
                         'DictLiteral[("k", ListLiteral[2])]]')


def test_builder_container_case_002():
    """容器字面量读回来还是字面量。"""
    body = viba_ast.parse(_writes([1, 2]) + "\n").body[0].body
    assert _shape(body) == "TypeApp" and body.constructor == "ListLiteral"


# ----------------------------------------------------------------------
# 元组 / 代码块
# ----------------------------------------------------------------------


def test_builder_tuple_case_001():
    """二元组。"""
    vb = builder.Builder()
    vb.X = (vb.A, vb.B)
    assert _text(vb) == "X =\n  (A, B)" and _shape(_one(vb, "X").body) == "Tuple"


def test_builder_tuple_case_002():
    """一元组要留逗号，`(B)` 读回来只是 B。"""
    vb = builder.Builder()
    vb.X = (vb.B,)
    assert _text(vb) == "X =\n  (B,)"
    assert len(viba_ast.parse(_text(vb) + "\n").body[0].body.elements) == 1


def test_builder_tuple_case_003():
    """空元组。"""
    vb = builder.Builder()
    vb.X = ()
    assert _text(vb) == "X =\n  ()" and _one(vb, "X").body.elements == []


def test_builder_code_case_001():
    """`code(text)` 写 `{...}`。"""
    vb = builder.Builder()
    vb.X = builder.code("return 1")
    assert _text(vb) == "X =\n  {return 1}"
    assert _shape(_one(vb, "X").body) == "CodeBlock"


def test_builder_code_case_002():
    """代码块能进应用（谓词那种写法）。"""
    vb = builder.Builder()
    vb.X = vb.Checked[builder.code("def check(self):\n    return 1")]
    assert str(vb).startswith("X =\n  Checked[{def check(self):")


# ----------------------------------------------------------------------
# import / comment / check / 输出
# ----------------------------------------------------------------------


def test_builder_import_case_001():
    """`add_import` 写在最前，哪怕定义先写。"""
    vb = builder.Builder()
    vb.X = vb.A
    builder.add_import(vb, "store_core", "sc")
    assert _text(vb) == "import store_core as sc\n\nX =\n  A"


def test_builder_import_case_002():
    """没有别名就没有 as。"""
    vb = builder.Builder()
    builder.add_import(vb, "numpy")
    assert _text(vb) == "import numpy"


def test_builder_comment_case_001():
    """注释写在它站的位置。"""
    vb = builder.Builder()
    vb.A = 1
    builder.comment(vb, "说明")
    vb.B = 2
    assert _text(vb) == "A =\n  1\n\n# 说明\n\nB =\n  2"


def test_builder_comment_case_002():
    """空注释只留一个井号。"""
    vb = builder.Builder()
    builder.comment(vb, "")
    assert _text(vb) == "#"


def test_builder_comment_case_003():
    """注释能在文件最前面。"""
    vb = builder.Builder()
    builder.comment(vb, "文件说明")
    vb.A = 1
    assert _text(vb) == "# 文件说明\n\nA =\n  1"


def test_builder_check_case_001():
    """`check` 读回来的就是写出去的。"""
    assert len(builder.check(_sketch()).body) == 5


def test_builder_check_case_002():
    """`check` 出的是 Module。"""
    vb = builder.Builder()
    vb.A = 1
    assert _shape(builder.check(vb)) == "Module"


def test_builder_output_case_001():
    """空 builder 写出来是空串。"""
    assert str(builder.Builder()) == ""


def test_builder_output_case_002():
    """只有起点没有新定义时，原样输出加一个换行。"""
    assert str(builder.Builder("import a.b as c")) == "import a.b as c\n"


def test_builder_output_case_003():
    """收尾只有一个换行。"""
    vb = builder.Builder()
    vb.A = 1
    assert str(vb) == "A =\n  1\n"


# ----------------------------------------------------------------------
# 续写既有文件：只能追加新定义
# ----------------------------------------------------------------------


def test_builder_append_case_001():
    """既有内容原样在最前。"""
    vb = builder.Builder(_DEMO.read_text())
    vb.Added = vb.Object
    assert str(vb).startswith(_DEMO.read_text().strip())


def test_builder_append_case_002():
    """新定义追加在最后。"""
    vb = builder.Builder(_DEMO.read_text())
    vb.Added = vb.Object * tag.x(vb.T)
    assert str(vb).endswith("Added =\n  Object\n  * $x T\n")


def test_builder_append_case_003():
    """读回来：旧的都在，新的也在。"""
    vb = builder.Builder(_DEMO.read_text())
    vb.Added = vb.Object
    names = [d.name for d in builder.check(vb).body
             if isinstance(d, (TypeDefinition, GenericDefinition))]
    assert names == ["CodeLength", "DocCoverage", "Keywords", "Report", "Added"]


def test_builder_append_case_004():
    """注释也能追加。"""
    vb = builder.Builder(_DEMO.read_text())
    builder.comment(vb, "加上一条")
    assert str(vb).endswith("# 加上一条\n")


def test_builder_append_case_005():
    """既有文件里的名字不能再定义一次。"""
    vb = builder.Builder(_DEMO.read_text())
    _raises(TypeError, lambda: setattr(vb, "Report", 1))
    _raises(TypeError, lambda: setattr(vb, "CodeLength", 1))


def test_builder_append_case_006():
    """带形参的重定义也拦。"""
    vb = builder.Builder("X[T] = T\n")
    _raises(TypeError, lambda: vb.X.__setitem__(vb.T, vb.T))


def test_builder_append_case_007():
    """续写既有文件时插不进 import（它得站在最上面）。"""
    vb = builder.Builder(_DEMO.read_text())
    _raises(TypeError, lambda: builder.add_import(vb, "x"))


def test_builder_append_case_008():
    """起点本身得是 Viba 源码。"""
    _raises(ValueError, lambda: builder.Builder("X = ("))


def test_builder_append_case_009():
    """起点只有注释也算合法源码。"""
    vb = builder.Builder("# 只是一句话")
    vb.A = 1
    assert _text(vb).endswith("A =\n  1")


# ----------------------------------------------------------------------
# 写错的地方要当场拦下
# ----------------------------------------------------------------------


def test_builder_guard_case_001():
    """`**` 右边是名字：拦。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: setattr(vb, "X", vb.A ** vb.B))


def test_builder_guard_case_002():
    """忘了调用 tag：不管在 `**` 右边还是在积里，都拦。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: setattr(vb, "X", vb.A ** tag.head))
    _raises(TypeError, lambda: setattr(vb, "X", vb.Object * tag.float_value * float))


def test_builder_guard_case_013():
    """忘了调用 tag 时，报出来的话点出是哪个 tag。"""
    vb = builder.Builder()
    try:
        vb.X = vb.Object * tag.float_value
        raise AssertionError("本该抛")
    except TypeError as error:
        assert "tag.float_value" in str(error), error


def test_builder_guard_case_003():
    """`**` 右边是和或积：拦。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: setattr(vb, "X", vb.A ** (vb.B | vb.C)))
    _raises(TypeError, lambda: setattr(vb, "X", vb.A ** (vb.B * vb.C)))


def test_builder_guard_case_004():
    """`**` 右边是字面量：拦。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: setattr(vb, "X", vb.A ** 1))


def test_builder_guard_case_005():
    """`vb.true` / `vb.false` 是写错，Python 里该写 True / False。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: setattr(vb, "X", vb.true))
    _raises(TypeError, lambda: setattr(vb, "X", vb.false))


def test_builder_guard_case_006():
    """定义名不能是关键字。"""
    for name in ("nil", "never", "true", "false", "void", "import", "as"):
        vb = builder.Builder()
        _raises(TypeError, lambda name=name, vb=vb: setattr(vb, name, 1))


def test_builder_guard_case_007():
    """定义名得是个标识符。"""
    for name in ("a-b", "9x", "", "a b"):
        vb = builder.Builder()
        _raises(TypeError, lambda name=name, vb=vb: setattr(vb, name, 1))


def test_builder_guard_case_008():
    """`vb.a.b = …` 是写错：定义名是一个名字。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: setattr(vb.a, "b", 1))


def test_builder_guard_case_009():
    """泛型形参只能是名字。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: vb.X.__setitem__(1, vb.A))


def test_builder_guard_case_010():
    """泛型形参不能重复。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: vb.X.__setitem__((vb.T, vb.T), vb.A))


def test_builder_guard_case_011():
    """写不出来的 Python 值：拦。"""
    for bad in (object(), b"bytes", range(3)):
        vb = builder.Builder()
        _raises(TypeError, lambda bad=bad, vb=vb: setattr(vb, "X", bad))


def test_builder_guard_case_014():
    """内建容器名不能当定义名，也不能当泛型形参（语法层面就不许）。"""
    for name in ("list", "set", "dict",
                 "ListLiteral", "SetLiteral", "DictLiteral"):
        vb = builder.Builder()
        _raises(TypeError, lambda name=name, vb=vb: setattr(vb, name, 1))
    vb = builder.Builder()
    _raises(TypeError, lambda: vb.X.__setitem__(vb.ListLiteral, vb.A))
    _raises(TypeError, lambda: vb.X.__setitem__((vb.K, vb.SetLiteral), vb.A))
    _raises(TypeError, lambda: vb.X.__setitem__(vb.list, vb.A))
    _raises(TypeError, lambda: vb.X.__setitem__((vb.K, vb.dict), vb.A))


def test_builder_guard_case_015():
    """差一点的名字照旧能当定义名与形参。"""
    vb = builder.Builder()
    vb.List = 1
    vb.list2 = 2
    vb.X[vb.list2] = vb.A
    assert _text(vb) == "List =\n  1\n\nlist2 =\n  2\n\nX[list2] =\n  A"


def test_builder_guard_case_012():
    """`add_import` 的返回值是 builder，不是表达式，套进定义里也会拦。"""
    vb = builder.Builder()
    _raises(TypeError, lambda: setattr(vb, "X", builder.add_import(vb, "a")))


# ----------------------------------------------------------------------
# 规范写法
# ----------------------------------------------------------------------


def test_builder_canonical_case_001():
    """上层那份：写出来就是打印器的规范形式。"""
    source = str(_sketch())
    assert viba_ast.unparse(viba_ast.parse(source)) == source.rstrip("\n")


def test_builder_canonical_case_002():
    """算子那份也是。"""
    vb = builder.Builder()
    vb.X = vb.A | (vb.B | vb.C)
    vb.Y = vb.A * (vb.B | vb.C)
    vb.Z = vb.A ** (vb.B ** tag.p(vb.C))
    source = str(vb)
    assert viba_ast.unparse(viba_ast.parse(source)) == source.rstrip("\n")


def test_builder_canonical_case_003():
    """打印→解析→打印 稳定。"""
    once = viba_ast.unparse(viba_ast.parse(str(_sketch())))
    twice = viba_ast.unparse(viba_ast.parse(once))
    assert once == twice


def test_builder_canonical_case_004():
    """续写出来的整份文件也能读回来。"""
    vb = builder.Builder(_DEMO.read_text())
    vb.Added = vb.Object * tag.x(vb.T)
    assert _shape(builder.check(vb)) == "Module"


# ----------------------------------------------------------------------
# 上界 Any
# ----------------------------------------------------------------------


def test_builder_any_case_001():
    """`vb.Any` 写出来是关键字 Any，读回来还是它。"""
    vb = builder.Builder()
    vb.Top = vb.Any
    vb.Maybe = vb.Top | None
    assert str(vb).rstrip("\n") == "Top =\n  Any\n\nMaybe =\n  Top\n  | nil"
    node = _one(vb, "Top")
    assert _shape(node.body) == "Any"
    assert viba_ast.unparse(viba_ast.parse(str(vb))) == str(vb).rstrip("\n")


# ----------------------------------------------------------------------
# 部分计算 `<<`
# ----------------------------------------------------------------------


def test_builder_partial_case_001():
    """`A << tag.b(B)` 写成 `A << $b B`，体是 Partial。"""
    vb = builder.Builder()
    vb.A = int
    vb.B = str
    vb.Given = vb.A << tag.b(vb.B)
    assert str(vb).rstrip("\n") == "A =\n  int\n\nB =\n  str\n\nGiven =\n  A << $b B"


def test_builder_partial_case_002():
    """给掉一个参数：剩下的链就是新体；写出来读回来还是同一份。"""
    vb = builder.Builder()
    vb.A = int
    vb.B = str
    vb.C = bool
    chain = vb.A ** tag.b(vb.B) ** tag.c(vb.C)
    vb.Given = chain << tag.b(vb.B)
    vb.All = chain << tag.c(vb.C) << tag.b(vb.B)
    assert _shape(_one(vb, "Given").body) == "Partial"
    assert _shape(_one(vb, "All").body) == "Partial"
    assert viba_ast.unparse(viba_ast.parse(str(vb))) == str(vb).rstrip("\n")


def test_builder_partial_case_003():
    """右参数必须是 tag 或分组：与 `**` 同一条规矩。"""
    vb = builder.Builder()
    vb.A = int
    vb.B = str
    _raises(TypeError, lambda: vb.A << vb.B)
    _raises(TypeError, lambda: vb.A << vb.B * vb.A)


def test_builder_partial_case_004():
    """`tag(body)`（分组）也收，写出来是括号；给完的判定交给判断层。"""
    vb = builder.Builder()
    vb.A = int
    vb.B = str
    vb.C = bool
    vb.Given = (vb.A ** tag.b(vb.B)) << tag(vb.C)
    module = viba_ast.parse(str(vb))
    assert any(getattr(d, "name", None) == "Given" for d in module.body)


def test_builder_private_names_case_001():
    """下划线开头的东西不是 tag、不是字段、不是定义位：AttributeError。"""
    vb = builder.Builder()
    _raises(AttributeError, lambda: vb.tag._private)
    _raises(AttributeError, lambda: vb._private)
    _raises(AttributeError, lambda: vb.Some._private)


def test_builder_tag_arity_case_001():
    """tag 只收一个参数（分组）或两个（字段）。"""
    _raises(TypeError, lambda: tag(1, 2, 3))


def test_builder_name_slot_case_001():
    """一个光名字不是定义位：泛型下标与应用的头部都要说得出为什么不行。"""
    _raises(TypeError, lambda: builder._Name("A").__setitem__(7, 1))
    _raises(TypeError, lambda: builder._TypeApp(7, []))


def test_builder_bad_source_case_001():
    """写出来的源编不过：check 把它说成 ValueError，而不是把 SyntaxError 放出来。
    名字不是标识符时落在定义名那一关（TypeError），根本写不进源里。"""
    _raises(TypeError, lambda: setattr(builder.Builder(), "a b", int))
    vb = builder.add_import(builder.Builder(), "a..b")
    _raises(ValueError, lambda: builder.check(vb))


def test_builder_origin_name_case_001():
    """一个没有名字的东西当泛型的头：说得出它没有名字可写。"""
    _raises(TypeError, lambda: builder._origin_name(object()))


def test_builder_comment_case_001():
    """注释行不是节点：下划线开头的名字照挡，挂东西仍可挂。"""
    vb = builder.comment(builder.Builder(), "note")
    note = vb._body[-1]
    _raises(AttributeError, lambda: note._private)
    note._marker = 1
    assert note._marker == 1


# ----------------------------------------------------------------------
# 跑
# ----------------------------------------------------------------------


def run():
    cases = [value for name, value in globals().items()
             if name.startswith("test_builder_") and callable(value)]
    for case in cases:
        case()
    print(f"builder: {len(cases)} cases passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
