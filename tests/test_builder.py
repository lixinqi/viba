"""viba.builder 的验收：上层写法 → 源码 → 解析回同一份 AST。

    python tests/test_builder.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba import builder, viba_ast
from viba.viba_ast import GenericDefinition, SumChain, TypeDefinition

tag = builder.tag


def _sketch():
    """上层那种写法：带 import、泛型参数、tagged 字段与指数字段。"""
    vb = builder.Builder("import meeting_core as mc")

    vb.UserName = str
    vb.Optional[vb.T] = vb.T | None
    vb.List[vb.T] = (vb.Oneof
                     | vb.Object * tag.head(vb.T) * tag.tail(vb.List[vb.T])
                     | None)
    vb.latest_file[vb.Ctx] = (vb.mc.Result[vb.mc.FileState]
                              ** tag.ctx(vb.Ctx)
                              ** tag.file(vb.mc.FileId))
    return vb


def _check_sketch():
    source = str(_sketch())
    assert source == """import meeting_core as mc

UserName :=
  str

Optional[T] :=
  T
  | nil

List[T] :=
  Oneof
  | Object
    * $head T
    * $tail List[T]
  | nil

latest_file[Ctx] :=
  mc.Result[mc.FileState]
  <- $ctx Ctx
  <- $file mc.FileId""", source

    module = viba_ast.canonical(viba_ast.parse(source))
    assert isinstance(module.body[0], viba_ast.Import)
    assert module.body[0].module == "meeting_core" and module.body[0].alias == "mc"

    definitions = module.body[1:]
    assert [d.name for d in definitions] == ["UserName", "Optional", "List", "latest_file"]
    assert isinstance(definitions[0], TypeDefinition)
    assert isinstance(definitions[1], GenericDefinition)
    assert definitions[1].generic_params == ["T"]
    assert isinstance(definitions[2], GenericDefinition)
    assert definitions[2].generic_params == ["T"]
    assert isinstance(definitions[2].body, SumChain)
    # 指数字段：** 是右结合，写出来必须是平的 A <- B <- C，不是 A <- (B <- C)
    assert [e.__class__.__name__ for e in definitions[3].body.elements] == [
        "TypeApp", "Tagged", "Tagged"]
    assert [e.tag for e in definitions[3].body.elements[1:]] == ["$ctx", "$file"]


def _check_operators():
    vb = builder.Builder()
    vb.Alias = vb.Deep.Name
    vb.Branch = vb.A | (vb.B | vb.C)
    vb.Nested = vb.A * (vb.B | vb.C)
    vb.Arrow = vb.A ** tag.p(vb.B) ** tag.q(vb.C)
    vb.Held = vb.A ** tag(vb.B ** tag.p(vb.C))
    vb.Number = 1.5
    vb.Text = "x"
    vb.Flag = True
    vb.Open = ...
    vb.Ends = vb.never | vb.nil
    vb.Nothing = None
    vb.Block = builder.code("return 1")
    vb.Pair = (vb.A, vb.B)
    vb.One = (vb.B,)
    vb.Items = [1, "x", (vb.A, vb.B)]
    vb.Seen = {"c", "a", "b"}
    vb.Table = {"k": 1, "m": vb.V}
    vb.Applied = vb.some.deep.Name[vb.A]
    builder.add_import(vb, "meeting_core", "mc")

    source = str(vb)
    assert "Alias :=\n  Deep.Name" in source
    assert "Branch :=\n  A\n  | (B\n    | C)" in source
    assert "Nested :=\n  A\n  * (B\n    | C)" in source
    assert "Arrow :=\n  A\n  <- $p B\n  <- $q C" in source
    assert "Held :=\n  A\n  <- (B\n    <- $p C)" in source   # tag(...) 留住一个分组
    assert "Number :=\n  1.5" in source
    assert 'Text :=\n  "x"' in source
    assert "Flag :=\n  true" in source
    assert "Open :=\n  ..." in source
    assert "Ends :=\n  never\n  | nil" in source
    assert "Nothing :=\n  nil" in source
    assert "Block :=\n  {return 1}" in source
    assert "Pair :=\n  (A, B)" in source
    assert "One :=\n  (B,)" in source          # 一元组要留逗号，(B) 读回来只是 B
    assert 'Items :=\n  ListLiteral[1, "x", (A, B)]' in source
    assert 'Seen :=\n  SetLiteral["a", "b", "c"]' in source      # 集合无序，按写出来的顺序排
    assert 'Table :=\n  DictLiteral[("k", 1), ("m", V)]' in source
    assert "Applied :=\n  some.deep.Name[A]" in source
    assert source.startswith("import meeting_core as mc\n\n")

    # nil / never 是关键字：建出来的就得是单位元节点，不只是打印成 never / nil
    ends = [d for d in vb._body if getattr(d, "name", None) == "Ends"][0]
    assert [type(e).__name__ for e in ends.body.elements] == ["Never", "Nil"]
    return source


def _check_round_trip():
    for source in (str(_sketch()), _check_operators()):
        once = viba_ast.unparse(viba_ast.parse(source))
        twice = viba_ast.unparse(viba_ast.parse(once))
        assert once == source, f"不是规范写法：\n{source}\n!=\n{once}"
        assert once == twice, f"unparse 不稳定：\n{once}\n!=\n{twice}"


def _check_errors():
    vb = builder.Builder()
    for bad in (object(), b"bytes"):
        try:
            vb.Bad = bad
            raise AssertionError(f"{bad!r} 本该写不出来")
        except TypeError:
            pass
    try:
        vb.Bad = vb.A ** vb.B                   # ** 右边只能是 tag.xxx(.) 或 tag(.)
        raise AssertionError("本该抛")
    except TypeError:
        pass
    try:
        vb.Bad = vb.A ** tag.head                  # 忘了调用也不算
        raise AssertionError("本该抛")
    except TypeError:
        pass
    for bad in ("vb.true", "vb.false"):
        try:
            vb.Bad = eval(bad)                  # 关键字写错：True / False
            raise AssertionError("本该抛")
        except TypeError:
            pass
    for bad in ("nil", "never", "import", "a-b", "9x"):
        try:
            setattr(vb, bad, 1)                 # 定义名不能是关键字、不能不是名字
            raise AssertionError("本该抛")
        except TypeError:
            pass
    try:
        vb.a.b = 1                              # 定义名是一个名字，不是一个点分路径
        raise AssertionError("本该抛")
    except TypeError:
        pass
    try:
        vb.List[1] = vb.A                       # 泛型形参只能是名字
        raise AssertionError("本该抛")
    except TypeError:
        pass


def run():
    checks = [_check_sketch, _check_operators, _check_round_trip, _check_errors]
    for check in checks:
        check()
    print(f"builder: {len(checks)} checks passed "
          f"(上层写法、算子、规范写法回环、错误)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
