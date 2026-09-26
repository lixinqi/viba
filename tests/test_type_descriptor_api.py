"""Descriptor-side API checks.

The 100 generated cases in data/type_descriptor/case_* are the regression;
these checks are the readable ones: each is a small directory of
hand-written .viba files under data/type_descriptor/api plus the exact
assertions it is there to make.

    python tests/test_type_descriptor_api.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.type import VibaProgramErr, Ok, custom_module, module_get_type
from viba.viba_type_descriptor import (
    definition_file,
    definition_find_member_by_index,
    definition_find_member_by_tag,
    definition_members,
    empty_pool,
    file_find_import_by_local_name,
    member_containing_definition,
    member_resolved_definition,
    member_type_name,
    parse_viba_file,
    pool_add_file,
    pool_find_definition,
    pool_find_file,
    pool_find_member,
)

MATERIAL = Path(__file__).resolve().parent / "data" / "type_descriptor" / "api"


def load(check, modules):
    """modules: [(相对路径, 模块名), ...] 依次编进一个池子。"""
    pool = empty_pool()
    for rel_path, module_name in modules:
        source = (MATERIAL / check / rel_path).read_text()
        parsed = parse_viba_file(pool, source, rel_path, module_name)
        assert isinstance(parsed, Ok), (rel_path, parsed)
        added = pool_add_file(pool, parsed.ok_value)
        assert isinstance(added, Ok), (rel_path, added)
        pool = added.ok_value
    return pool


def _member(pool, definition_full_name, tag):
    definition = pool_find_definition(pool, definition_full_name).ok_value
    return definition_find_member_by_tag(definition, tag).ok_value


def _check_alias_and_depth():
    """深浅目录 + 两种 import 写法 + 前缀必须是 import 的本地名。"""
    pool = load("alias_and_depth", [
        ("top.viba", "top"),
        ("pkg/mod.viba", "pkg.mod"),
        ("pkg/sub/sub2/deepest.viba", "pkg.sub.sub2.deepest"),
    ])
    file = pool_find_file(pool, "pkg/sub/sub2/deepest.viba").ok_value
    assert [(i.module_name, i.local_name) for i in file.imports] == [
        ("top", "top"), ("pkg.mod", "mod")]

    deep = pool_find_definition(pool, "pkg.sub.sub2.deepest.Deep").ok_value
    assert definition_file(deep).ok_value.file_name == "pkg/sub/sub2/deepest.viba"

    # 别名进来的模块，解析到别的文件里的定义
    assert member_resolved_definition(_member(pool, deep.full_name, "$m")).ok_value.full_name == "pkg.mod.Mid"
    # 单段模块名：import top 绑的就是 top，top.Base 解析得到
    assert member_resolved_definition(_member(pool, deep.full_name, "$root")).ok_value.full_name == "top.Base"
    # mod 是别名，写成 pkg.mod.Mid 不是别名，解析不了
    assert isinstance(member_resolved_definition(_member(pool, deep.full_name, "$bad")), VibaProgramErr)
    assert isinstance(file_find_import_by_local_name(file, "nope"), VibaProgramErr)


def _check_import_binding():
    """import 绑的是什么：带 as 绑别名，不带 as 绑模块全名，最长的前缀优先。"""
    pool = load("import_binding", [
        ("top.viba", "top"),
        ("pkg/mod.viba", "pkg.mod"),
        ("pkg/mod/sub.viba", "pkg.mod.sub"),
        ("user.viba", "user"),
    ])
    file = pool_find_file(pool, "user.viba").ok_value
    assert [(i.module_name, i.local_name) for i in file.imports] == [
        ("pkg.mod", "pkg.mod"), ("pkg.mod.sub", "pkg.mod.sub"), ("top", "top")]

    user = pool_find_definition(pool, "user.User").ok_value
    # pkg.mod 与 pkg.mod.sub 同时绑着：最长的那个前缀赢
    assert member_resolved_definition(
        _member(pool, user.full_name, "$long")).ok_value.full_name == "pkg.mod.sub.Deep"
    assert member_resolved_definition(
        _member(pool, user.full_name, "$short")).ok_value.full_name == "pkg.mod.Mid"
    # 单段模块名照旧
    assert member_resolved_definition(
        _member(pool, user.full_name, "$root")).ok_value.full_name == "top.Base"
    # 没有 as，最后一段不算绑定：mod.Mid 与 sub.Deep 都解析不了
    assert isinstance(member_resolved_definition(_member(pool, user.full_name, "$old")), VibaProgramErr)
    assert isinstance(member_resolved_definition(_member(pool, user.full_name, "$first")), VibaProgramErr)
    # 模块路径的前缀也不是模块：pkg.Mid 解析不了
    assert isinstance(member_resolved_definition(_member(pool, user.full_name, "$partial")), VibaProgramErr)


def _check_members():
    """成员的类型：容器、可选、元组、内联、字面量、代码块、指数。"""
    pool = load("members", [("members.viba", "members")])
    definition = pool_find_definition(pool, "members.Members").ok_value
    members = definition_members(definition).ok_value
    assert [m.tag for m in members] == ["$items", "$seen", "$table", "$maybe",
                                        "$pair", "$inline", "$lit", "$code", "$fn",
                                        "$top"]
    assert [m.member_type.kind for m in members] == ["type_app", "type_app", "type_app",
                                                     "sum", "tuple", "sum",
                                                     "literal", "code_block", "exponent",
                                                     "any"]
    # 只有写成名字的成员才有 type_name；内联结构没有
    assert isinstance(member_type_name(members[0]), VibaProgramErr)
    assert isinstance(member_type_name(members[5]), VibaProgramErr)
    assert member_containing_definition(members[0]).ok_value.full_name == "members.Members"
    # 内建构造器不是定义，解析不了
    assert isinstance(member_resolved_definition(members[0]), VibaProgramErr)


def _check_generics():
    """泛型形参：形参名进 $generic_params，形参不是定义；单个带标签的体算一个成员。"""
    pool = load("generics", [("generics.viba", "generics")])
    box = pool_find_definition(pool, "generics.Box").ok_value
    pair = pool_find_definition(pool, "generics.Pair").ok_value
    assert box.generic_params == ["T"] and pair.generic_params == ["K", "V"]
    assert [m.tag for m in definition_members(box).ok_value] == ["$v"]
    assert [m.tag for m in definition_members(pair).ok_value] == ["$k", "$v"]

    use = pool_find_definition(pool, "generics.Use").ok_value
    assert member_type_name(_member(pool, use.full_name, "$param")).ok_value == "T"
    assert isinstance(member_resolved_definition(_member(pool, use.full_name, "$param")), VibaProgramErr)
    assert isinstance(member_resolved_definition(_member(pool, use.full_name, "$unknown")), VibaProgramErr)
    assert _member(pool, use.full_name, "$box").member_type.kind == "type_app"


def _check_unit_heads():
    """积/和链头的单位元不算成员；别名体没有成员。"""
    pool = load("unit_heads", [("heads.viba", "unit_heads")])
    rule = pool_find_definition(pool, "unit_heads.Rule").ok_value
    color = pool_find_definition(pool, "unit_heads.Color").ok_value
    bare = pool_find_definition(pool, "unit_heads.Bare").ok_value
    alias = pool_find_definition(pool, "unit_heads.Alias").ok_value

    assert [(m.member_index, m.tag) for m in rule.members] == [(0, "$a"), (1, None)]
    assert member_type_name(rule.members[1]).ok_value == "str"
    assert [(m.member_index, m.tag) for m in color.members] == [(0, "$red"), (1, "$blue")]
    assert [m.tag for m in bare.members] == ["$only"]      # 体是一整个带标签类型
    assert alias.members == []                              # 体是别名，没有成员
    assert isinstance(definition_find_member_by_index(rule, 2), VibaProgramErr)


def _check_by_need():
    """`CalledByNeed[T]`：标记说的是那一格怎么给，不是类型。

    所以这一层读穿它：一格写成 `CalledByNeed[int]`，读出来就是 `int`；带着标记的调用也编得出来。
    名字自己不算数——本地那份不含保留 tag 的定义就只是个普通的类型应用。
    """
    pool = load("by_need", [("case.viba", "by_need")])

    # 一份写着标记的文件编得出来：以前这一步是 PartialError
    called = pool_find_definition(pool, "by_need.Called").ok_value
    assert definition_members(called).ok_value == []       # 函数，不是带几个成员的积

    box = pool_find_definition(pool, "by_need.Box").ok_value
    assert member_type_name(definition_find_member_by_tag(box, "$t").ok_value).ok_value == "int"

    holder = pool_find_definition(pool, "by_need.Holder").ok_value
    member = definition_find_member_by_tag(holder, "$sw").ok_value
    assert member_resolved_definition(member).ok_value.full_name == "by_need.id_or_never"

    # 本地同名、不带保留 tag：`CalledByNeed[int]` 就是个普通的类型应用（一个积），
    # 不是 `int`。这条源故意编不过，所以写在这里而不是放进语料目录（那份语料一份都该是
    # 干净的）；判定层那边用两份能打开的文件钉了同一条规矩（test_is_sub_type.py）。
    shadowed = ("CalledByNeed[F] =\n"
                "    Object\n"
                "  * $func F\n"
                "X = int <- $env Environment <- $x CalledByNeed[int]\n"
                "Try = X << $env environ << $x 1 << $env environ\n")
    assert isinstance(parse_viba_file(empty_pool(), shadowed, "shadowed.viba", "shadowed"),
                      VibaProgramErr)


def _check_errors():
    """反例：重名文件、重名全名、模块名撞车、import 指向的模块不在池子里。"""
    pool = load("errors", [("amb_a.viba", "err.amb"), ("amb_b.viba", "err.amb")])
    # 同一个模块名由两份文件供着：环境答不了
    assert isinstance(pool.module_environment("err.amb"), VibaProgramErr)
    assert isinstance(pool.module_environment("err.missing"), VibaProgramErr)

    dup_pool = load("errors", [("dup1.viba", "err.dup")])
    parsed = parse_viba_file(dup_pool, (MATERIAL / "errors" / "dup2.viba").read_text(),
                             "dup2.viba", "err.dup")
    assert isinstance(parsed, Ok)
    assert isinstance(pool_add_file(dup_pool, parsed.ok_value), VibaProgramErr)      # 全名撞车
    assert isinstance(pool_add_file(dup_pool, dup_pool.files[0]), VibaProgramErr)  # 文件重名
    elsewhere = parse_viba_file(empty_pool(), "Z = int\n", "elsewhere.viba", "elsewhere")
    assert isinstance(pool_add_file(dup_pool, elsewhere.ok_value), VibaProgramErr)   # 别的池子建出来的

    orphan_pool = load("errors", [("orphan.viba", "err.orphan")])
    orphan = pool_find_definition(orphan_pool, "err.orphan.Orphan").ok_value
    member = _member(orphan_pool, orphan.full_name, "$p")
    assert member_type_name(member).ok_value == "g.Thing"
    assert isinstance(member_resolved_definition(member), VibaProgramErr)          # gone.mod 不在池子里
    assert isinstance(pool_find_member(orphan_pool, "err.orphan.Orphan.$nope"), VibaProgramErr)

    # 内建容器不是名字，是内建写法：等号左边出现就编不出来
    for bad in ("ListLiteral = int\n",
                "SetLiteral[T] = T\n",
                "DictLiteral[K] = K\n",
                "list = int\n",
                "set[T] = T\n",
                "dict[K, V] = K\n",
                "X[ListLiteral] = int\n",
                "X[K, SetLiteral] = K\n",
                "X[DictLiteral, K] = K\n",
                "X[list] = int\n",
                "X[K, set] = K\n",
                "X[dict, K] = K\n"):
        assert isinstance(parse_viba_file(empty_pool(), bad, "literal.viba", "literal"), VibaProgramErr), bad
    # 差一点的名字照旧能编：只认完全同名的六个
    for good in ("List = int\n",
                 "lists[T] = T\n",
                 "list2 = int\n",
                 "ListLiteral2 = int\n",
                 "mylist = int\n",
                 "Set = int\n",
                 "dicts[T] = T\n",
                 "X[List] = int\n",
                 "X[list2] = int\n",
                 "X[list_literal] = int\n",
                 "X[SetLiteral2] = int\n"):
        assert isinstance(parse_viba_file(empty_pool(), good, "near.viba", "near"), Ok), good
    # 出现在后面的定义里、或者第三个形参上，一样拦住
    for bad in ("A = int\nlist = int\n",
                "A = int\nX[B, C, set] = B\n",
                "X[dict] = int\n"):
        assert isinstance(parse_viba_file(empty_pool(), bad, "later.viba", "later"), VibaProgramErr), bad
    # 只在类型表达式里用的写法不受影响
    for good in ("Y = ListLiteral[1]\n",
                 "Y = set[dict[str, int]]\n",
                 "Y = $items list[int] * $more set[str]\n",
                 "Y = list\n",
                 "Y = list[]\n",
                 "import list\nY = int\n",          # 模块名可以叫 list
                 "X = Object * $list list[int]\n"):  # tag 可以叫 $list
        assert isinstance(parse_viba_file(empty_pool(), good, "uses.viba", "uses"), Ok), good
    # 空白的写法不影响判定
    for bad in ("list [ T ] = T\n",
                "list\n= int\n",
                "list = int  # note\n",
                "X[ list ] = int\n"):
        assert isinstance(parse_viba_file(empty_pool(), bad, "space.viba", "space"), VibaProgramErr), bad
    # import 一个叫 list 的模块不会把内建容器顶掉
    shadow = custom_module("import list\nZ = int\n")
    assert isinstance(module_get_type(shadow, "list"), Ok), "list still resolves"
    # `<<` 给的不是函数：设计写错，编的时候就给 VibaProgramErr（不是留到判定）
    for bad in ("P = (int * str) << $b str\n",
                "S = (int | str) << $b str\n",
                "L = 7 << $b str\n",
                "N = int << $b str\n",
                "M = (int <- $b str) << $c str\n"):
        assert isinstance(parse_viba_file(empty_pool(), bad, "partial.viba", "partial"), VibaProgramErr), bad
    # 给的那个参数得能坐进那一格：C <: B 才算合法
    assert isinstance(parse_viba_file(empty_pool(),
                                      "W = (int <- $b (int | str)) << $b int\n",
                                      "fit.viba", "fit"), Ok)
    for bad in ("N = (int <- $b int) << $b (int | str)\n",):
        assert isinstance(parse_viba_file(empty_pool(), bad, "unfit.viba", "unfit"), VibaProgramErr), bad
    for good in ("G = (int <- $b str) << $b str\n",
                 "H = (int <- $b str <- $c bool) << $c bool << $b str\n"):
        assert isinstance(parse_viba_file(empty_pool(), good, "partial_ok.viba", "partial_ok"), Ok), good
    # 已知的分歧：点分定义名现在能编（builder 那边 `vb.a.b = …` 是拦的）
    assert isinstance(parse_viba_file(empty_pool(), "a.b = int\n", "dotted.viba", "dotted"), Ok)
    # 绑定是计算的写法，不是类型的写法：定义体上出现它，这一层就编不出来
    for bad in ("A = (a := 7  a)\n",
                "X[T] = (T := int  T)\n"):
        assert isinstance(parse_viba_file(empty_pool(), bad, "binding.viba", "binding"),
                          VibaProgramErr), bad
    # 材料里也不跑表达式，所以材料里同样放不进去
    assert isinstance(parse_viba_file(empty_pool(), "X = $field (p := 7  p)\n",
                                      "material.viba", "material"), VibaProgramErr)
    # 不是模块的东西：问它要名字，说的是"不认识这种模块"，不是崩
    for not_a_module in ("not a module", None, 7):
        assert isinstance(module_get_type(not_a_module, "X"), VibaProgramErr), not_a_module


def _check_reprs():
    """描述符的 repr：调试和 diff 都要用，所以既说得出是什么，也不能带地址。"""
    pool = load("members", [("members.viba", "members")])
    assert repr(pool) == "VibaPool(1 files)"
    definition = pool_find_definition(pool, "members.Members").ok_value
    assert repr(definition) == "VibaDefinitionDescriptor('members.Members')"
    assert repr(definition.members[0]) == "VibaMemberDescriptor(0, '$items')"
    seen = {m.tag: repr(m.member_type) for m in definition.members}
    assert seen["$items"] == \
        "VibaTypeDescriptor(type_app, VibaTypeAppDescriptor('list', 1 args))"
    assert seen["$maybe"] == "VibaTypeDescriptor(sum, VibaChainDescriptor(2 elements))"
    assert seen["$pair"] == "VibaTypeDescriptor(tuple, VibaTupleDescriptor(2 elements))"
    assert seen["$lit"] == "VibaTypeDescriptor(literal, VibaLiteralDescriptor(42))"
    assert seen["$code"] == "VibaTypeDescriptor(code_block, VibaCodeBlockDescriptor(...))"
    assert seen["$top"] == "VibaTypeDescriptor(any, None)"
    binding = load("import_binding", [("user.viba", "user")])
    imports = pool_find_file(binding, "user.viba").ok_value.imports
    assert repr(imports[0]) == "VibaImportDescriptor('pkg.mod' as 'pkg.mod')"
    written = "".join([repr(pool), repr(definition)]
                      + [repr(m.member_type) for m in definition.members])
    assert "0x" not in written, written


def run():
    checks = [_check_alias_and_depth, _check_import_binding, _check_members,
              _check_generics, _check_unit_heads, _check_by_need,
              _check_errors, _check_reprs]
    for check in checks:
        check()
    print(f"type_descriptor_api: {len(checks)} checks passed "
          f"(imports and prefixes, import binding, members, generics, "
          f"unit chain heads, an argument read on demand, negatives, "
          f"descriptor reprs)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
