"""Descriptor-side regression: the 100 cases under data/type_descriptor.

Every case is a directory with .viba files at different depths (some at the
top, some three or four directories down) plus an ``expected.json``.  Each
file is parsed into one pool, and then the descriptor side has to report
exactly what the case says: imports, definitions, members, their written
names, their shapes, and which definition a written name resolves to.
"""

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from viba.type import Err, Ok
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

CASES = Path(__file__).resolve().parent / "data" / "type_descriptor"


def _normalize(source: str) -> str:
    return source.replace("\r\n", "\n").replace("\r", "\n")


def build_pool(case_dir: Path, expected: dict):
    """把 case 里的每个模块编进一个池子。"""
    pool = empty_pool()
    for module in expected["modules"]:
        source = (case_dir / module["file"]).read_text()
        parsed = parse_viba_file(pool, source, module["file"], module["module"])
        assert isinstance(parsed, Ok), (case_dir, module["module"], parsed)
        added = pool_add_file(pool, parsed.ok_value)
        assert isinstance(added, Ok), (case_dir, module["module"], added)
        pool = added.ok_value
    return pool


def check_case(case_dir: Path):
    expected = json.loads((case_dir / "expected.json").read_text())
    pool = build_pool(case_dir, expected)

    # 这一份 case 本身要有浅目录和深目录
    depths = {len(Path(module["file"]).parts) for module in expected["modules"]}
    assert 1 in depths and max(depths) >= 3, (case_dir, depths)

    checked_members = 0
    for module in expected["modules"]:
        source = (case_dir / module["file"]).read_text()
        found = pool_find_file(pool, module["file"])
        assert isinstance(found, Ok), (case_dir, module["file"], found)
        file = found.ok_value
        assert file.module_name == module["module"]
        assert file.file_hash == hashlib.sha256(_normalize(source).encode("utf-8")).hexdigest()
        assert [(i.module_name, i.local_name) for i in file.imports] == \
               [(i["module"], i["local"]) for i in module["imports"]], (case_dir, module["module"])
        assert [d.def_name for d in file.definitions] == \
               [d["name"] for d in module["definitions"]], (case_dir, module["module"])

        for wanted in module["definitions"]:
            found_def = pool_find_definition(pool, wanted["full_name"])
            assert isinstance(found_def, Ok), (case_dir, wanted["full_name"])
            definition = found_def.ok_value
            assert definition.generic_params == wanted["generic_params"]
            assert definition.body.pool is pool
            assert definition_file(definition).ok_value.file_name == module["file"]

            members = definition_members(definition).ok_value
            assert len(members) == len(wanted["members"]), (case_dir, wanted["full_name"])
            for truth, member in zip(wanted["members"], members):
                assert member.member_index == truth["index"]
                assert member.tag == truth["tag"], (case_dir, truth, member.tag)
                assert member.pool is pool
                assert member.member_type.kind == truth["kind"], (case_dir, truth)
                assert member_containing_definition(member).ok_value.full_name == wanted["full_name"]

                written = member_type_name(member)
                if truth["type_name"] is None:
                    assert isinstance(written, Err), (case_dir, truth, written)
                else:
                    assert isinstance(written, Ok) and written.ok_value == truth["type_name"], (case_dir, truth)

                resolved = member_resolved_definition(member)
                if truth["resolved"] is None:
                    assert isinstance(resolved, Err), (case_dir, truth, resolved)
                else:
                    assert isinstance(resolved, Ok) and resolved.ok_value.full_name == truth["resolved"], \
                        (case_dir, truth, resolved)

                assert definition_find_member_by_index(definition, member.member_index).ok_value is member
                if truth["tag"]:
                    assert definition_find_member_by_tag(definition, truth["tag"]).ok_value is member
                    full = f"{wanted['full_name']}.{truth['tag']}"
                    assert pool_find_member(pool, full).ok_value is member
                checked_members += 1

            assert isinstance(definition_find_member_by_tag(definition, "$nope"), Err)
            assert isinstance(definition_find_member_by_index(definition, len(members) + 5), Err)
            assert isinstance(definition_find_member_by_index(definition, -1), Err)

    # 池子上的反例
    first = expected["modules"][0]
    assert isinstance(pool_find_file(pool, "no/such/file.viba"), Err)
    assert isinstance(pool_find_definition(pool, "no.such.Definition"), Err)
    assert isinstance(pool_find_member(pool, "no.such.Definition.$x"), Err)
    assert isinstance(file_find_import_by_local_name(pool.file_name2file[first["file"]], "nope"), Err)
    assert isinstance(pool_add_file(pool, pool.files[0]), Err)  # 同一个文件加两次
    elsewhere = parse_viba_file(empty_pool(), "X := int\n", "elsewhere.viba", "elsewhere")
    assert isinstance(pool_add_file(pool, elsewhere.ok_value), Err)  # 别的池子建出来的

    return len(expected["modules"]), checked_members


def run():
    case_dirs = sorted(p for p in CASES.glob("case_*") if p.is_dir())
    assert len(case_dirs) == 100, f"expected 100 cases, found {len(case_dirs)}"

    # 文件名与模块名各由调用方给，两者不必同名
    pool = empty_pool()
    named = parse_viba_file(pool, "A := int\n", "deep/dir/x.viba", "some.module")
    assert isinstance(named, Ok)
    assert named.ok_value.file_name == "deep/dir/x.viba"
    assert named.ok_value.module_name == "some.module"

    files = members = 0
    for case_dir in case_dirs:
        case_files, case_members = check_case(case_dir)
        files += case_files
        members += case_members
    print(f"type_descriptor: {len(case_dirs)} cases, {files} files, "
          f"{members} members checked (imports, resolution, shapes, negatives)")
    return 0


if __name__ == "__main__":
    sys.exit(run())
