"""Build 100 multi-file cases for tests/test_type_descriptor.py.

Each case is a directory holding a few .viba files at different depths
(some at the top, some three or four directories down) plus an
``expected.json`` that says what the descriptor side must report about
them.  The files under test live here, one directory per case; the test
only reads them.

    python tests/data/type_descriptor/build.py          # write the cases
    python tests/data/type_descriptor/build.py --check   # fail if stale
"""

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASE_COUNT = 100

# Paths are picked from here; depth 0 files are shallow, the rest are deep.
CANDIDATES = [
    "top.viba",
    "util.viba",
    "pkg/mod.viba",
    "pkg/sub/deep.viba",
    "pkg/sub/sub2/deepest.viba",
    "lv1/lv2/lv3/leaf.viba",
]


def module_name_of(rel_path: str) -> str:
    return rel_path[: -len(".viba")].replace("/", ".")


def local_name_of(module: str, aliased: bool, rng) -> str:
    """What the import binds: an alias, or the whole module name when the
    import carries none (references then read a.b.Name)."""
    if aliased:
        return rng.choice(["m", "dep", "base", "up", "lib"])
    return module


# ----------------------------------------------------------------------
# 定义模板：每个返回 (源码行, 成员真值, 泛型形参, 需要的 import)
# ----------------------------------------------------------------------


def member(index, tag, type_name, kind, optional=False, resolved=None):
    return {
        "index": index,
        "tag": tag,
        "type_name": type_name,
        "kind": kind,
        "optional": optional,
        "resolved": resolved,
    }


def t_product_tagged(rng, name):
    lines = [f"{name} :=", "  Object", "  * $count int", "  * $label str"]
    truth = [member(0, "$count", "int", "type_ref"), member(1, "$label", "str", "type_ref")]
    return lines, truth, [], None


def t_product_positional(rng, name):
    written = rng.choice(["int * str * bool", "str * (int | nil)", "int * str"])
    lines = [f"{name} := {written}"]
    if written == "int * str * bool":
        truth = [member(0, None, "int", "type_ref"),
                 member(1, None, "str", "type_ref"),
                 member(2, None, "bool", "type_ref")]
    elif written == "str * (int | nil)":
        truth = [member(0, None, "str", "type_ref"),
                 member(1, None, None, "sum", optional=True)]
    else:
        truth = [member(0, None, "int", "type_ref"),
                 member(1, None, "str", "type_ref")]
    return lines, truth, [], None


def t_sum_tagged(rng, name):
    if rng.random() < 0.5:
        lines = [f"{name} :=", "  Oneof", "  | $left int", "  | $right str"]
    else:
        lines = [f"{name} := $left int | $right str"]
    truth = [member(0, "$left", "int", "type_ref"), member(1, "$right", "str", "type_ref")]
    return lines, truth, [], None


def t_containers(rng, name):
    lines = [
        f"{name} :=",
        "  Object",
        "  * $items list[int]",
        "  * $seen set[str]",
        "  * $table dict[str, int]",
        "  * $maybe (int | nil)",
    ]
    truth = [
        member(0, "$items", None, "type_app"),
        member(1, "$seen", None, "type_app"),
        member(2, "$table", None, "type_app"),
        member(3, "$maybe", None, "sum", optional=True),
    ]
    return lines, truth, [], None


def t_generic(rng, name):
    lines = [f"{name}[T] :=", "  $v T"]
    truth = [member(0, "$v", "T", "type_ref")]
    return lines, truth, ["T"], None


def t_single_tagged(rng, name):
    lines = [f"{name} := $only int"]
    truth = [member(0, "$only", "int", "type_ref")]
    return lines, truth, [], None


def t_foreign_ref(rng, name, foreign_module, foreign_def, local):
    lines = [f"{name} :=", "  Object", f"  * $peer {local}.{foreign_def}"]
    truth = [member(0, "$peer", f"{local}.{foreign_def}", "type_ref",
                    resolved=f"{foreign_module}.{foreign_def}")]
    return lines, truth, [], foreign_module


def t_missing(rng, name):
    lines = [f"{name} :=", "  Object", "  * $ghost Missing", "  * $mix list[Missing]"]
    truth = [
        member(0, "$ghost", "Missing", "type_ref"),
        member(1, "$mix", None, "type_app"),
    ]
    return lines, truth, [], None


PLAIN_TEMPLATES = [t_product_tagged, t_product_positional, t_sum_tagged,
                   t_containers, t_generic, t_single_tagged, t_missing]


# ----------------------------------------------------------------------
# 一个 case
# ----------------------------------------------------------------------


def pick_layout(rng):
    count = rng.randint(3, 5)
    shallow = [p for p in CANDIDATES if "/" not in p]
    deep = [p for p in CANDIDATES if "/" in p]
    chosen = [rng.choice(shallow)]
    if rng.random() < 0.6:
        remaining = [p for p in shallow if p not in chosen]
        if remaining:
            chosen.append(rng.choice(remaining))
    # 每个 case 至少有一个三层以上的深目录
    chosen.append(rng.choice([p for p in deep if len(Path(p).parts) >= 3]))
    rest = [p for p in deep if p not in chosen]
    chosen += rng.sample(rest, min(count - len(chosen), len(rest)))
    return chosen


def build_case(case_index: int):
    rng = random.Random(1000 + case_index)
    layout = pick_layout(rng)
    modules = []
    known_definitions = []  # (module, local_name, def_name) 供后面的文件引用
    for rel_path in layout:
        module = module_name_of(rel_path)
        imports = []          # (module, local, aliased)
        definitions = []
        blocks = []
        plans = rng.sample(PLAIN_TEMPLATES, rng.randint(1, 3))
        if known_definitions and rng.random() < 0.8:
            plans.append(t_foreign_ref)
        rng.shuffle(plans)
        for index, template in enumerate(plans):
            name = f"{rng.choice(['Config', 'Report', 'Shape', 'Node', 'Rule', 'Plan'])}{index}"
            if template is t_foreign_ref:
                foreign_module, foreign_local, foreign_def = rng.choice(known_definitions)
                aliased = rng.random() < 0.5
                local = local_name_of(foreign_module, aliased, rng)
                body_lines, truth, params, needed = t_foreign_ref(
                    rng, name, foreign_module, foreign_def, local)
                imports.append((foreign_module, local, aliased))
            else:
                body_lines, truth, params, needed = template(rng, name)
            definitions.append({
                "name": name,
                "full_name": f"{module}.{name}",
                "generic_params": params,
                "members": truth,
            })
            if not params:
                known_definitions.append((module, None, name))
            blocks.append("\n".join(body_lines))
        if rng.random() < 0.3 and known_definitions and not imports:
            foreign_module, _, _ = rng.choice(known_definitions)
            local = local_name_of(foreign_module, True, rng)
            imports.append((foreign_module, local, True))
        statements = []
        seen = set()
        for import_module, local, aliased in imports:
            if (import_module, local) in seen:
                continue
            seen.add((import_module, local))
            statements.append(f"import {import_module} as {local}" if aliased
                              else f"import {import_module}")
        modules.append({
            "module": module,
            "file": rel_path,
            "imports": [{"module": m, "local": local} for m, local, _ in imports
                        if (m, local) in seen],
            "definitions": definitions,
            "source": ("\n".join(statements) + "\n\n" if statements else "")
                      + "\n\n".join(blocks) + "\n",
        })
    return {"case": case_index, "modules": modules}


def write_case(case: dict, check: bool) -> list:
    case_dir = HERE / f"case_{case['case']:03d}"
    stale = []
    files = {}
    for module in case["modules"]:
        files[module["file"]] = module["source"]
    files["expected.json"] = json.dumps(
        {"case": case["case"],
         "modules": [{k: v for k, v in module.items() if k != "source"}
                     for module in case["modules"]]},
        ensure_ascii=False, indent=2) + "\n"
    for rel_path, content in files.items():
        path = case_dir / rel_path
        if check:
            if not path.exists() or path.read_text() != content:
                stale.append(str(path.relative_to(HERE)))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if not check:
        print(f"{case_dir.relative_to(HERE)}: {len(case['modules'])} files, "
              f"{sum(len(m['definitions']) for m in case['modules'])} definitions")
    return stale


def main(argv):
    check = "--check" in argv
    stale = []
    for case_index in range(CASE_COUNT):
        stale += write_case(build_case(case_index), check)
    if check:
        if stale:
            print(f"{len(stale)} stale file(s):")
            for path in stale[:20]:
                print("  ", path)
            return 1
        print(f"type_descriptor corpus is current ({CASE_COUNT} cases)")
        return 0
    print(f"wrote {CASE_COUNT} cases under {HERE.relative_to(HERE.parent.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
