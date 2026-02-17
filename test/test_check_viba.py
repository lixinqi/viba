from viba.parser import parser
from pathlib import Path

ast_obj = parser.parse(open(Path(__file__).resolve().parent / "a.viba").read())
print(type(ast_obj))
ast_obj = parser.parse(open(Path(__file__).resolve().parent / "b.viba").read())
print(type(ast_obj))
