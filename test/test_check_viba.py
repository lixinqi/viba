from viba.parser import parser
from pprint import pprint
from viba.type import parse
from viba.unparser import unparse
from viba.chain import convert_to_chain_style
import pprint
from pathlib import Path

#ast_obj = parser.parse(open(Path(__file__).resolve().parent / "a.viba").read())
#print(type(ast_obj))
#ast_obj = parser.parse(open(Path(__file__).resolve().parent / "b.viba").read())
ast_obj = parser.parse(open("/tmp/a.viba").read())
ast_obj = [convert_to_chain_style(t) for t in parse(ast_obj)]
pp = pprint.PrettyPrinter(indent=2, width=20)

for t in ast_obj:
    pp.pprint(t.to_dict())
print(unparse(ast_obj))
