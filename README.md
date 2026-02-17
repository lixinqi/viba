# viba

## Demo

define a struct pattern matcher:

```
# all classes/functions defined by ImportFrom need no definition or decorator, just import.
# the type annotation in ImportFrom/Declare is only for comprehension. do not define
# Keep indent depth small.
# English comments.
# write test code in test_main whitch is called in __main__
# typical examples
# 1. pattern match: ($matched list[$name str * $edge int] <- StudentInfos)
# 2. function call: ($is_valid bool < $array_idx int <- CheckOutRange). $array_idx is predefined type-alias
# 3. for-loop expressed as match plus handle function with the same name shared in matched element and an argument of function. example: void <- ($for_each ($element str) <- List) <- ($do_each (void <- $element str))
# inline means no mock. the inlined logic should be implemented in this function body

MatchResult := dict[$pattern fx.Node, $target fx.Node]
MatchContext :=
	Object
  * $match_result MatchResult
  * $target fx.GraphModule
  * $pattern fx.GraphModule

StructMatcher := ImportFrom[
  "tst/torch_ap/struct_matcher.py",
  # __call__
  list[MatchResult]
  <- $target_gm fx.GraphModule
  # __init__
  <- $pattern_gm fx.GraphModule
]

PassResult := ImportFrom[
  "torch.fx.passes.infra.pass_manager",
  dataclass
  	* $graph_module fx.GraphModule
  	* $modified bool
]

StructPatternReplacer :=
  # __call__
  PassResult
  <- $target fx.GraphModule
  # __init__
  <- $pattern fx.GraphModule
  <- $get_replacement ($replacement fx.GraphModule <- MatchContext)
  <- $constraint ($meet_all_constraints bool <- MatchContext)
  # inline
  <- ($match_result MatchResult <- ... <- StructMatcher)
  <- ($match_ctx MatchContext <- ...)
  <- ($meet_all_constraints bool <- ... <- $constraint)
  <- $dead_code_eliminate (fx.GraphModule <- fx.GraphModule)
  <- $replacer ($replaced fx.GraphModule <- $target <- $replacement)
  <- ($ret_graph_module <- $replacer <- $dead_code_eliminate)
```
