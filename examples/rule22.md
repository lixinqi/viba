```viba
CodeLength := int
MaxLines := int
CoverageRatio := float
TimeBudget := float
HasTests := bool
ModuleName := str
Tags := set[str]
Scores := dict[str, int]

Rule22 :=
  RuleObject
  * $code_length Metric[CodeLength]
  * $max_lines Metric[MaxLines]
  * $coverage_ratio Metric[CoverageRatio]
  * $time_budget Metric[TimeBudget]
  * $has_tests Metric[HasTests]
  * $module_name Metric[ModuleName]
  * $tags Metric[Tags]
  * $scores Metric[Scores]
  * $assert_00_code_length_at_most_24
      Assert[{code_length at most 24}, $python_code {
def handler(self):
    return self.code_length.value <= 24
}]
  * $assert_01_code_length_under_100
      Assert[{code_length under 100}, $python_code {
def handler(self):
    return self.code_length.value < 100
}]
  * $assert_02_max_lines_at_most_24
      Assert[{max_lines at most 24}, $python_code {
def handler(self):
    return self.max_lines.value <= 24
}]
  * $assert_03_max_lines_under_100
      Assert[{max_lines under 100}, $python_code {
def handler(self):
    return self.max_lines.value < 100
}]
  * $assert_04_code_length_not_above_max_lines
      Assert[{code_length not above max_lines}, $python_code {
def handler(self):
    return self.code_length.value <= self.max_lines.value
}]
  * $assert_05_coverage_ratio_within_50_5
      Assert[{coverage_ratio within 50.5}, $python_code {
def handler(self):
    return self.coverage_ratio.value <= 50.5
}]
  * $assert_06_coverage_ratio_non_negative
      Assert[{coverage_ratio non-negative}, $python_code {
def handler(self):
    return self.coverage_ratio.value >= 0.0
}]
  * $assert_07_time_budget_within_50_5
      Assert[{time_budget within 50.5}, $python_code {
def handler(self):
    return self.time_budget.value <= 50.5
}]
  * $assert_08_time_budget_non_negative
      Assert[{time_budget non-negative}, $python_code {
def handler(self):
    return self.time_budget.value >= 0.0
}]
  * $assert_09_has_tests_required
      Assert[{has_tests required}, $python_code {
def handler(self):
    return self.has_tests.value
}]
  * $assert_10_has_tests_forbidden
      Assert[{has_tests forbidden}, $python_code {
def handler(self):
    return not self.has_tests.value
}]
  * $assert_11_module_name_short
      Assert[{module_name short}, $python_code {
def handler(self):
    return len(self.module_name.value) <= 8
}]
  * $assert_12_module_name_not_empty
      Assert[{module_name not empty}, $python_code {
def handler(self):
    return len(self.module_name.value) >= 1
}]
  * $assert_13_tags_bounded
      Assert[{tags bounded}, $python_code {
def handler(self):
    return len(self.tags.value) <= 3
}]
  * $assert_14_tags_non_empty
      Assert[{tags non-empty}, $python_code {
def handler(self):
    return len(self.tags.value) >= 1
}]
  * $assert_15_scores_non_empty
      Assert[{scores non-empty}, $python_code {
def handler(self):
    return len(self.scores.value) >= 1
}]
  * $assert_16_scores_bounded
      Assert[{scores bounded}, $python_code {
def handler(self):
    return len(self.scores.value) <= 4
}]
```
