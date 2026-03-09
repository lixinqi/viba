import os


def get_all_relative_file_paths(
    root_dir: str,
    extention: str | None = None,
) -> list[str]:
    """
    Get all relative file paths under root_dir.

    :param root_dir: The root directory to walk.
    :param extention: Optional file extension filter (e.g. ".py"). None means all files.
    :return: List of relative file paths (no "./" prefix).
    """
    result = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if extention is not None and not fname.endswith(extention):
                continue
            full = os.path.join(dirpath, fname)
            rel = os.path.relpath(full, root_dir)
            result.append(rel)
    result.sort()
    return result


if __name__ == "__main__":
    import tempfile
    import unittest

    class TestGetAllRelativeFilePaths(unittest.TestCase):

        def test_empty_directory(self):
            with tempfile.TemporaryDirectory() as td:
                result = get_all_relative_file_paths(td)
                self.assertEqual(result, [])

        def test_flat_files(self):
            with tempfile.TemporaryDirectory() as td:
                for name in ["a.txt", "b.py", "c.md"]:
                    open(os.path.join(td, name), "w").close()
                result = get_all_relative_file_paths(td)
                self.assertEqual(result, ["a.txt", "b.py", "c.md"])

        def test_nested_directories(self):
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "sub", "deep"))
                open(os.path.join(td, "top.txt"), "w").close()
                open(os.path.join(td, "sub", "mid.txt"), "w").close()
                open(os.path.join(td, "sub", "deep", "bot.txt"), "w").close()
                result = get_all_relative_file_paths(td)
                self.assertIn("top.txt", result)
                self.assertIn(os.path.join("sub", "mid.txt"), result)
                self.assertIn(os.path.join("sub", "deep", "bot.txt"), result)

        def test_filter_by_extension(self):
            with tempfile.TemporaryDirectory() as td:
                for name in ["a.py", "b.py", "c.txt", "d.md"]:
                    open(os.path.join(td, name), "w").close()
                result = get_all_relative_file_paths(td, extention=".py")
                self.assertEqual(result, ["a.py", "b.py"])

        def test_filter_none_returns_all(self):
            with tempfile.TemporaryDirectory() as td:
                for name in ["x.py", "y.txt"]:
                    open(os.path.join(td, name), "w").close()
                result = get_all_relative_file_paths(td, extention=None)
                self.assertEqual(result, ["x.py", "y.txt"])

        def test_nested_empty_subdirs_ignored(self):
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "a", "b", "c"))
                os.makedirs(os.path.join(td, "empty"))
                open(os.path.join(td, "a", "b", "c", "leaf.txt"), "w").close()
                result = get_all_relative_file_paths(td)
                self.assertEqual(result, [os.path.join("a", "b", "c", "leaf.txt")])

        def test_nested_filter_extension_across_levels(self):
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "src", "pkg"))
                open(os.path.join(td, "readme.md"), "w").close()
                open(os.path.join(td, "src", "main.py"), "w").close()
                open(os.path.join(td, "src", "util.txt"), "w").close()
                open(os.path.join(td, "src", "pkg", "core.py"), "w").close()
                open(os.path.join(td, "src", "pkg", "data.json"), "w").close()
                result = get_all_relative_file_paths(td, extention=".py")
                self.assertEqual(result, [
                    os.path.join("src", "main.py"),
                    os.path.join("src", "pkg", "core.py"),
                ])

        def test_nested_same_filename_different_dirs(self):
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "a"))
                os.makedirs(os.path.join(td, "b"))
                os.makedirs(os.path.join(td, "a", "c"))
                for d in ["a", "b", os.path.join("a", "c")]:
                    open(os.path.join(td, d, "init.py"), "w").close()
                result = get_all_relative_file_paths(td)
                self.assertEqual(result, [
                    os.path.join("a", "c", "init.py"),
                    os.path.join("a", "init.py"),
                    os.path.join("b", "init.py"),
                ])

        def test_nested_deeply_four_levels(self):
            with tempfile.TemporaryDirectory() as td:
                path = os.path.join(td, "l1", "l2", "l3", "l4")
                os.makedirs(path)
                open(os.path.join(td, "l1", "f1.txt"), "w").close()
                open(os.path.join(td, "l1", "l2", "f2.txt"), "w").close()
                open(os.path.join(td, "l1", "l2", "l3", "f3.txt"), "w").close()
                open(os.path.join(td, "l1", "l2", "l3", "l4", "f4.txt"), "w").close()
                result = get_all_relative_file_paths(td)
                self.assertEqual(result, [
                    os.path.join("l1", "f1.txt"),
                    os.path.join("l1", "l2", "f2.txt"),
                    os.path.join("l1", "l2", "l3", "f3.txt"),
                    os.path.join("l1", "l2", "l3", "l4", "f4.txt"),
                ])

        def test_nested_sibling_subtrees(self):
            with tempfile.TemporaryDirectory() as td:
                os.makedirs(os.path.join(td, "x", "y"))
                os.makedirs(os.path.join(td, "x", "z"))
                os.makedirs(os.path.join(td, "w"))
                open(os.path.join(td, "x", "y", "a.py"), "w").close()
                open(os.path.join(td, "x", "z", "b.py"), "w").close()
                open(os.path.join(td, "w", "c.py"), "w").close()
                result = get_all_relative_file_paths(td)
                self.assertEqual(result, [
                    os.path.join("w", "c.py"),
                    os.path.join("x", "y", "a.py"),
                    os.path.join("x", "z", "b.py"),
                ])

        def test_no_dot_slash_prefix(self):
            with tempfile.TemporaryDirectory() as td:
                open(os.path.join(td, "file.txt"), "w").close()
                result = get_all_relative_file_paths(td)
                for p in result:
                    self.assertFalse(p.startswith("./"))

    unittest.main()
