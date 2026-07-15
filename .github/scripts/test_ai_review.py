import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).parent))

import ai_review


class ReviewPipelineTests(unittest.TestCase):
    def test_chunks_preserve_line_numbers_and_overlap(self):
        text = "\n".join(f"line {index}" for index in range(10))
        result = ai_review.chunks("src/main.rs", text, "code", lines_per_chunk=5, overlap=2)
        self.assertEqual([1, 4, 7], [item["line"] for item in result])
        self.assertIn("line 3", result[1]["text"])

    def test_rank_chunks_selects_documentation_and_code(self):
        items = [
            {"path": "src/router.rs", "kind": "code", "line": 1, "text": "route request"},
            {"path": "src/persist.rs", "kind": "code", "line": 1, "text": "save state"},
            {"path": "docs/router.md", "kind": "documentation", "line": 1, "text": "router architecture"},
        ]
        result = ai_review.rank_chunks(
            items,
            "router routing change",
            limits={"code": 1, "documentation": 1},
            character_budget=1000,
        )
        self.assertEqual(["docs/router.md", "src/router.rs"], sorted(item["path"] for item in result))

    def test_knowledge_files_excludes_build_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "src").mkdir()
            (root / "target").mkdir()
            (root / "README.md").write_text("readme", encoding="utf-8")
            (root / "docs" / "architecture.md").write_text("docs", encoding="utf-8")
            (root / "src" / "main.rs").write_text("fn main() {}", encoding="utf-8")
            (root / "target" / "generated.rs").write_text("generated", encoding="utf-8")
            files = ai_review.knowledge_files(root)
        self.assertEqual(
            ["README.md", "docs/architecture.md", "src/main.rs"],
            [path for path, _ in files],
        )

    def test_clip_marks_omitted_content(self):
        result = ai_review.clip("abcdefghij", 4, "diff")
        self.assertIn("abcd", result)
        self.assertIn("omitted 6 characters", result)

    def test_diff_falls_back_to_file_patches(self):
        files = [{"filename": "src/main.rs", "patch": "@@ -1 +1 @@\n-old\n+new"}]
        with patch.object(ai_review, "request", side_effect=RuntimeError("unavailable")):
            result = ai_review.fetch_diff("https://api.github.com", "owner/repo", 1, files, "token")
        self.assertIn("diff --git a/src/main.rs b/src/main.rs", result)
        self.assertIn("+new", result)

    def test_provider_defaults_to_github_models(self):
        with patch.dict(ai_review.os.environ, {}, clear=True):
            base_url, api_key, model = ai_review.provider_config("github-token")
        self.assertEqual("https://models.github.ai/inference", base_url)
        self.assertEqual("github-token", api_key)
        self.assertEqual("openai/gpt-4.1", model)

    def test_custom_provider_requires_key(self):
        with patch.dict(
            ai_review.os.environ,
            {"AI_REVIEW_BASE_URL": "https://example.com/v1"},
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "AI_REVIEW_API_KEY"):
                ai_review.provider_config("github-token")

    def test_prompt_has_hard_total_budget(self):
        prompt = ai_review.build_prompt(
            {"title": "title" * 1000, "body": "body" * 10000},
            "files" * 10000,
            "diff" * 10000,
            "sources" * 10000,
            [{"path": "src/main.rs", "kind": "code", "line": 1, "text": "code" * 10000}],
        )
        self.assertLessEqual(len(prompt), 48100)
        self.assertIn("Changed files", prompt)
        self.assertIn("Diff", prompt)


if __name__ == "__main__":
    unittest.main()
