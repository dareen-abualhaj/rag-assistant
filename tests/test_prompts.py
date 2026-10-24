from src.prompts import NO_ANSWER, build_context, build_messages


def _hit(title, text):
    return {"text": text, "metadata": {"title": title}}


def test_build_context_numbers_chunks():
    ctx, used = build_context([_hit("A", "x" * 200), _hit("B", "y" * 200)])
    assert ctx.startswith("[1] A")
    assert "[2] B" in ctx
    assert len(used) == 2


def test_build_context_respects_char_budget():
    hits = [_hit(f"T{i}", "z" * 1000) for i in range(10)]
    ctx, used = build_context(hits, max_chars=2500)
    assert len(ctx) <= 2500
    assert 0 < len(used) < 10


def test_build_context_empty():
    assert build_context([]) == ("", [])


def test_messages_structure_and_no_answer_rule():
    msgs = build_messages("Q?", "[1] A\ntext")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert NO_ANSWER in msgs[0]["content"]
    assert "Question: Q?" in msgs[1]["content"]
    assert "rejected" not in msgs[0]["content"]


def test_strict_messages_add_extra_rule():
    msgs = build_messages("Q?", "[1] A\ntext", strict=True)
    assert "rejected" in msgs[0]["content"]