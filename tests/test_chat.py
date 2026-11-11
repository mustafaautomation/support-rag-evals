from support_rag.chat import main


def test_chat_quotes_the_refund_window(capsys: object) -> None:
    assert main(["How long is the refund window for an annual plan purchase?"]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "refunded in full within 14 days" in captured.out
    assert "refund-policy" in captured.out
