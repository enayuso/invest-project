from stock_advisor_mcp.units import format_krw, format_pct


def test_format_krw_trillion():
    result = format_krw(333_605_938_000_000)
    assert result == {"display": "333.6조원", "krw": 333_605_938_000_000}


def test_format_krw_eok():
    result = format_krw(21_334_708_470)
    assert result == {"display": "213.3억원", "krw": 21_334_708_470}


def test_format_krw_small_value_renders_in_eok_not_bare_won():
    # Regression for the bug found in live testing: this value rendered as
    # "-31,003,660원" while every neighbouring value in the same series
    # rendered in 억원, forcing the model to convert units in its head.
    result = format_krw(-31_003_660)
    assert result["display"] == "-0.31억원"
    assert "million" not in result["display"].lower()
    assert "billion" not in result["display"].lower()


def test_format_krw_zero_is_zero_eok_no_sign():
    result = format_krw(0)
    assert result == {"display": "0억원", "krw": 0}


def test_format_krw_rounds_to_zero_eok_has_no_sign():
    assert format_krw(400_000)["display"] == "0억원"
    assert format_krw(-400_000)["display"] == "0억원"


def test_format_krw_jo_boundary_never_shows_10000_point_0_eok():
    # 9,999.96억 rounds to "10000.0" at 1 decimal, which is not a valid
    # 억원 display (10000억 == 1조) -- the unit must be chosen after
    # rounding, not before.
    result = format_krw(999_996_000_000)
    assert result["display"] == "1.0조원"


def test_format_krw_eok_boundary_never_shows_2dp_at_or_above_1_eok():
    # The matching boundary one tier down: 0.99996억 rounds to "1.00" at
    # 2 decimals, which would look like it is still short of a full 1억.
    # It must render with the >=1억 tier's own precision instead.
    result = format_krw(99_996_000)
    assert result["display"] == "1.0억원"


def test_format_krw_negative():
    result = format_krw(-13_276_391_590)
    assert result["display"] == "-132.8억원"
    assert result["krw"] == -13_276_391_590


def test_format_krw_none_passthrough():
    assert format_krw(None) is None


def test_format_krw_never_contains_english_scale_words_across_magnitudes():
    for sign in (1, -1):
        for value in [
            0,
            123,
            45_000,
            1_000_000,
            99_996_000,
            1_000_000_000,
            367_389_147_990,
            999_996_000_000,
            1_000_000_000_000,
            5_500_000_000_000,
            1_792_671_836_274_688,
        ]:
            display = format_krw(sign * value)["display"]
            assert "million" not in display.lower()
            assert "billion" not in display.lower()
            # Never a bare "...원" reading -- always converted to 억원 or 조원.
            assert display.endswith("억원") or display.endswith("조원"), display


def test_format_pct_ratio_to_percent():
    assert format_pct(0.30794) == 30.79


def test_format_pct_none():
    assert format_pct(None) is None
