import pytest

from kaldialign import align, edit_distance, streaming_edit_distance

EPS = "*"


def test_align():
    a = ["a", "b", "c"]
    b = ["a", "s", "x", "c"]
    ali = align(a, b, EPS)
    assert ali == [("a", "a"), ("b", "s"), (EPS, "x"), ("c", "c")]
    dist = edit_distance(a, b)
    assert dist == {"ins": 1, "del": 0, "sub": 1, "total": 2}

    a = ["a", "b"]
    b = ["b", "c"]
    ali = align(a, b, EPS)
    assert ali == [("a", EPS), ("b", "b"), (EPS, "c")]
    dist = edit_distance(a, b)
    assert dist == {"ins": 1, "del": 1, "sub": 0, "total": 2}

    a = ["A", "B", "C"]
    b = ["D", "C", "A"]
    ali = align(a, b, EPS)
    assert ali == [("A", "D"), ("B", EPS), ("C", "C"), (EPS, "A")]
    dist = edit_distance(a, b)
    assert dist == {"ins": 1, "del": 1, "sub": 1, "total": 3}

    a = ["A", "B", "C", "D"]
    b = ["C", "E", "D", "F"]
    ali = align(a, b, EPS)
    assert ali == [
        ("A", EPS),
        ("B", EPS),
        ("C", "C"),
        (EPS, "E"),
        ("D", "D"),
        (EPS, "F"),
    ]
    dist = edit_distance(a, b)
    assert dist == {"ins": 2, "del": 2, "sub": 0, "total": 4}


def test_edit_distance():
    a = ["a", "b", "c"]
    b = ["a", "s", "x", "c"]
    results = edit_distance(a, b)
    assert results == {"ins": 1, "del": 0, "sub": 1, "total": 2}


def test_edit_distance_sclite():
    a = ["a", "b"]
    b = ["b", "c"]
    results = edit_distance(a, b, sclite_mode=True)
    assert results == {"ins": 1, "del": 1, "sub": 0, "total": 2}


@pytest.mark.parametrize(
    ["tau", "expected"],
    [
        (-3, 20),
        (-2, 18),
        (-1, 17),
        (0, 15),
        (1, 13),
        (2, 11),
        (3, 9),
        (4, 7),
        (5, 6),
    ],
)
def test_streaming_edit_distance(tau, expected):
    # Example (from Ehsan Variani):
    # This example calculates streaming chracter edit-distance. Note that we usually want word level edit-distance.

    # ref: kkkkkiiiiiitttttttttttteeeeeeeeennnnnnnnnn---iiiinnnnn----ttttthhhhhhheeeeeee--kkkkiiiiiittcccchhhhheeeeennnnnnn
    # reference timestamps (corresponding to the end of character)
    # t_r:     5    11  15      23       32        42 45  49   54  58   63     70     77 79 83    89 91 95  100  105    112
    # hyp: sssssssiiiiiitttttttttiiiiiiinnnnnnnnnnggggg--iiinn-----ttttthhhhhheeeeee-------kkkkkiiiitttttccccchhhhhheeeeeennnnn
    # t_h:       7    13   18  22     29        39   44 46 49 51 56   61    67    73     80   85  89   94   99   105   111  116
    # hypothesis timestamps (corresponding to the end of character)
    # c_del = 1
    # c_ins = 1
    # c_sub = 2
    # c_str = 1
    ref_text = "kitten in the kitchen"
    hyp_text = "sitting in the kitchen"
    # fmt: off
    ref_times = [5, 11, 15, 23, 32, 42, 45, 49, 54, 58, 63,70, 77, 79, 83, 89, 91, 95, 100, 105, 112]
    hyp_times = [7, 13, 18, 22, 29, 39, 44, 46, 49, 51, 56, 61, 67, 73, 80, 85, 89, 94, 99, 105, 111, 116]
    # fmt: on
    ref = list(zip(ref_text, ref_times, ref_times))
    hyp = list(zip(hyp_text, hyp_times, hyp_times))
    distance = streaming_edit_distance(
        ref, hyp, ins_cost=1, del_cost=1, sub_cost=2, str_cost=1, threshold=tau
    )
    assert distance == expected


if __name__ == "__main__":
    test_align()
    test_edit_distance()
    test_edit_distance_sclite()
    test_streaming_edit_distance()
