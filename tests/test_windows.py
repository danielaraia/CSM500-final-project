from windows import (
    EVAL_STRIDE_SECONDS,
    TRAIN_STRIDE_SECONDS,
    WINDOW_SECONDS,
    end_is_on_stride,
    window_end_times,
    window_has_no_hole,
)


def test_window_needs_sixty_consecutive_valid_seconds():
    step_is_valid = [True] * 80
    ends = window_end_times(step_is_valid, WINDOW_SECONDS, TRAIN_STRIDE_SECONDS)
    assert ends[0] == 59
    assert ends == [59, 69, 79]


def test_window_is_not_built_across_a_hole():
    step_is_valid = [True] * 80
    # hole at sec 15, so teh windows that cover 15 get dropped
    step_is_valid[15] = False
    assert window_has_no_hole(step_is_valid, 59) is False
    assert window_has_no_hole(step_is_valid, 79) is True
    ends = window_end_times(step_is_valid, WINDOW_SECONDS, TRAIN_STRIDE_SECONDS)
    assert 59 not in ends
    assert 69 not in ends
    assert 79 in ends


def test_eval_stride_scores_every_valid_end_second():
    step_is_valid = [True] * 62
    ends = window_end_times(step_is_valid, WINDOW_SECONDS, EVAL_STRIDE_SECONDS)
    assert ends == [59, 60, 61]


def test_train_hop_ends_are_every_ten_from_fifty_nine():
    assert end_is_on_stride(59, TRAIN_STRIDE_SECONDS) is True
    assert end_is_on_stride(60, TRAIN_STRIDE_SECONDS) is False
    assert end_is_on_stride(69, TRAIN_STRIDE_SECONDS) is True
    assert end_is_on_stride(61, EVAL_STRIDE_SECONDS) is True
    assert end_is_on_stride(58, TRAIN_STRIDE_SECONDS) is False


def test_eval_hop_skips_a_hole_but_keeps_neighbours():
    step_is_valid = [True] * 125
    step_is_valid[61] = False
    ends = window_end_times(step_is_valid, WINDOW_SECONDS, EVAL_STRIDE_SECONDS)
    # a 60 sec window covers the hole at 61 when its end is 61 through 120
    assert 59 in ends
    assert 60 in ends
    assert 61 not in ends
    assert 62 not in ends
    assert 120 not in ends
    assert 121 in ends
    assert 122 in ends
