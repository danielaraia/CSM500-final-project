from windows import (
    EVAL_STRIDE_SECONDS,
    TRAIN_STRIDE_SECONDS,
    WINDOW_SECONDS,
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
