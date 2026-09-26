WINDOW_SECONDS = 60
TRAIN_STRIDE_SECONDS = 10
EVAL_STRIDE_SECONDS = 1


def window_end_times(step_is_valid, window_seconds=WINDOW_SECONDS, stride_seconds=TRAIN_STRIDE_SECONDS):
    if window_seconds < 1:
        raise ValueError("window has to be at least 1 sec")
    if stride_seconds < 1:
        raise ValueError("stride has to be at least 1")
    n_steps = len(step_is_valid)
    ends = []
    end_time = window_seconds - 1
    while end_time < n_steps:
        if window_has_no_hole(step_is_valid, end_time, window_seconds):
            ends.append(end_time)
        end_time += stride_seconds
    return ends


def end_is_on_stride(end_time, stride_seconds, window_seconds=WINDOW_SECONDS):
    if stride_seconds < 1:
        raise ValueError("stride has to be at least 1")
    if window_seconds < 1:
        raise ValueError("window has to be at least 1 sec")
    first_end = window_seconds - 1
    if end_time < first_end:
        return False
    return (end_time - first_end) % stride_seconds == 0


def window_has_no_hole(step_is_valid, end_time, window_seconds=WINDOW_SECONDS):
    # a hole inside the 60 sec means we dont build that window at all
    start_time = end_time - window_seconds + 1
    if start_time < 0:
        return False
    if end_time >= len(step_is_valid):
        return False
    second = start_time
    while second <= end_time:
        if not step_is_valid[second]:
            return False
        second += 1
    return True
