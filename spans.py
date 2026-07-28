def spans_overlap(start_a, end_a, start_b, end_b):
    if end_a <= start_a or end_b <= start_b:
        raise ValueError("a span has to end after it starts")
    # [start, end) so touching at an endpoint is not an overlap
    return start_a < end_b and start_b < end_a


def second_hits_a_span(second_start, spans):
    second_end = second_start + 1
    for start, end in spans:
        if spans_overlap(second_start, second_end, start, end):
            return True
    return False


def gap_spans_from_record_onsets(onsets, record_duration):
    if record_duration <= 0:
        raise ValueError(
            f"record duration has to be positive, we got {record_duration}"
        )
    if len(onsets) < 1:
        raise ValueError("need at least one data record time stamp")
    spans = []
    if onsets[0] > 0:
        spans.append((0.0, float(onsets[0])))
    index = 0
    while index < len(onsets) - 1:
        start = float(onsets[index])
        nxt = float(onsets[index + 1])
        if nxt < start:
            raise ValueError(
                f"edf record times went backwards, we got {start} then {nxt}"
            )
        jump = nxt - start
        if jump < record_duration:
            raise ValueError(
                f"edf records overlap, we got start {start} then {nxt}"
            )
        # r14: a jump bigger than one data record is a silent gap
        if jump > record_duration:
            spans.append((start + record_duration, nxt))
        index += 1
    return spans
