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
