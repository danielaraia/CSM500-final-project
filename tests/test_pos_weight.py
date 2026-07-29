import pytest

from labels import LABEL_INTERICTAL, LABEL_SOP, LABEL_UNUSED
from pos_weight import (
    positive_class_weight,
    weight_from_class_numbers,
    weight_from_train_labels,
)


def test_weight_is_neg_over_pos():
    assert positive_class_weight(90, 10) == pytest.approx(9.0)


def test_zero_positives_fails():
    with pytest.raises(ValueError, match="positive"):
        positive_class_weight(10, 0)


def test_nine_interictal_and_one_sop_is_weight_nine():
    labels = [LABEL_INTERICTAL] * 9 + [LABEL_SOP]
    weight, n_negative, n_positive = weight_from_train_labels(labels)
    assert n_negative == 9
    assert n_positive == 1
    assert weight == pytest.approx(9.0)


def test_unused_label_fails():
    with pytest.raises(ValueError, match="cant include"):
        weight_from_train_labels([LABEL_SOP, LABEL_UNUSED])


def test_class_numbers_match_named_labels():
    weight, n_negative, n_positive = weight_from_class_numbers([0, 0, 0, 1])
    assert n_negative == 3
    assert n_positive == 1
    assert weight == pytest.approx(3.0)

