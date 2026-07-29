from labels import LABEL_INTERICTAL, LABEL_SOP


def positive_class_weight(n_negative, n_positive):
    if n_positive < 1:
        raise ValueError(f"need some positive train windows, we got {n_positive}")
    if n_negative < 1:
        raise ValueError(f"need some negative train windows, we got {n_negative}")
    # so the two classes pull equally on the loss
    return float(n_negative) / float(n_positive)


def weight_from_train_labels(train_labels):
    n_positive = 0
    n_negative = 0
    index = 0
    while index < len(train_labels):
        lab = train_labels[index]
        if lab == LABEL_SOP:
            n_positive += 1
        elif lab == LABEL_INTERICTAL:
            n_negative += 1
        else:
            raise ValueError(f"train labels cant include {lab}")
        index += 1
    return positive_class_weight(n_negative, n_positive), n_negative, n_positive


def weight_from_class_numbers(labels):
    n_positive = 0
    n_negative = 0
    index = 0
    while index < len(labels):
        lab = int(labels[index])
        if lab == 1:
            n_positive += 1
        elif lab == 0:
            n_negative += 1
        else:
            raise ValueError(f"class numbers have to be 0 or 1, we got {lab}")
        index += 1
    return positive_class_weight(n_negative, n_positive), n_negative, n_positive
