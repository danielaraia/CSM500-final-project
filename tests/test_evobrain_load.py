from model.evobrain_load import load_evobrain, locked_model_args


def test_locked_args_beat_evobrain_code_defaults():
    args = locked_model_args()
    # evobrain args.py defaults are 19 nodes, 12 sec, 100 bins, augment on
    assert args.num_nodes == 8
    assert args.max_seq_len == 60
    assert args.input_dim == 128
    assert args.data_augment is False
    assert args.agg == "max"
    assert args.rnn_units == 64
    assert args.dropout == 0.0
    assert args.top_k == 3


def test_load_evobrain_returns_model_or_names_missing_packages():
    try:
        model = load_evobrain()
    except ValueError as exc:
        assert "cant load EvoBrain, missing" in str(exc)
    else:
        assert model is not None
