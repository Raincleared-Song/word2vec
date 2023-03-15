class Config:
    seed = 200
    sparse = False
    input_path = 'data/text8'
    output_path = 'checkpoints/embed_dense_200_b128_nc4_wz8'

    embed_dim = 100
    batch_size = 128
    init_learning_rate = 0.001
    epoch_number = 20
    device = 'cuda:0'

    remove_stopwords = False
    discard_words = True
    use_neg_prob = True
    soft_slide_max = -1  # maximum size (inclusive) for soft sliding window, -1 to switch off
    discard_t = 0.0001
    neg_pow = 0.5
    neg_count = 4
    window_size = 8
    neg_table_size = int(1e7)
