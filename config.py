class Config:
    seed = 300
    sparse = False
    input_path = 'data/text8'
    output_path = 'checkpoints/embed_dense_300_b128_nc8_wz6'

    embed_dim = 100
    batch_size = 128
    init_learning_rate = 0.001
    epoch_number = 10
    device = 'cuda:0'

    discard_t = 0.0001
    neg_pow = 0.5
    neg_count = 8
    window_size = 6
    neg_table_size = int(1e7)
