class Parameters(object):
    review_path = "data/clear_t_review.csv"
    train_review_path = "data/train_review.csv"
    test_review_path = "data/test_review.csv"
    user_history_review_path = "data/user_history_data.txt"

    shop_index_path = "data/shop_index.txt"
    shopMes_path = "data/clear_t_shop.csv"

    hieraechical_path = "data/type_embedding"
    attribute_path = "data/attribute_embedding"
    
    concat_embedding_matrix_path = "data/embedding_matrix.npy"
    concat_graph_matrix_path = "data/graph_matrix.npy"
    MAX_LEN = 20
    type_embedding_dim = 30
    BATCH_SIZE = 128
    EMBEDDING_DIM = 256
    HIDDEN_SIZE = 30
    DENSE_SIZE = 256
    DROPOUT_RATE = 0.5
    DISTANCE = 200
    k = 10                                                       # top-k
    history_review_count = 5