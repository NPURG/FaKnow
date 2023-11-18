__all__ = ['run_m3fend', 'run_m3fend_from_yaml']

def run_m3fend(
        dataset : str = 'ch',
        domain_num : int = 3,
        emb_dim : int = 768,
        mlp_dims : list = [384],
        use_cuda : bool = True,
        lr : float = 0.0001,
        dropout : float = 0.2,
        # train_loader : ,
        # val_loader : ,
        # test_loader : ,
        weight_decay : float = 0.00005,
        save_param_dir : str = './param_model',
        semantic_num : int = 7,
        emotion_num : int = 7,
        style_num : int = 2,
        lnn_dim : int = 50,
        early_stop : int = 3,
        epoches : int = 50
):
    if dataset == 'en':
        root_path = './data/en/'
        category_dict = {
            "gossipcop": 0,
            "politifact": 1,
            "COVID": 2,
        }
    elif dataset == 'ch':
        root_path = './data/ch/'
        if domain_num == 9:
            category_dict = {
                "科技": 0,
                "军事": 1,
                "教育考试": 2,
                "灾难事故": 3,
                "政治": 4,
                "医药健康": 5,
                "财经商业": 6,
                "文体娱乐": 7,
                "社会生活": 8,
            }
        elif domain_num == 6:
            category_dict = {
                "教育考试": 0,
                "灾难事故": 1,
                "医药健康": 2,
                "财经商业": 3,
                "文体娱乐": 4,
                "社会生活": 5,
            }
        elif domain_num == 3:
            category_dict = {
                "政治": 0,  # 852
                "医药健康": 1,  # 1000
                "文体娱乐": 2,  # 1440
            }