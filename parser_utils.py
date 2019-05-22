import argparse


def get_parser():
    parser = argparse.ArgumentParser(description="Main")
    parser.add_argument('--input_folder', action="store", dest="input_folder", help="input_folder", default="../codes/")
    parser.add_argument('--output_folder', action="store", dest="output_folder", help="output_folder", default="results")
    parser.add_argument('--classifier', action="store", dest="classifier", help="randomforest for now", default="randomforest")
    parser.add_argument('--metric', action="store", dest="metric", help="jaccard or cosine", default="cosine")
    parser.add_argument('--vectorizer', action="store", dest="vectorizer", help="count or tfidf", default="tfidf")
    parser.add_argument('--clustering_method', action="store", dest="clustering_method", help="single complete average ward weighted centroid median", default="average")
    parser.add_argument('--matrix_form', action="store", dest="matrix_form", help="tfidf for now", default="tdidf")
    parser.add_argument('--max_features', action="store", dest="max_features", type=int, default=None)
    parser.add_argument('--files_limit', action="store", dest="files_limit", type=int, default=100)
    parser.add_argument('--override', action="store", dest="override", default=True, type=lambda x:x.lower not in ['false', '0', 'n'])
    parser.add_argument('--profiler', action="store_true", dest="profiler", default=False)  # type=lambda x:x.lower in ['true', '1', 'y']

    parser.add_argument('--num_features', action="store", dest="num_features", type=int, default=None)
    parser.add_argument('--select_top_tokens', action="store", dest="select_top_tokens", type=int, default=1000)
    parser.add_argument('--ngram_range', action="store", dest="ngram_range", type=int, default=1)
    # parser.add_argument('--n_clusters', action="store", dest="n_clusters", type=int, default=7)
    # parser.add_argument('--n_topics', action="store", dest="n_topics", type=int, default=20)
    # parser.add_argument('--top_similar_functions', action="store", dest="top_similar_functions", type=int, default=10)
    parser.add_argument('--min_word_count', action="store", dest="min_word_count", type=int, default=40)
    parser.add_argument('--num_workers', action="store", dest="num_workers", type=int, default=4)
    parser.add_argument('--cores_to_use', action="store", dest="cores_to_use", type=int, default=1)
    parser.add_argument('--min_token_count', action="store", dest="min_token_count", type=int, default=-1)
    parser.add_argument('--context', action="store", dest="context", type=int, default=10)
    parser.add_argument('--seed', action="store", dest="seed", type=int, default=0)
    parser.add_argument('--no_top_words', action="store", dest="no_top_words", type=int, default=10)
    parser.add_argument('--cluster_analysis_count', action="store", dest="cluster_analysis_count", type=int, default=-1)
    # parser.add_argument('--downsampling', action="store", dest="downsampling", type=int, default=1e-3)
    parser.add_argument('--color_thresh', action="store", dest="color_thresh", type=float, default=0.115)
    # pass with spaces between arguments, eg --security_keywords xss sql injection
    parser.add_argument('--security_keywords', action="store", dest="security_keywords", nargs='*', default=None)
    # --stages_to_run vectors tfidf distances clustering
    parser.add_argument('--stages_to_run', action="store", dest="stages_to_run", nargs='*', default=['vectors', 'tfidf', 'distances', 'clustering'])
    return parser


def str_to_params(args):
    parser = get_parser()
    params = parser.parse_args(args=args)
    return params
