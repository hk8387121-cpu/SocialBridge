import os

class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "socialbridge-daa-capstone")
    DEBUG = True
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    ALLOWED_EXTENSIONS = {"csv", "txt", "edgelist", "gz"}
    BASE_DIR = os.path.dirname(__file__)
    DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
    DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
    RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
    DEFAULT_PROPAGATION_PROB = 0.10
    DEFAULT_SIMULATION_RUNS = 20
    DEFAULT_SEED_BUDGET = 5
    DEFAULT_RANDOM_SEED = 42
    DEFAULT_LOUVAIN_RESOLUTION = 1.0
    VISUALIZATION_MAX_NODES = 500
    EXPENSIVE_ALGO_MAX_NODES = 5000
    CELF_MAX_CANDIDATES = 150
    GN_MAX_NODES = 500
    BETWEENNESS_APPROX_THRESHOLD = 1000
    SHORTEST_PATH_MAX_NODES = 2000
    SCALABILITY_SIZES = [500, 1000, 2000, 3000, 4000]
