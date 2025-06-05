import bz2
import pickle

with bz2.BZ2File(r"C:\Users\mfaiz\Documents\Capstone\random_forest_model.pbz2", "rb") as f:
    model = pickle.load(f)
