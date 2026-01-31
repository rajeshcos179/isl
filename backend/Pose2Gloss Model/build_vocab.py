# build_vocab.py
import pandas as pd

def build_vocab(csv_path):
    df = pd.read_csv(csv_path)
    glosses = sorted(df['gloss'].unique())
    gloss2id = {g:i for i,g in enumerate(glosses)}
    id2gloss = {i:g for g,i in gloss2id.items()}
    return gloss2id, id2gloss