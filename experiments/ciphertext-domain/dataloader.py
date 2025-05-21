import numpy as np
from collections import Counter
from sklearn.decomposition import PCA

def preprocess(embeds_in, dim, minmax):
    pca = PCA(n_components=dim)
    embeds_ = pca.fit_transform(embeds_in)
    if minmax:
        #global minmax
        m = np.min(embeds_)
        M = np.max(embeds_)
        embeds_ = np.array([(embed-m)/(M-m) for embed in embeds_])

        # per feature min max
        # scaler = MinMaxScaler()
        # embeds_ = scaler.fit_transform(embeds_)
    return embeds_, sum(pca.explained_variance_ratio_) * 100

class DataLoader:
    def __init__(self) -> None:
        pass

    def load_embeddings(self, data_path,dimensions=32):
        data = np.load(data_path)
        self.embeds = data["embeddings"]
        self.classes = data["classes"]

        self.preprocessed_embeds, _  = preprocess(self.embeds,dimensions, True)

        assert self.embeds.shape == (13233,512)
        assert np.all((self.preprocessed_embeds >= 0) & (self.preprocessed_embeds <= 1))
        assert np.allclose(np.linalg.norm(self.preprocessed_embeds, axis=1), 1), "Not all embeddings are normalized!"

        self.max_float = np.max(self.preprocessed_embeds)
        self.embed_shape =  self.preprocessed_embeds[0].shape[0]

    def select_people(self, number_imgs, number_of_ppl):
        person_counts = Counter(self.classes)
        eligible_people = [(person, count) for person, count in person_counts.items() if count >= number_imgs]
        selected_people = sorted(eligible_people, key=lambda x: x[1])[:number_of_ppl] 
        selected_labels = {person for person, _ in selected_people}

        data = {}  
        for embed, label in zip(self.preprocessed_embeds, self.classes):
            if label in selected_labels:
                if label not in data:
                    data[label] = []
                if len(data[label]) < number_imgs: 
                    data[label].append(embed)
        return data

    def get_max_float(self):
        return self.max_float

    def get_embed_shape(self):
        return self.embed_shape

    def load_pairs(self, data_path, dimensions, minmax = True):
        data = np.load(data_path)
        pair_embeds = data["embeddings"]
        issame_list = data["issame_list"]

        prepped_embeds,_ = preprocess(pair_embeds, dimensions, minmax)

        assert pair_embeds.shape == (12000,512)
        # assert np.all((prepped_embeds >= 0) & (prepped_embeds <= 1))
        # assert np.all((prepped_embeds >= 0) )
        # assert np.allclose(np.linalg.norm(prepped_embeds, axis=1), 1)

        self.max_float = np.max(prepped_embeds)
        self.embed_shape = prepped_embeds.shape[1]
        embeds1 = prepped_embeds[0::2]
        embeds2 = prepped_embeds[1::2]
        assert embeds1.shape == (6000,dimensions)
        assert embeds2.shape == (6000,dimensions)
        assert issame_list.shape == (6000,)
        return embeds1, embeds2, issame_list



