import numpy as np
from collections import Counter
from src.helpers import preprocess
class DataLoader:
    """
    DataLoader class for loading, preprocessing, and selecting embeddings.

    Methods
    -------
    load_embeddings(data_path, dimensions=32, minmax=True):
        Loads embeddings from a given path, applies preprocessing

    select_people(number_imgs, number_of_ppl):
        Selects a specified number of people with a specified number of images.

    get_max_float():
        Returns the maximum float value in the preprocessed embeddings.

    get_embed_shape():
        Returns the dimensionality of the processed embeddings.

    load_pairs(data_path, dimensions=32, minmax=True):
        Loads and processes embedding pairs for similarity evaluation.
    """
    def __init__(self) -> None:
        pass

    def load_embeddings(self, data_path,dimensions=32):
        """
        Load and preprocess embeddings from a .npz file.

        Parameters
        ----------
        data_path : str
            Path to the .npz file containing 'embeddings' and 'classes'.
        dimensions : int, optional (default=32)
            Number of PCA dimensions to reduce the embeddings to.

        Raises
        ------
        AssertionError:
            If original embeddings do not have the expected shape (13233,512).
            If embeddings are not normalized between 0 and 1.
            If embeddings do not have a norm of 1.
        """
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
        """
        Selects a specified number of people who have a specified number of images.

        Parameters
        ----------
        number_imgs : int
            Number of images per person.
        number_of_ppl : int
            Number of people to select.

        Returns
        -------
        dict
            Dictionary mapping selected labels to their corresponding embeddings.
        """
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
        """
        Get the maximum float value resulting from preprocessing.

        Returns
        -------
        float
            Maximum value in the preprocessed embeddings.
        """
        return self.max_float

    def get_embed_shape(self):
        """
        Get the dimensionality of a single processed embedding.

        Returns
        -------
        int
            The number of dimensions in each embedding.
        """
        return self.embed_shape

    def load_pairs(self, data_path, dimensions=32):
        """
        Used for testing.
        Load and preprocess pairs of embeddings for similarity evaluation.

        Parameters
        ----------
        data_path : str
            Path to the .npz file containing 'embeddings' and 'issame_list'.
        dimensions : int, optional (default=32)
            Number of PCA dimensions for embeddings.

        Returns
        -------
        tuple:
            - embeds1 (np.ndarray): First element embeddings from each pair.
            - embeds2 (np.ndarray): Second element embeddings from each pair.
            - issame_list (list): Boolean list indicating if pairs are from the same class.

        Raises
        ------
        AssertionError:
            If embeddings are not correctly shaped or normalized.
        """
        data = np.load(data_path)
        pair_embeds = data["embeddings"]
        issame_list = data["issame_list"]

        prepped_embeds,_ = preprocess(pair_embeds, dimensions, True)

        assert pair_embeds.shape == (12000,512)
        assert np.all((prepped_embeds >= 0) & (prepped_embeds <= 1))

        self.max_float = np.max(prepped_embeds)
        self.embed_shape = prepped_embeds.shape[1]
        embeds1 = prepped_embeds[0::2]
        embeds2 = prepped_embeds[1::2]
        assert embeds1.shape == (6000,dimensions)
        assert embeds2.shape == (6000,dimensions)
        assert issame_list.shape == (6000,)
        return embeds1, embeds2, issame_list



