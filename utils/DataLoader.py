import datasets
class DataLoader:
    def __init__(self, data="hotpot_qa", seed=2023):
        self.data = data
        self.seed = seed

    def load(self, sample_size=100, type="train"):
        if self.data == "hotpot_qa":
            return self.load_hotpot_qa(sample_size=sample_size, type=type)
        elif self.data == "fever":
            return self.load_fever(sample_size=sample_size, type=type)
        elif self.data == "trivia_qa":
            return self.load_trivia_qa(sample_size=sample_size, type=type)
        elif self.data == "gsm8k":
            return self.load_gsm8k(sample_size=sample_size, type=type)
        else:
            raise ValueError("Data not supported.")

    def load_hotpot_qa(self, cache_dir="data/hotpot_qa", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('hotpot_qa', 'fullwiki', cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["question", "answer"]].reset_index(drop=True)
        return sampled_df

    def load_fever(self, cache_dir="data/fever", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('copenlu/fever_gold_evidence', cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["claim", "label"]].reset_index(drop=True)
        return sampled_df

    def load_trivia_qa(self, cache_dir="data/trivia_qa", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('trivia_qa', 'rc.nocontext', cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["question", "answer"]].reset_index(drop=True)
        return sampled_df

    def load_gsm8k(self, cache_dir="data/gsm8k", sample_size=100, type="test"):
        assert type in ["train", "validation", "test"]
        data = datasets.load_dataset('gsm8k', name="main", cache_dir=cache_dir)
        df = data[type].to_pandas()
        sampled_df = df.sample(sample_size, random_state=self.seed)[["question", "answer"]].reset_index(drop=True)
        return sampled_df
