import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from .tokenize import Tokenizer, PAD_TOKEN


class SLPDataset(Dataset):
    def __init__(
            self,
            folder_path: str = None,
            file_path: str = None,
            skel_field: str = 'poses_3d',
            text_field: str = 'text',
            gloss_field: str = 'gloss',
            id_field: str = "id",
            extra_fields: List[str] = None,
            load_only_skel: bool = False,
            skip_frames: int = 1,
            max_sent_len: int = None,
            text_tokenizer: Tokenizer = None,
            gloss_tokenizer: Tokenizer = None,
    ):
        """
        FOLDER INPUT
        ----------------------------------------------------------------------------------
        If folder_path is not None, we expect a folder containing one .pt file per sample.
        Each file typically contains the following keys:
            - "id": example name (str)
            - "poses_3d": torch.Tensor of shape (T, 178, 3) with the 3D coordinates of 178 joints (8 body, 21 left hand, 21 right hand, 128 face)
            - "raw_text": raw text annotation (str) [OPTIONAL]
            - "text": text annotations as lemmas (str)
            - "gloss": glosses annotations (str) [OPTIONAL]

        FILE INPUT
        ----------------------------------------------------------------------------------
        If file_path is not None, we expect one .pt file containing a dict of dicts (one per sample).
        The form of file's content is:
            {
                EX_ID_1: {
                            "id" OR "name": ...,
                            "poses_3d": ...,
                            "text": ...,
                            "gloss": ..., [OPTIONAL]
                            "speaker": ..., [OPTIONAL]
                        },
                ...
            }
        """
        super().__init__()

        # -- path
        self.folder_path = folder_path
        self.file_path = file_path

        # -- prepro params
        self.skip_frames = skip_frames
        self.max_sent_len = max_sent_len
        self.text_tokenizer = text_tokenizer
        self.gloss_tokenizer = gloss_tokenizer

        # -- fields
        self.skel_field = skel_field
        self.id_field = id_field
        self.text_field = text_field
        self.gloss_field = gloss_field
        if load_only_skel:
            self.text_field = None
            self.gloss_field = None
        self.load_only_skel = load_only_skel
        self.extra_fields = []
        if extra_fields is not None:
            self.extra_fields = extra_fields

        # -- load samples ids & data if file + only keep one for len(text)<=max_sent_len if provided
        self.data = None
        self.ids = []
        if folder_path:
            all_files = [f for f in os.listdir(folder_path) if (os.path.isfile(os.path.join(folder_path, f)) and f.split(".")[-1] == "pt")]
            all_ids = [Path(f).stem for f in all_files]
            if max_sent_len:
                for f in tqdm(all_files, desc=f"Retrieving IDs for which len(sentence) < max_sent_len={max_sent_len}"):
                    example = torch.load(os.path.join(folder_path, f))
                    text = example[text_field].split()
                    if len(text) <= max_sent_len:
                        self.ids.append(Path(f).stem)
            else:
                self.ids = all_ids
        elif file_path:
            data = torch.load(file_path)
            all_ids = list(data.keys())
            if max_sent_len:
                for k in tqdm(all_ids, desc=f"Retrieving IDs for which len(sentence) < max_sent_len={max_sent_len}"):
                    text = data[k][text_field].split()
                    if len(text) <= max_sent_len:
                        self.ids.append(k)
            else:
                self.ids = all_ids
            self.data = data
        else:
            raise ValueError("Must provide either file or folder path to data")
        assert len(self.ids) > 0

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        if self.folder_path:
            example = torch.load(os.path.join(self.folder_path, self.ids[idx]))
        elif self.file_path:
            example = self.data[self.ids[idx]]
        else:
            return None

        ex_id: str = example[self.id_field]
        skels: torch.Tensor = example[self.skel_field][::self.skip_frames]  # shape=(T, Npts, 3)
        ex = {"id": ex_id, "skels": skels}

        if self.text_field:
            txt = example[self.text_field]
            txt = txt.lower()
            txt_tokens = self.text_tokenizer.tokenize(txt)
            txt_tokens_ids = self.text_tokenizer.tokens_to_ids(txt_tokens)
            ex["text"] = txt_tokens_ids
        if self.gloss_field:
            gloss = example[self.gloss_field]
            gloss = gloss.lower()
            gloss_tokens = self.gloss_tokenizer.tokenize(gloss)
            gloss_tokens_ids = self.gloss_tokenizer.tokens_to_ids(gloss_tokens)
            ex["gloss"] = gloss_tokens_ids

        # for field in self.extra_fields:
        #     ex[field] = example[field]

        return ex


def build_slp_dataloader(
        dataset: SLPDataset,
        text_vocab: Dict[str, int] = None,
        gloss_vocab: Dict[str, int] = None,
        fixed_seq_len: int = None,  # fixed number of frames
        pad_value: float = 0.0,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 4,
) -> DataLoader:
    def collate_fn(batch):
        batch_ids = [ex["id"] for ex in batch]  # list of str (ids)
        if fixed_seq_len:
            batch_skels = []
            for ex in batch:
                skels = ex["skels"]  # (T_seq, Npts, 3)
                skels = skels[:fixed_seq_len]
                pad_len = fixed_seq_len - len(skels)
                # for padding we give len. of padding for right and left for each dimension STARTING FROM THE LAST!!
                pd = (0, 0, 0, 0, 0, pad_len)
                skels = F.pad(skels, pd, value=pad_value)
                batch_skels.append(skels)
            batch_skels_tensor = torch.stack(batch_skels, dim=0)
        else:
            batch_skels = [ex["skels"] for ex in batch]  # B tensors of shape (T_seq, Npts, 3)
            batch_skels_tensor = pad_sequence(batch_skels, batch_first=True, padding_value=pad_value)

        ex_batch = {"id": batch_ids, "skels": batch_skels_tensor}

        if dataset.text_field:
            batch_text = [ex["text"] for ex in batch]
            max_len = max(len(s) for s in batch_text)

            pad_id = getattr(dataset.text_tokenizer, "pad_id", None)
            if pad_id is None:
                if text_vocab is not None:
                    pad_id = text_vocab[PAD_TOKEN]
                else:
                    raise ValueError("Tokenizer does not have pad ID and no fallback vocab was given for text")

            batch_text = [s + [pad_id] * (max_len - len(s)) for s in batch_text]

            batch_text_tensor = torch.tensor(batch_text, dtype=torch.long)
            ex_batch["text"] = batch_text_tensor

        if dataset.gloss_field:
            batch_gloss = [ex["gloss"] for ex in batch]
            max_len = max(len(s) for s in batch_gloss)

            pad_id = getattr(dataset.gloss_tokenizer, "pad_id", None)
            if pad_id is None:
                if gloss_vocab is not None:
                    pad_id = gloss_vocab[PAD_TOKEN]
                else:
                    raise ValueError("Tokenizer does not have pad ID and no fallback vocab was given for gloss")

            batch_gloss = [s + [pad_id] * (max_len - len(s)) for s in batch_gloss]

            batch_gloss_tensor = torch.tensor(batch_gloss, dtype=torch.long)
            ex_batch["gloss"] = batch_gloss_tensor

        return ex_batch

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )

    return dataloader
