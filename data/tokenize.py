from abc import ABC, abstractmethod, abstractproperty
# from transformers import AutoTokenizer


PAD_TOKEN = '<pad>'

class Tokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens: list[str]) -> list[int]:
        pass

    @property
    @abstractmethod
    def pad_id(self):
        pass

# class HFTokenizer(Tokenizer):
#     """
#     Uses HF transformers library AutoTokenizer to instantiate the tokenizer
#     associated to the pre-trained HF BERT-like model `name`.
#     """
#     def __init__(self, name: str):
#         super().__init__()
#         self.hf_tokenizer = AutoTokenizer.from_pretrained(name)
#
#         # special tokens (<s> and </s>)
#         self.cls_token = self.hf_tokenizer.cls_token
#         self.sep_token = self.hf_tokenizer.sep_token
#
#     def tokenize(self, text: str, add_special_tokens: bool = True) -> list[str]:
#         if add_special_tokens:
#             return [self.cls_token] + self.hf_tokenizer.tokenize(text) + [self.sep_token]
#         else:
#             return self.hf_tokenizer.tokenize(text)
#
#     def tokens_to_ids(self, tokens: list[str]) -> list[int]:
#         return self.hf_tokenizer.convert_tokens_to_ids(tokens)
#
#     @property
#     def pad_id(self) -> int:
#         return self.hf_tokenizer.pad_token_id


# def get_tokenizer(name: str, from_hf_library: bool = True) -> Tokenizer:
#     if from_hf_library:
#         return HFTokenizer(name)
#     else:
#         # todo: other cases not implemented yet - must implement in the future
#         raise ValueError("For now, only HF library tokenizer are supported")
