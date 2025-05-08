import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_path="my_spm.model"):
        # Load the trained SentencePiece model
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)

        # Keep track of vocab size
        self.vocab_size = self.sp.GetPieceSize()

        # If you want to ensure you know which IDs map to which special tokens:
        # NOTE: This is optional. You can also rely on sp.IdToPiece() as needed.
        self.pad_id = self.sp.PieceToId("<PAD>")  # Should be 0 if you forced --pad_id=0
        self.unk_id = self.sp.PieceToId("<UNK>")  # Should be 1 if you forced --unk_id=1
        self.bos_id = self.sp.PieceToId("<SOS>")  # Should be 2 if you forced --bos_id=2
        self.eos_id = self.sp.PieceToId("<EOS>")  # Should be 3 if you forced --eos_id=3

    def encode(self, text):
        """
        Convert a text string into a list of integer IDs.
        If you want an explicit BOS/EOS included, you can manually add them.
        """
        # Example: add a BOS token, then the tokenized pieces, then an EOS
        pieces = [self.bos_id] + self.sp.EncodeAsIds(text) + [self.eos_id]
        return pieces

    def decode(self, ids):
        """
        Convert a list of integer IDs back into text.
        You may want to remove BOS/EOS (and any other special tokens) first.
        """
        # For example, strip out BOS/EOS (2,3):
        filtered_ids = [i for i in ids if i not in [self.bos_id, self.eos_id, self.pad_id]]
        return self.sp.DecodeIds(filtered_ids)
