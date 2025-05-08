import sentencepiece as spm

input_file = "all_prompts.txt"
vocab_size = 5000


spm.SentencePieceTrainer.Train(
    f"--input={input_file} --model_prefix=my_spm --vocab_size={vocab_size} "
    "--unk_id=1 --pad_id=0 --bos_id=2 --eos_id=3 "
    "--user_defined_symbols=<SOS>,<EOS>,<PAD>"
)