import sentencepiece as spm

input_file = "all_prompts_new2.txt"
vocab_size = 32000

spm.SentencePieceTrainer.Train(
    f"--input={input_file} "
    f"--model_prefix=my_spm_new2 "
    f"--vocab_size={vocab_size} "
    "--model_type=UNIGRAM "
    "--unk_id=1 --pad_id=0 --bos_id=2 --eos_id=3 "
    "--user_defined_symbols=<SOS>,<EOS>,<PAD> "
    "--input_sentence_size=1000000 "
    "--shuffle_input_sentence=true"
)
