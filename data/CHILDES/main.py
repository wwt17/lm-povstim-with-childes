from CHILDES_xml_Processing import process_childes_xml
from CHILDES_txt_Processing import clean_and_unk
from CHILDES_Treebank_Processing import process_childes_treebank
from CHILDES_Treebank_txt_Processing import split_treebank
from pathlib import Path
import argparse


def main(
        pretraining_dir = Path('pretraining'),
        finetuning_dir = Path('finetuning'),
        splitting = False,
        unking = False,
        cutoff = 0,
        seed = 1,
):
    raw_dataset = process_childes_xml("./", "childes-xml", splitting=splitting, seed=seed)
    dataset, vocab = clean_and_unk(raw_dataset, unking=unking, cutoff=cutoff)
    pretraining_dir.mkdir(exist_ok=True)
    for split, data in dataset.items():
        with open(pretraining_dir/f'{split}.txt', 'w') as f:
            f.write("\n".join((' '.join(s) for filepath, s in data)))
        with open(pretraining_dir/f'{split}.map', 'w') as f:
            f.write("\n".join((filepath + '\t' + ' '.join(s) for filepath, s in data)))
    with open(pretraining_dir/'vocab.txt', 'w') as f:
        f.write("\n".join(vocab))

    if finetuning_dir is not None:
        decl, quest = process_childes_treebank("childes-treebank")
        with open(pretraining_dir/'excluded.txt') as f:
            excluded = f.read()
        finetuning_dataset = split_treebank(excluded, decl, quest)
        finetuning_dir.mkdir(exist_ok=True)
        for split, data in finetuning_dataset.items():
            with open(finetuning_dir/f'{split}.txt', 'w') as f:
                f.write("".join(data))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--pretraining_dir", type=Path, default=Path('pretraining'))
    argparser.add_argument("--finetuning_dir", type=Path)
    argparser.add_argument("--splitting", action="store_true")
    argparser.add_argument("--unking", action="store_true")
    argparser.add_argument("--cutoff", type=int, default=0)
    argparser.add_argument("--seed", type=int, default=1)
    args = argparser.parse_args()
    main(
        pretraining_dir = args.pretraining_dir,
        finetuning_dir = args.finetuning_dir,
        splitting = args.splitting,
        unking = args.unking,
        seed = args.seed,
    )