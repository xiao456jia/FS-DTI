
import os
import sys
from FlexMol.dataset.loader import *
from FlexMol.encoder.FM import *
from FlexMol.task import *

esm_pickes_dir = None
chemberta_pickles_dir = None
subpocket_pickles_dir = None
device = 'cuda:0'

epoch = 30
patience = 7
lr = 0.0001
batch_size = 64


task = "davis"
train_split = "data/DAVIS/train_set.txt"
val_split = "data/DAVIS/validation_set.txt"
test_split = "data/DAVIS/test_set.txt"
pdb_dir = "data/DAVIS/DAVIS_pdb/"
subpocket_dir = "data/DAVIS/subpocket/"
metrics_dir = "test_result/DAVIS_drug/"


def load_data(split_path, task):
    """Load the dataset split from the specified path."""
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Dataset split not found at {split_path}")
    print(f"Loading data from {split_path}...")
    if task == "davis":
        return load_DAVIS(split_path)
    elif task == "biosnap":
        return load_BIOSNAP(split_path)
    return None


def main():

    train = load_data(train_split, task)
    val = load_data(val_split, task)
    test = load_data(test_split, task)

    if not os.path.exists(pdb_dir):
        raise FileNotFoundError(f"PDB directory not found at {pdb_dir}")
    print(f"Using PDB files from {pdb_dir}...")

    if not os.path.exists(subpocket_dir):
        raise FileNotFoundError(f"Subpocket directory not found at {subpocket_dir}")
    print(f"Using subpocket files from {subpocket_dir}...")


    FM = FlexMol()
    de = FM.init_drug_encoder("GCN_Chemberta", output_feats=128)
    pe = FM.init_prot_encoder("GCN_ESM", pdb=True, data_dir=pdb_dir, pickle_dir=esm_pickes_dir, output_feats=128, hidden_feats=[128, 128, 128])
    subpocket = FM.init_prot_encoder("Subpocket", pdb=True, pdb_dir=pdb_dir, subpocket_dir=subpocket_dir, pickle_dir=subpocket_pickles_dir)
    fragments=FM.init_drug_encoder("Fragments",fragments=True, output_feats=128)
    dp = FM.set_interaction([de, pe], "cat")
    fo = FM.set_interaction([subpocket, fragments, dp], "f_o_attention")

    FM.build_model()


    trainer = BinaryTrainer(FM, task="DTI", early_stopping="roc-auc", test_metrics=["roc-auc", "pr-auc", "precision", "recall", "f1"],
                            device=device, epochs=epoch, patience=patience, lr=lr, batch_size=batch_size, auto_threshold="max-f1", metrics_dir=metrics_dir)



    train, val, test = trainer.prepare_datasets(train_df=train, val_df=val, test_df=test)
    trainer.train(train, val)

    threshold = trainer.test(val)
    trainer.test(test, threshold=threshold)

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()