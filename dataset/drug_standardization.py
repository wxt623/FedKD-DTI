from rdkit import Chem
from rdkit.Chem.Descriptors import MolWt

def standardize_smi(smiles):
    # smiles = x.SMILES
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol_weight = MolWt(mol)
        numAtom = mol.GetNumAtoms()
        if mol_weight>900:
            return None
        if numAtom<=3 or mol_weight<50:
            return None
            #set to True 保存立体信息，set to False 移除立体信息，并将分子存为标准化后的SMILES形式\n",
        stan_smiles=Chem.MolToSmiles(mol, isomericSmiles=True,canonical=True) 
    except Exception as e:
        return None
    return stan_smiles