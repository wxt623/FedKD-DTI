# 379\68\25772  满秩

from dataset.drug_standardization import standardize_smi

"""
    获取蛋白质字典
"""
def get_proteins():

    protein_dict = {}
    with open('/home/wangxuetao/FedKD_DTI/dataset/Davis/Davis_protein.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            protein_dict[elements[0]] = elements[1]
    file.close()

    # 379
    print(len(protein_dict))
    return protein_dict


"""
    获取药物字典
"""
def get_drugs():

    drug_dict = {}
    with open('/home/wangxuetao/FedKD_DTI/dataset/Davis/Davis_smiles.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            # 药物数量，标准化前2068，表转化后2035
            if standardize_smi(elements[1]):
                drug = standardize_smi(elements[1])
                drug_dict[elements[0]] = drug
    file.close()

    # 68
    print(len(drug_dict))
    return drug_dict


"""
    获取DTIs
"""
def get_DTIs(drugs_dict, proteins_dict):

    DTIs = []
    DTIs_set = {}
    with open('/home/wangxuetao/FedKD_DTI/dataset/Davis/Davis_DTI.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            if elements[0] in drugs_dict and elements[1] in proteins_dict and elements[0] + elements[1] not in DTIs_set:
                # [[drug, protein], 1|0]
                DTIs.append([[drugs_dict[elements[0]], proteins_dict[elements[1]]], elements[2]])
                DTIs_set[elements[0] + elements[1]] = 1

    file.close()

    # 25772（满秩矩阵，DTIs矩阵内所有元素都存在）
    print(len(DTIs))
    return DTIs


"""
drugs_dict = get_drugs()
proteins_dict = get_proteins()
DTIs = get_DTIs(drugs_dict, proteins_dict)
print(DTIs[0])
"""