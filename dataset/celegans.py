# 1767\1876\5931    0.0018

from drug_standardization import standardize_smi

"""
    获取DTIs
"""
def get_DTIs():

    drug_dict = {}
    protein_dict = {}

    DTIs = []
    DTIs_set = {}
    with open('/home/wangxuetao/FedKD_DTI/dataset/Celegans/Celegans_data.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            # 1767
            if elements[0] not in drug_dict:
                drug_dict[elements[0]] = 1
            # 1876
            if elements[1] not in protein_dict:
                protein_dict[elements[1]] = 1
            # 5931
            if elements[0] + elements[1] not in DTIs_set:
                # [[drug, protein], 1|0]
                if standardize_smi(elements[0]):
                    drug = standardize_smi(elements[0])
                    DTIs.append([[drug, elements[1]], elements[2]])
                    DTIs_set[elements[0] + elements[1]] = 1 

    file.close()

    print(len(drug_dict))
    print(len(protein_dict))
    print(len(DTIs))
    return DTIs

