# 225\2035\115948   0.2532


from dataset.drug_standardization import standardize_smi


"""
    获取蛋白质字典
"""
def get_proteins():

    protein_dict = {}
    with open('/home/wangxuetao/FedKD_DTI/dataset/KIBA/KIBA_protein.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            protein_dict[elements[0]] = elements[1]
    file.close()

    print(len(protein_dict))
    return protein_dict


"""
    获取药物字典
"""
def get_drugs():

    drug_dict = {}
    with open('/home/wangxuetao/FedKD_DTI/dataset/KIBA/KIBA_smiles.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            # 药物数量，标准化前2068，表转化后2035
            if standardize_smi(elements[1]):
                drug = standardize_smi(elements[1])
                drug_dict[elements[0]] = drug
    file.close()

    print(len(drug_dict))
    return drug_dict


"""
    获取DTIs
"""
def get_DTIs(drugs_dict, proteins_dict):

    DTIs = []
    DTIs_set = {}
    DTIs_dcit_by_protein = {}
    with open('/home/wangxuetao/FedKD_DTI/dataset/KIBA/KIBA_DTI.txt', "r") as file:
        lines = file.readlines()
        for line in lines:
            elements = line.strip().split(" ")
            # 判断前117472，判断后115948，有非标准smiles，无重复项
            if elements[0] in drugs_dict and elements[1] in proteins_dict and elements[0] + elements[1] not in DTIs_set:
                # [[drug, protein], 1|0]
                DTIs.append([[drugs_dict[elements[0]], proteins_dict[elements[1]]], elements[2]])
                DTIs_set[elements[0] + elements[1]] = 1

                """
                # 构造以蛋白质为中心的字典
                if elements[1] not in DTIs_dcit_by_protein:
                    DTIs_dcit_by_protein[elements[1]] = []
                    DTIs_dcit_by_protein[elements[1]].append([[drugs_dict[elements[0]], proteins_dict[elements[1]]], elements[2]])
                else:
                    DTIs_dcit_by_protein[elements[1]].append([[drugs_dict[elements[0]], proteins_dict[elements[1]]], elements[2]])
                """

    file.close()

    print(len(DTIs))
    
    #return DTIs, DTIs_dcit_by_protein
    return DTIs
