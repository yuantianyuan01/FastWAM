import re

from mujoco import MjModel


def get_child_body_ids(model: MjModel, parent_id: int) -> list[int]:
    # Get all body parent IDs
    parent_ids = model.body_parentid  # numpy array of length nbody
    # Find indices where parent is the given ID
    return [i for i, pid in enumerate(parent_ids) if pid == parent_id]


def get_child_body_names(model: MjModel, parent_id: int) -> list[str]:
    child_ids = get_child_body_ids(model, parent_id)
    return [model.body(i).name for i in child_ids]


# Example procthor names:
#  Knife|surface|2|0_Knife_3_Cube.013 -> Knife_3
#  Mug|surface|8|93_Mug_2_mug_2 -> Mug_2
#  SoapBottle|surface|2|10_Soap_Bottle_5_soap_bottle_5  -> Soap_Bottle_5
#  KeyChain|surface|4|40_Keychain_2_KeyChain2 -> Keychain_2
#  CellPhone|surface|7|75_Cellphone_8_cellphone_4 -> Cellphone_8
#  Vase|surface|3|33_Vase_Open_1_vase_open _, Vase_Open_1
def get_thor_name(model, pickup_obj):
    child_names = get_child_body_names(model, pickup_obj.object_id)
    object_name = child_names[0] if len(child_names) > 0 else pickup_obj.name

    if "|" in object_name:  # name from proctor
        name_end = object_name.split("|")[-1]
        match = re.search(r"^\d+_([A-Za-z_]+[\d_]+)", name_end)
        return match.group(1).strip("_")
    else:  # name from ithor
        name_end = object_name.replace(pickup_obj.name + "_", "")
        match = re.search(r"^([A-Za-z_]+[\d_]+)", name_end)
        return match.group(1).strip("_")
