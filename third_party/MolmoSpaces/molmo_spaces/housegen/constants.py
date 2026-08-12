VISUAL_CLASS: str = "__VISUAL_MJT__"
DYNAMIC_CLASS: str = "__DYNAMIC_MJT__"
ARTICULABLE_DYNAMIC_CLASS: str = "__ARTICULABLE_DYNAMIC_MJT__"
STRUCTURAL_CLASS: str = "__STRUCTURAL_MJT__"
STRUCTURAL_WALL_CLASS = "__STRUCTURAL_WALL_MJT__"

# TODO(wilbert): this value is a bit too high, and makes the simulation looks weird, like moving
# on viscous
FREE_JOINT_DAMPING: float = 0.001
FREE_JOINT_FRICTIONLOSS: float = 0.001

DYNAMIC_OBJ_GEOMS_MARGIN = 0.001
DEFAULT_SETTLE_TIME = 20.0

TYPES_TO_REMOVE_ALL_JOINTS: set[str] = {
    "LightSwitch",
    "StoveKnob",
    "Bathroom_Faucet",
    "BathTub_Faucet",
    "Faucet",
    "ShowerHead",
    "Book",
}

CATEGORIES_FROZEN_IN_SPACE: set[str] = {
    "Painting",
    "CounterTop",
    "Window",
    "WallPanel",
    "Television",
    "Toilet",
    "ToiletPaperHanger",
    "Sink",
    "Fridge",
    "Bathroom_Faucet",
    "BathTub_Faucet",
    "Stove",
    "Faucet",
    "Bed",
    # "Desk",
    "Cabinet",
    "Drawer",
    "StoveKnob",
    "Light",
    "LightSwitch",
    "Blinds",
    "Oven",
    "Dishwasher",
    "ShowerDoor",
    "ShowerHead",
}

ASSETS_FROZEN_IN_SPACE: list[str] = ["painting", "wall", "window", "doorway", "doorframe"]

# TODO(wilbert): shouldn't we check with the asset_id instead of the object_id?
TYPES_TO_USE_ONLY_PRIM: list[str] = [
    # too thin
    "pen",
    "pencil",
    "plate",
    "key",
    "cd",
    "book",
    "phone",
    "card",
    "bedsheet",  # NOTE(wilbert): it seems this one is not part of any object_id or asset_id
    "spoon",
    "fork",
    "remote",  # NOTE(wilbert): mesh version shows physics instability
    # boxy objects that are better prim description
    "laptop",
    "box",
    "statue",  # some have very curvy bottom that it cannot stand
    "lamp",  # not likely to pickup...
    "light_switch",  # TODO(yejin): for now, use prim only for lightswitch
    "houseplant",
    "stool",
    "garbagebin",  # NOTE(wilbert): some object ids have 'garbagebin' instead of 'bin'
    "garbagecan",  # NOTE(wilbert): some object ids have 'garbagecan' instead of 'bin'
    "window",
    # furntiure with receptacles with objects on top or inside
    "bed",
    "shelving",
    "table",  # covers sidetable, dining table, coffee table, too
    "dresser",
    "desk",
    "countertop",
    "cabinet",
    "bin",  # NOTE(wilbert): some object ids have 'bin' instead of 'garbagecan' or 'garbagebin'
    "chair",
    "painting",  # NOTE(wilbert): these are static and not interactable, so no need for mesh I think
    "stand",
    # "television", # NOTE(wilbert): could be prim, as it's fixed and not interactable
    # "door", # NOTE(wilbert): could be prim, but requires to repair its colliders
    "room_decor",  # NOTE(wilbert): these are failing the stability test, so make them prim
    "tennis",  # NOTE(wilbert): we can make this prim, as won't be interacted with as it's big
    "pan",
    "watch",
    "basketball",
    "candle",
    "dish_sponge",  # NOTE(wilbert): could use prim, as the shape is pretty much a box
]

VALID_VISUAL_CLASSES: set[str] = {VISUAL_CLASS, "visual"}

VALID_DYNAMIC_CLASSES: set[str] = {
    DYNAMIC_CLASS,
    ARTICULABLE_DYNAMIC_CLASS,
    STRUCTURAL_CLASS,
    STRUCTURAL_WALL_CLASS,
    "collision",
}

ARTICULABLE_DYNAMIC_CATEGORIES: set[str] = {
    "oven",
    "dishwasher",
    "cabinet",
    "drawer",
    "showerdoor",
    "door",
    "fridge",
}

ITHOR_HOUSES_FLOOR_OFFSET: dict[str, float] = {"FloorPlan407_physics": 0.043}

THOR_OBJECT_MASSES: dict[str, float] = {
    "Sofa": 60,
    "Fridge": 80,
    "WashingMachine": 75,
    "ClothesDryer": 60,
    "Bed": 50,
    "Dresser": 40,
    "ShelvingUnit": 30,
    "DiningTable": 29,
    "Desk": 26,
    "TVStand": 20,
    "CoffeeTable": 14,
    "ArmChair": 18,
    "Chair": 8,
    "Stool": 5,
    "Footstool": 5,
    "Ottoman": 5,
    "SideTable": 6,
    "LaundryHamper": 2,
    "GarbageCan": 1.5,
    "DogBed": 1.5,
    "Television": 10,
    "VacuumCleaner": 6,
    "Microwave": 10,
    "CoffeeMachine": 3,
    "Laptop": 1.5,
    "Desktop": 7,
    "Safe": 20,
    "FloorLamp": 5,
    "DeskLamp": 2,
    "Sink": 20,
    "Toilet": 25,
    "Faucet": 2,
    "ShowerHead": 0.5,
    "ToiletPaperHanger": 0.3,
    "HandTowelHolder": 0.5,
    "TowelHolder": 0.5,
    "LightSwitch": 0.1,
    "Pan": 1.2,
    "Pot": 1.5,
    "Kettle": 1,
    "Toaster": 2,
    "Bowl": 0.4,
    "Plate": 0.7,
    "Mug": 0.35,
    "Cup": 0.2,
    "Ladle": 0.2,
    "Fork": 0.05,
    "Knife": 0.1,
    "ButterKnife": 0.05,
    "Spoon": 0.05,
    "SaltShaker": 0.2,
    "PepperShaker": 0.2,
    "AluminumFoil": 0.1,
    "CellPhone": 0.2,
    "RemoteControl": 0.1,
    "AlarmClock": 0.5,
    "Watch": 0.1,
    "CD": 0.02,
    "SoapBottle": 0.5,
    "SoapBar": 0.15,
    "Plunger": 0.8,
    "ScrubBrush": 0.3,
    "DishSponge": 0.01,
    "SprayBottle": 0.5,
    "TissueBox": 0.3,
    "ToiletPaper": 0.1,
    "PaperTowelRoll": 0.5,
    "GarbageBag": 0.5,
    "Book": 0.5,
    "Pencil": 0.02,
    "Pen": 0.02,
    "CreditCard": 0.02,
    "KeyChain": 0.05,
    "Bottle": 0.5,
    "WineBottle": 1.2,
    "Box": 0.5,
    "Newspaper": 0.3,
    "TeddyBear": 0.5,
    "Blinds": 2,
    "ShowerCurtain": 1,
    "Towel": 0.6,
    "HandTowel": 0.15,
    "Cloth": 0.05,
    "Vase": 1,
    "Candle": 0.5,
    "Painting": 1,
    "Statue": 3,
    "RoomDecor": 0.5,
    "TableTopDecor": 0.3,
    "Potato": 0.30,
    "Tomato": 0.15,
    "Apple": 0.15,
    "Bread": 0.5,
    "Egg": 0.06,
    "Lettuce": 0.4,
    "Basketball": 0.6,
    "TennisRacket": 0.3,
    "BaseballBat": 1,
    "Dumbbell": 5,
    "Boots": 1.5,
    "WateringCan": 0.5,
    "HousePlant": 3,
    "Cart": 10,
}
