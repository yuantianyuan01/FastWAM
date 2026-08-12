import VLABench.tasks.components as components
import os

xml_root = os.path.join(os.getenv("VLABENCH_ROOT"), "assets")

def get_object_list(xml_dir, all=True, seen=True):
    """
    Get all the xml files under the xml_dir
    Parameters:
      - xml_dir: the root dir to search for xml files
      - all: whether to return all the xml files
      - seen: if not return all the files, if True, return seen objects in traning set, else return unseen objects for unseen evaluation
    Return:
        a list of split paths of xml files
    """
    xml_paths = []
    subdirs = os.listdir(xml_dir)
    for subdir in subdirs:
        if subdir.endswith(".xml"):
            xml_paths.append(os.path.join(xml_dir, subdir))
        elif os.path.isdir(os.path.join(xml_dir, subdir)):
            xml_paths.extend(get_object_list(os.path.join(xml_dir, subdir)))
    if all: xml_paths = xml_paths
    else:
        if seen: xml_paths = xml_paths[:len(xml_paths)//2]
        else: xml_paths = xml_paths[len(xml_paths)//2:]
    split_xml_paths = [xml_path.split("assets/")[-1] for xml_path in xml_paths]
    return sorted(split_xml_paths)

# name2config = {
#     "select_mahjong_series": ["select_mahjong", "select_mahjong_spatial", "select_unique_type_mahjong", "select_mahjong_semantic"],
#     "select_poker_series": ["select_poker", "select_poker_spatial", "select_nth_largest_poker", "select_poker_semantic"],
#     "select_chemistry_tube_series": ["select_chemistry_tube", "select_chemistry_tube_spatial", "select_chemistry_tube_common_sense", "select_chemistry_tube_semantic"],
#     "select_fruit_series": ["select_fruit", "select_fruit_spatial", "select_fruit_common_sense", "select_fruit_semantic"],
#     "add_condiment_series": ["add_condiment", "add_condiment_spatial", "add_condiment_common_sense", "add_condiment_semantic"],
#     "insert_flower_series": ["insert_flower", "insert_flower_spatial", "insert_flower_common_sense", "insert_flower_semantic", "insert_bloom_flower"],
#     "select_book_series": ["select_book", "select_book_spatial", "select_specific_type_book", "select_book_semantic"],
#     "select_billiards_series": ["select_billiards", "select_billiards_spatial", "select_billiards_common_sense", "select_billiards_semantic"],
#     "select_drink_series": ["select_drink", "select_drink_spatial", "select_drink_common_sense", "select_drink_semantic"],
#     "select_toy_series": ["select_toy", "select_toy_spatial", "select_toy_common_sense", "select_toy_semantic"],
#     "select_ingredient_series": ["select_ingredient", "select_ingredient_spatial", "select_ingredient_common_sense", "select_ingredient_semantic"],
#     "select_painting_series":["select_painting", "put_box_on_painting", "select_painting_semantic"],
#     # composite tasks
#     "texas_holdem_series":["texas_holdem", "texas_holdem_explore"],
#     "set_dining_table_series":["set_dining_table", "set_dining_left_hand", "set_dining_chopstick"],
#     "cool_drink_series":["cool_drink", "take_out_cool_drink"],
#     "heat_food_series":["heat_food"],
#     "get_coffee_series":["get_coffee", "get_coffee_with_sugar", "get_coffee_with_milk"],
#     "hammer_nail_series":["hammer_loose_nail", "assemble_hammer", "hammer_nail_and_hang_picture"],
#     "make_juice_series":["make_juice"],
#     "seesaw_series":["simple_seesaw_use", "complex_seesaw_use"],
#     "set_study_table_series":["set_study_table"],
# }

name2class_xml = {
    # containers/receptacles
    "basket":[components.CommonContainer, get_object_list(os.path.join(xml_root, "obj/meshes/containers/basket"))],
    "shelf": [components.Shelf, get_object_list(os.path.join(xml_root, "obj/meshes/containers/shelf"))],
    "placemat":[components.PlaceMat, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/placemats"))],
    "cut_board":[components.CuttingBoard, get_object_list(os.path.join(xml_root, "obj/meshes/containers/cutting_board"))],
    "big_fridge":[components.Fridge, get_object_list(os.path.join(xml_root, "obj/meshes/containers/fridge/big_fridge"))],
    "small_fridge":[components.Fridge, get_object_list(os.path.join(xml_root, "obj/meshes/containers/fridge/small_fridge"))],
    "fridge_open":[components.Fridge, get_object_list(os.path.join(xml_root, "obj/meshes/containers/fridge/fridge_open"))],
    "cabinet":[components.ContainerWithDrawer, get_object_list(os.path.join(xml_root, "obj/meshes/containers/cabinets"))],
    "tray":[components.FlatContainer, get_object_list(os.path.join(xml_root, "obj/meshes/containers/tray"))],
    "microwave":[components.Microwave, get_object_list(os.path.join(xml_root, "obj/meshes/containers/microwaves"))],
    "coffee_machine":[components.CoffeeMachine, get_object_list(os.path.join(xml_root, "obj/meshes/coffee_machines"))],
    "vase":[components.Vase, get_object_list(os.path.join(xml_root, "obj/meshes/containers/vases"))],    
    "billiards_table":[components.BilliardTable, get_object_list(os.path.join(xml_root, "obj/meshes/containers/billiards_table"))],
    "giftbox": [components.CommonContainer, get_object_list(os.path.join(xml_root, "obj/meshes/containers/boxes/giftbox"))],

    # objects with multiple meaning textures
    "painting": [components.Painting, "obj/meshes/paintings/painting.xml"],
    "poker": [components.Poker, "obj/meshes/poker/poker_asset.xml"],
    "mahjong": [components.Mahjong, "obj/meshes/mahjong/mahjong.xml"],
    "billiards": [components.BilliardBall, "obj/meshes/billiard_balls/billiards.xml"],
    "number_cube":[components.NumberCube, "obj/meshes/number_cube/numbercube.xml"],
    
    # flowers
    "wilted_flower": [components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/wilted_flower"))],
    "bloom_flower": [components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower"))],
    "rose":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower/rose"))],
    "magnolia":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower/magnolia"))],
    "tulip":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower/tulip"))],
    "sunflower":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower/sunflower"))],
    "chrysanthemum":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower/chrysanthemum"))],
    "peony":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower/peony"))],
    "daisy_flower":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/flowers/bloom_flower/daisy"))],
    
    # fruits
    "fruit": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit"))],
    "apple": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/apple"))],
    "avocado": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/avocado"))],
    "banana": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/banana"))],
    "kiwi": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/kiwi"))],
    "lemon": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/lemon"))],
    "mango": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/mango"))],
    "orange": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/orange"))],
    "peach": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/peach"))],
    "pear": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/pear"))],
    "strawberry": [components.Fruit, get_object_list(os.path.join(xml_root, "obj/meshes/fruit/strawberry"))],
    
    # ingredients
    "ingredient": [components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients"))],
    "cooked_food": [components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/cooked_food"))],
    "bell_pepper":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/bell_pepper"))],
    "broccoli":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/broccoli"))],
    "carrot":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/carrot"))],
    "cheese":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/cheese"))],
    "corn":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/corn"))],
    "egg":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/egg"))],
    "eggplant":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/eggplant"))],
    "fish":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/fish"))],
    "garlic":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/garlic"))],
    "mushroom":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/mushroom"))],
    "onion":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/onion"))],
    "potato":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/potato"))],
    "steak":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/steak"))],
    "sweet_potato":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/sweet_potato"))],
    "tomato":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/tomato"))],
    "cucumber":[components.Ingredient, get_object_list(os.path.join(xml_root, "obj/meshes/ingredients/cucumber"))],
    "dishes":[components.FlatContainer, get_object_list(os.path.join(xml_root, "obj/meshes/dishes"))],
    # snacks
    "snack": [components.Snack, get_object_list(os.path.join(xml_root, "obj/meshes/snacks"))],
    "bagged_food": [components.Snack, get_object_list(os.path.join(xml_root, "obj/meshes/snacks/bagged_food"))],
    "bar": [components.Snack, get_object_list(os.path.join(xml_root, "obj/meshes/snacks/bar"))],
    "boxed_food": [components.Snack, get_object_list(os.path.join(xml_root, "obj/meshes/snacks/boxed_food"))],
    "canned_food": [components.Snack, get_object_list(os.path.join(xml_root, "obj/meshes/snacks/canned_food"))],
    "chips": [components.Snack, get_object_list(os.path.join(xml_root, "obj/meshes/snacks/chips"))],
    "chocolate": [components.Snack, get_object_list(os.path.join(xml_root, "obj/meshes/snacks/chocolate"))],
    
    # chemistry objects
    "tube": [components.ChemistryTube, "obj/meshes/tube/tube/tube.xml"],
    "chemistry_tube_stand": [components.TubeStand, "obj/meshes/tube/tube_container/tube_stand.xml"],
    "nametag": [components.NameTag, "obj/meshes/tube/nametag/nametag.xml"],
    "flask": [components.CommonContainer, "obj/meshes/tube/flask/flask.xml"],
    
    # breads
    "bagel": [components.Bread, get_object_list(os.path.join(xml_root, "obj/meshes/breads/bagel"))],
    "baguette": [components.Bread, get_object_list(os.path.join(xml_root, "obj/meshes/breads/baguette"))],
    "bread": [components.Bread, get_object_list(os.path.join(xml_root, "obj/meshes/breads/bread"))],
    "croissant": [components.Bread, get_object_list(os.path.join(xml_root, "obj/meshes/breads/croissant"))],
    "hot_dog": [components.Bread, get_object_list(os.path.join(xml_root, "obj/meshes/breads/hot_dog"))],
    
    # desserts
    "cake": [components.Dessert, get_object_list(os.path.join(xml_root, "obj/meshes/dessert/cake"))],
    "cupcake": [components.Dessert, get_object_list(os.path.join(xml_root, "obj/meshes/dessert/cupcake"))],
    "donut": [components.Dessert, get_object_list(os.path.join(xml_root, "obj/meshes/dessert/donut"))],
    "waffle": [components.Dessert, get_object_list(os.path.join(xml_root, "obj/meshes/dessert/waffle"))],
    
    # tablewares
    "mug":[components.Mug, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/cups"))],
    "pan_seen":[components.FlatContainer, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/pans"), all=False)],
    "pan_unseen":[components.FlatContainer, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/pans"), all=False, seen=False)],
    "plate":[components.Plate, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/plates"))],
    "plate_seen":[components.Plate, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/plates"), all=False)],
    "plate_unseen":[components.Plate, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/plates"), all=False, seen=False)],
    "knife":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/knifes"))],
    "fork":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/forks"))],
    "spoon":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/spoons"))],
    "stove":[components.Stove, get_object_list(os.path.join(xml_root, "obj/meshes/stoves"))],
    "chopstick":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/chopsticks"))],
    "counter":[components.Counter, get_object_list(os.path.join(xml_root, "obj/meshes/counters"))],
    "knifeholder":[components.CommonContainer, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/holders/knifeholder"))],
    "knifeholder_seen":[components.CommonContainer, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/holders/knifeholder"), all=False)],
    "knifeholder_unseen":[components.CommonContainer, get_object_list(os.path.join(xml_root, "obj/meshes/tablewares/holders/knifeholder"), all=False, seen=False)],
    
    # condiments
    "bbq_sauce":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/bbq_sauce"))],
    "bbq_sauce_seen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/bbq_sauce"), all=False)],
    "bbq_sauce_unseen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/bbq_sauce"), all=False, seen=False)],
    "ketchup":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/ketchup"))],
    "ketchup_seen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/ketchup"), all=False)],
    "ketchup_unseen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/ketchup"), all=False, seen=False)],
    "shaker":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"))],
    "shaker_seen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"), all=False)],
    "shaker_unseen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"), all=False, seen=False)],
    "sugar":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"))],
    "sugar_seen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"), all=False)],
    "sugar_unseen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"), all=False, seen=False)],
    "salt":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"))],
    "salt_seen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"), all=False)],
    "salt_unseen":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/shaker"), all=False, seen=False)],
    
    "salad_dressing":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/salad_dressing"))],
    "hotsauce":[components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/condiment/hotsauce"))],
    # toys
    # DC series
    "aquaman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/DC/aquaman"))],
    "batman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/DC/batman"))],
    "flashman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/DC/flashman"))],
    "joker": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/DC/joker"))],
    "superman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/DC/superman"))],
    "wonder_woman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/DC/wonder_woman"))],
    # mickey series
    "mickey": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/mickey/mickey"))],
    "minnie": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/mickey/minnie"))],
    "donald": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/mickey/donald"))],
    "daisy": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/mickey/daisy"))],
    "pluto": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/mickey/pluto"))],
    # toystory
    "buzz_lightyear": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/toystory/buzz_lightyear"))],
    "woody": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/toystory/woody"))],
    "rex": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/toystory/rex"))],
    "jessie": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/toystory/jessie"))],
    "slinky_dog": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/toystory/slinky_dog"))],
    "alien": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/disney/toystory/alien"))],
    # marvel series
    "antman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/antman"))],
    "doctor_strange": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/doctor_strange"))],
    "hulk": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/hulk"))],
    "hawkeye": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/hawkeye"))],
    "ironman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/ironman"))],
    "loki": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/loki"))],
    "spiderman": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/spiderman"))],
    "thor": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/thor"))],
    "thanos": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/thanos"))],
    "vision": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/marvel/vision"))],
    # onepiece series
    "ace": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/ace"))],
    "luffy": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/luffy"))],
    "nami": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/nami"))],
    "robin": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/robin"))],
    "zoro": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/zoro"))],
    "chopper": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/chopper"))],
    "reiju": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/reiju"))],
    "sanji": [components.Toy, get_object_list(os.path.join(xml_root, "obj/meshes/toys/onepiece/sanji"))],
    
    # drinks
    "wine": [components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/wine"))],
    "alcohol":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/alcohol"))],
    "beer": [components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/beer"))],
    "cola":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/sota_drink/cola"))],
    "spirit":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/sota_drink/spirit"))],
    "monster":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/energy_drink/monster"))],
    "redbull":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/energy_drink/redbull"))],
    "jumpstart":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/energy_drink/jumpstart"))],
    "juice":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/juice"))],
    "milk":[components.Drink, get_object_list(os.path.join(xml_root, "obj/meshes/drinks/milk"))],
    
    # books
    'cotton_the_fabric_that_made_the_modern_world': [components.Book, 'obj/meshes/books/bookscan/cotton_the_fabric_that_made_the_modern_world/cotton_the_fabric_that_made_the_modern_world.xml'], 
    'one_hundred_years_of_solitude': [components.Book, 'obj/meshes/books/bookscan/one_hundred_years_of_solitude/one_hundred_years_of_solitude.xml'], 
    'electron_transport_in_quantum_dots': [components.Book, 'obj/meshes/books/bookscan/electron_transport_in_quantum_dots/electron_transport_in_quantum_dots.xml'], 
    'landmark_cases_in_property_law': [components.Book, 'obj/meshes/books/bookscan/landmark_cases_in_property_law/landmark_cases_in_property_law.xml'], 
    'cybersecurity': [components.Book, 'obj/meshes/books/bookscan/cybersecurity/cybersecurity.xml'], 
    'the_story_of_american_freedom': [components.Book, 'obj/meshes/books/bookscan/the_story_of_american_freedom/the_story_of_american_freedom.xml'], 
    'data_visualization': [components.Book, 'obj/meshes/books/bookscan/data_visualization/data_visualization.xml'], 
    'biopolymer_composites_in_electronics': [components.Book, 'obj/meshes/books/bookscan/biopolymer_composites_in_electronics/biopolymer_composites_in_electronics.xml'], 
    'educated': [components.Book, 'obj/meshes/books/bookscan/educated/educated.xml'], 
    'a_tale_of_two_cities': [components.Book, 'obj/meshes/books/bookscan/a_tale_of_two_cities/a_tale_of_two_cities.xml'], 
    'critical_infrastructure_protection_in_homeland_security': [components.Book, 'obj/meshes/books/bookscan/critical_infrastructure_protection_in_homeland_security/critical_infrastructure_protection_in_homeland_security.xml'], 
    'contract_law_in_japan': [components.Book, 'obj/meshes/books/bookscan/contract_law_in_japan/contract_law_in_japan.xml'], 
    'mobile_computing_principles': [components.Book, 'obj/meshes/books/bookscan/mobile_computing_principles/mobile_computing_principles.xml'], 
    'flash_mx_bible': [components.Book, 'obj/meshes/books/bookscan/flash_mx_bible/flash_mx_bible.xml'], 
    'foundamentals_of_molecular_biology': [components.Book, 'obj/meshes/books/bookscan/foundamentals_of_molecular_biology/foundamentals_of_molecular_biology.xml'], 
    'engineering_biopolymers': [components.Book, 'obj/meshes/books/bookscan/engineering_biopolymers/engineering_biopolymers.xml'], 
    'the_adventures_of_huckleberry_finn': [components.Book, 'obj/meshes/books/bookscan/the_adventures_of_huckleberry_finn/the_adventures_of_huckleberry_finn.xml'], 
    'cloud_computing': [components.Book, 'obj/meshes/books/bookscan/cloud_computing/cloud_computing.xml'], 
    'the_autobiography_of_benjamin_franklin': [components.Book, 'obj/meshes/books/bookscan/the_autobiography_of_benjamin_franklin/the_autobiography_of_benjamin_franklin.xml'], 
    'the_great_divergence_china_europe_and_the_making_of_the_modern_world_economy': [components.Book, 'obj/meshes/books/bookscan/the_great_divergence_china_europe_and_the_making_of_the_modern_world_economy/the_great_divergence_china_europe_and_the_making_of_the_modern_world_economy.xml'], 
    'scholars_of_contract_law': [components.Book, 'obj/meshes/books/bookscan/scholars_of_contract_law/scholars_of_contract_law.xml'], 
    'python': [components.Book, 'obj/meshes/books/bookscan/python/python.xml'], 
    'opengl': [components.Book, 'obj/meshes/books/bookscan/opengl/opengl.xml'], 
    'the_life_of_samuel_johnson': [components.Book, 'obj/meshes/books/bookscan/the_life_of_samuel_johnson/the_life_of_samuel_johnson.xml'], 
    'wi-foo': [components.Book, 'obj/meshes/books/bookscan/wi-foo/wi-foo.xml'], 
    'government_and_information_rights': [components.Book, 'obj/meshes/books/bookscan/government_and_information_rights/government_and_information_rights.xml'], 
    'introduction_to_electronics': [components.Book, 'obj/meshes/books/bookscan/introduction_to_electronics/introduction_to_electronics.xml'], 
    'steve_jobs': [components.Book, 'obj/meshes/books/bookscan/steve_jobs/steve_jobs.xml'], 

    # books by order
    'century21_the_three_body_problem_book': [components.Book, 'obj/meshes/books/books_time_order/Century21_The_Three_Body_Problem_Book/century21_the_three_body_problem_book.xml'], 
    'century21_harry_potter_and_the_chamber_of_secrets_book': [components.Book, 'obj/meshes/books/books_time_order/Century21_Harry_Potter_and_the_Chamber_of_Secrets_Book/century21_harry_potter_and_the_chamber_of_secrets_book.xml'],
    'century20_the_metamorphosis_book': [components.Book, 'obj/meshes/books/books_time_order/Century20_The_Metamorphosis_Book/century20_the_metamorphosis_book.xml'], 
    'century20_the_old_man_and_the_sea_book': [components.Book, 'obj/meshes/books/books_time_order/Century20_The_Old_Man_and_the_Sea_Book/century20_the_old_man_and_the_sea_book.xml'], 
    'century19_boule_de_suif_book': [components.Book, 'obj/meshes/books/books_time_order/Century19_Boule_de_Suif_Book/century19_boule_de_suif_book.xml'], 
    'century19_les_miserables_book': [components.Book, 'obj/meshes/books/books_time_order/Century19_Les_Misérables_Book/century19_les_misérables_book.xml'], 
    'century18_pride_and_prejudice_book': [components.Book, 'obj/meshes/books/books_time_order/Century18_Pride_and_Prejudice_Book/century18_pride_and_prejudice_book.xml'], 
    'century18_du_contrat_social_book': [components.Book, 'obj/meshes/books/books_time_order/Century18_Du_Contrat_Social_Book/century18_du_contrat_social_book.xml'], 
    'century17_don_quixote_book': [components.Book, 'obj/meshes/books/books_time_order/Century17_Don_Quixote_Book/century17_don_quixote_book.xml'], 
    'century17_romeo_and_juliet_book': [components.Book, 'obj/meshes/books/books_time_order/Century17_Romeo_and_Juliet_Book/century17_romeo_and_juliet_book.xml'], 

    # laptop
    "laptop": [components.Laptop, get_object_list(os.path.join(xml_root, "obj/meshes/tools/laptop"))],
    # coffee machine
    "coffee_machine": [components.CoffeeMachine, get_object_list(os.path.join(xml_root, "obj/meshes/coffee_machines"))],
    # juicer
    "juicer": [components.Juicer, get_object_list(os.path.join(xml_root, "obj/meshes/tools/juicer/juicer_0/juicer_body"))],
    "juicer_cap": [components.CommonGraspedEntity, get_object_list(os.path.join(xml_root, "obj/meshes/tools/juicer/juicer_0/juicer_cap"))],
    
    # tools
    "hammer": [components.Hammer, get_object_list(os.path.join(xml_root, "obj/meshes/tools/hammer/whole_hammer"))],
    "hammer_handle": [components.HammerHandle, get_object_list(os.path.join(xml_root, "obj/meshes/tools/hammer/hammer_handle"))],
    "hammer_head": [components.HammerHead, get_object_list(os.path.join(xml_root, "obj/meshes/tools/hammer/hammer_head"))],
    "nail": [components.Nail, get_object_list(os.path.join(xml_root, "obj/meshes/tools/nail"))],
    "seesaw": [components.CommonContainer, "obj/meshes/tools/seesaw_machine/seesaw_machine.xml"],
    "button":[components.Button, None],
    "cord": [components.Cord, get_object_list(os.path.join(xml_root, "obj/meshes/tools/electronic_cord/cord_head"))],
    "electronic_outlet": [components.Entity, get_object_list(os.path.join(xml_root, "obj/meshes/tools/outlets"))],
    "mirrors": [components.Mirrors, get_object_list(os.path.join(xml_root, "obj/meshes/tools/mirror"))],
}   

additional_dict = {}
for key, value in name2class_xml.items():
    if value[-1] is None:continue
    additional_dict[key+"_seen"] = [value[0], value[1][:len(value[1])//2]]
    additional_dict[key+"_unseen"] = [value[0], value[1][len(value[1])//2:]]
name2class_xml.update(additional_dict)