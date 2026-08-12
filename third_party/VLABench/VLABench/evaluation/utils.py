import networkx as nx
import json
import matplotlib.pyplot as plt
from collections import Counter

# predefined subtask patterns
SUBTASK_PATTERN = [
    ["pick", "place"],
    ["pick", "insert"],
    ["pick", "pour", "place"],
    ["pick", "pour"],
    ["pick", "pull"],
    ["pick", "lift"],
    ["pick", "push", "pull"],
    ["pick", "push", "place"],
    ["pick", "push"],
    ["pick", "open_door"],
    ["press"]
]

def find_subtasks(skill_sequence, patterns):
    """
    find all matching subtasks in skill_sequence based on patterns, return a list of subtasks in order
    """
    subtasks = []
    sequence = [skill["name"] for skill in skill_sequence]
    i = 0

    while i < len(sequence):
        matched = False
        for pattern in patterns:
            if sequence[i:i + len(pattern)] == pattern:
                # print(i, "to", i + len(pattern), "match", pattern)
                subtasks.append(skill_sequence[i:i + len(pattern)])
                i += len(pattern)
                matched = True
                break
        if not matched:
            i += 1
    return subtasks
   
def build_graph(skill_sequence, patterns, dependency="Sequential"):
    """
    Build graph structure based on dependency relationship, where each subtask is a node of the graph
    """
    G = nx.DiGraph()

    G.add_node("START", subtask="START", target_entity=None, target_container=None)
    
    subtasks = find_subtasks(skill_sequence, patterns)
    node_count = 1
    node_ids = []
    
    for subtask in subtasks:
        subtask_name = "-".join([skill["name"] for skill in subtask])
        
        target_entities = [skill["params"].get("target_entity_name") for skill in subtask]
        target_containers = [skill["params"].get("target_container_name") for skill in subtask]

        target_entities = [entity for entity in target_entities if entity is not None]
        target_containers = [container for container in target_containers if container is not None]
        target_entity = target_entities[-1] if target_entities else None
        target_container = target_containers[-1] if target_containers else None
        
        # add subtask node to the graph
        node_id = f"{subtask_name}_{node_count}"
        G.add_node(node_id, 
                   subtask=subtask_name,
                   target_entity=target_entity,
                   target_container=target_container)
        node_ids.append(node_id)
        node_count += 1
    
    if dependency == "Sequential":
        # linear dependency between subtasks
        previous_node = "START"
        for node_id in node_ids:
            G.add_edge(previous_node, node_id)
            previous_node = node_id
    elif dependency == "Seq-independent":
        # independent sequential dependency between subtasks
        for node_id in node_ids:
            G.add_edge("START", node_id)
    elif isinstance(dependency, dict):
        # custom dependency relationship
        for src, dests in dependency.items():
            for dest in dests:
                if src <= len(node_ids) and dest <= len(node_ids):
                    G.add_edge(node_ids[src - 1], node_ids[dest - 1])
                else:
                    raise ValueError("The dependency index is out of range.")
        
        for node_id in node_ids:
            if G.in_degree(node_id) == 0:
                G.add_edge("START", node_id)

    
    return G

def hierarchical_layout(G, root="START", layer_width=2.0, layer_height=2.0):
    """
        create a hierarchical layout to make the graph horizontally arranged by layers,
    """
    pos = {}
    layer_nodes = {0: [root]}
    layers = {root: 0}
    visited = set([root])
    
    for node in nx.topological_sort(G):
        if node == root:
            continue
        
        max_layer = max(layers[predecessor] for predecessor in G.predecessors(node) if predecessor in layers)
        current_layer = max_layer + 1
        layers[node] = current_layer

        if current_layer not in layer_nodes:
            layer_nodes[current_layer] = []
        layer_nodes[current_layer].append(node)
        visited.add(node)
    
    for layer, nodes in layer_nodes.items():
        x = layer * layer_width
        y_start = -(len(nodes) - 1) * layer_height / 2
        for i, node in enumerate(nodes):
            pos[node] = (x, y_start + i * layer_height)
    
    return pos

def visualize_graph(G, title="Dependency Graph"):
    """
        Visualize the dependency graph with hierarchical layout and arrows
    """
    pos = hierarchical_layout(G)
    plt.figure(figsize=(10, 8))
    plt.title(title)

    nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue', edgecolors='black')

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=20, edge_color='gray', arrows=True)
    
    labels = {node: node for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='black')

    plt.axis('off')
    plt.show()

def exact_match_percentage(graph1, graph2):
    """
    Compute the exact match percentage of graph2 relative to graph1, where the match requires the pattern, target_entity, target_container

    Params:
        graph1: expert subtask graph
        graph2: the graph to match
    """
    exact_match_count = 0
    total_nodes = len([node for node in graph1.nodes if node != "START"])

    # Obtain the topological order of graph1 and graph2.
    topo_order1 = list(nx.topological_sort(graph1))
    topo_order2 = list(nx.topological_sort(graph2))

    # Create the hierarchical structure of the nodes in graph1 and graph2
    layers1 = {}
    layers2 = {}
    
    for node in topo_order1:
        if node == "START":
            layers1[node] = 0
        else:
            layers1[node] = max(layers1[predecessor] for predecessor in graph1.predecessors(node)) + 1

    for node in topo_order2:
        if node == "START":
            layers2[node] = 0
        else:
            layers2[node] = max(layers2[predecessor] for predecessor in graph2.predecessors(node)) + 1

    # compare the nodes in each layer
    max_layer = max(layers1.values())
    for layer in range(1, max_layer + 1):
        nodes_layer1 = [node for node, lvl in layers1.items() if lvl == layer]
        nodes_layer2 = [node for node, lvl in layers2.items() if lvl == layer]
        
        for node1 in nodes_layer1:
            pattern1 = graph1.nodes[node1].get("subtask")
            target_entity1 = graph1.nodes[node1].get("target_entity")
            target_container1 = graph1.nodes[node1].get("target_container")

            match_found = False
            for node2 in nodes_layer2:
                pattern2 = graph2.nodes[node2].get("subtask")
                target_entity2 = graph2.nodes[node2].get("target_entity")
                target_container2 = graph2.nodes[node2].get("target_container")

                if pattern1 == pattern2 and target_entity1 == target_entity2 and target_container1 == target_container2:

                    predecessors1 = set(graph1.predecessors(node1))
                    predecessors2 = set(graph2.predecessors(node2))

                    if len(predecessors1) == len(predecessors2):
                        matched_predecessors = all(
                            any(
                                graph2.nodes[p2].get("subtask") == graph1.nodes[p1].get("subtask") and
                                graph2.nodes[p2].get("target_entity") == graph1.nodes[p1].get("target_entity") and
                                graph2.nodes[p2].get("target_container") == graph1.nodes[p1].get("target_container")
                                for p2 in predecessors2
                            )
                            for p1 in predecessors1
                        )

                        if matched_predecessors:
                            exact_match_count += 1
                            nodes_layer2.remove(node2)
                            match_found = True
                            break
                        
            if not match_found:
                continue

    exact_match_score = (exact_match_count / total_nodes) * 100

    return exact_match_score

def get_exact_match(skill_sequence1, skill_sequence2, dependency):
    graph1 = build_graph(skill_sequence1, SUBTASK_PATTERN, dependency=dependency)
    graph2 = build_graph(skill_sequence2, SUBTASK_PATTERN, dependency=dependency)
    exact_match_score = exact_match_percentage(graph1, graph2)
    return exact_match_score

def calculate_skill_and_entity_scores(sequence1, sequence2):
    """
    Calculate the skill match score and entity/container recognition match score of sequence2
    relative to sequence1, without considering the order and the correspondence between skills and entities.
    
    Parameters:
        sequence1: list - The standard skill sequence (includes skills and target objects).
        sequence2: list - The skill sequence to be compared.

    Returns:
        dict - A dictionary containing the skill match score and the percentage of correct entity/container recognition.
    """

    skills1 = [skill["name"] for skill in sequence1]
    skills2 = [skill["name"] for skill in sequence2]

    entities1 = []
    entities2 = []

    for skill in sequence1:
        if skill["params"].get("target_entity_name") is not None:
            entities1.append(("target_entity", skill["params"].get("target_entity_name")))
        if skill["params"].get("target_container_name") is not None:
            entities1.append(("target_container", skill["params"].get("target_container_name")))
    
    for skill in sequence2:
        if skill["params"].get("target_entity_name") is not None:
            entities2.append(("target_entity", skill["params"].get("target_entity_name")))
        if skill["params"].get("target_container_name") is not None:
            entities2.append(("target_container", skill["params"].get("target_container_name")))

    skills1_counter = Counter(skills1)
    skills2_counter = Counter(skills2)
    skill_match_count = sum((skills1_counter & skills2_counter).values())  
    
    entities1_counter = Counter(entities1)
    entities2_counter = Counter(entities2)
    entity_match_count = sum((entities1_counter & entities2_counter).values())

    total_skills = len(skills1)
    total_entities = len(entities1)
    skill_match_score = (skill_match_count / total_skills) * 100 if total_skills > 0 else 0
    entity_match_score = (entity_match_count / total_entities) * 100 if total_entities > 0 else 0

    return {
        "skill_match_score": skill_match_score,
        "entity_match_score": entity_match_score
    }

def calculate_skill_with_entity_scores(sequence1, sequence2):
    """
     Calculate the simultaneous skill and entity/container recognition match score of sequence2
     relative to sequence1, without considering the order.
    
     Parameters:
       sequence1: list - The standard skill sequence (includes skills and target objects).
       sequence2: list - The skill sequence to be compared.
    
     Returns:
       dict - A dictionary containing the percentage of correct skill and entity/container recognition.
    """
    skill_with_entity1 = [(skill["name"], skill["params"].get("target_entity_name"), skill["params"].get("target_container_name")) for skill in sequence1]
    skill_with_entity2 = [(skill["name"], skill["params"].get("target_entity_name"), skill["params"].get("target_container_name")) for skill in sequence2]

    skill_with_entity1_counter = Counter(skill_with_entity1)
    skill_with_entity2_counter = Counter(skill_with_entity2)
    skill_with_entity_match_count = sum((skill_with_entity1_counter & skill_with_entity2_counter).values())  # 取交集中的最小匹配数

    total_skills = len(skill_with_entity1)
    skill_with_entity_match_score = (skill_with_entity_match_count / total_skills) * 100 if total_skills > 0 else 0

    return {
        "skill_with_entity_match_score": skill_with_entity_match_score
    }

def get_final_score(standard_skill_sequence, model_skill_sequence, dependency):
    """
     Calculate the matching score of the model's skill sequence relative to the standard skill sequence.
    
     Parameters:
       standard_skill_sequence: list - The standard skill sequence.
       model_skill_sequence: list - The skill sequence output by the model.
       dependency: str - The type of dependency relation.
    
     Returns:
       dict - A dictionary containing the skill match score and the percentage of correct entity/container recognition.
    """

    skill_entity_scores = calculate_skill_and_entity_scores(standard_skill_sequence, model_skill_sequence)

    skill_with_entity_scores = calculate_skill_with_entity_scores(standard_skill_sequence, model_skill_sequence)

    exact_match_score = get_exact_match(standard_skill_sequence, model_skill_sequence, dependency)

    score_weight = {
        "skill_match_score": 0.4,
        "entity_match_score": 0.4,
        "skill_with_entity_match_score": 0.1,
        "exact_match_score": 0.1
    }
    total_score = sum(score_weight[key] * value for key, value in skill_entity_scores.items())
    
    return {
        "skill_match_score": skill_entity_scores["skill_match_score"],
        "entity_match_score": skill_entity_scores["entity_match_score"],
        "skill_with_entity_match_score": skill_with_entity_scores["skill_with_entity_match_score"],
        "exact_match_score": exact_match_score,
        "total_score": total_score
    }