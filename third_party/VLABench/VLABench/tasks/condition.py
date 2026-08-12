import numpy as np
from VLABench.utils.register import register
from VLABench.utils.utils import distance
from VLABench.tasks.components.entity import Entity

class Condition:
    def __init__(self):
        pass
    
    def is_met(self, physics=None):
        raise NotImplementedError()
    
    def met_progress(self, physics=None):
        return self.is_met(physics)

@register.add_condition("order")
class OrderCondition(Condition):
    """
    check the position order of the given entities.
    params:
        entities: the given entities should in the expected order.
        axis: the  axis to check the order.
        offset: the acceptable offset between the entities in other axis.
    """
    def __init__(self, entities, axis=[0], offset=0.1):
        self.entities = entities
        self.axis = axis
        self.offset = offset
        
    def is_met(self, physics):
        if isinstance(self.entities[-1], Entity):
            self.entities_mjcf = [entity.mjcf_model.worldbody for entity in self.entities]
        entity_points = [physics.bind(entity_mjcf).xpos for entity_mjcf in self.entities_mjcf]
        for axis in [0, 1, 2]: # x y z
            if axis in self.axis:
                if not all([entity_points[i][axis] < entity_points[i+1][axis] for i in range(len(entity_points)-1)]):
                    return False
            else:
                if not all([(entity_points[i][axis] - entity_points[i+1][axis]) < self.offset for i in range(len(entity_points)-1)]):
                    return False
        return True

@register.add_condition("contain")
class ContainCondition(Condition):
    """
    Check if the container contains the target entities
    params:
        container: the container to contain the target eneities
        entities: the target entities to be contained
    """
    def __init__(self, container, entities, **kwargs):
        assert container is not None, "container must be provided"
        self.container = container
        self.entities = entities
        self.kwargs = kwargs
    
    def is_met(self, physics=None):
        if isinstance(self.entities[-1], Entity):
            self.entities_mjcf = [entity.mjcf_model.worldbody for entity in self.entities]
        entity_points = [physics.bind(entity_mjcf).xpos for entity_mjcf in self.entities_mjcf]
        for point in entity_points:
            if not self.container.contain(point, physics, **self.kwargs):
                return False
        return True

    def met_progress(self, physics=None):
        # TODO: return the progress of the condition
        return super().met_progress(physics)
    
@register.add_condition("not_contain")
class NotContainCondition(Condition):
    """
    Check if the container does not contain the target entities.
    params:
        container: the container to not contain the target entities
        entities: the target entities to be not contained
    """
    def __init__(self, container, entities):
        assert container is not None, "container must be provided"
        self.container = container
        self.entities = entities

    def is_met(self, physics=None):
        if isinstance(self.entities[-1], Entity):
            self.entities_mjcf = [entity.mjcf_model.worldbody for entity in self.entities]
        entity_points = [physics.bind(entity_mjcf).xpos for entity_mjcf in self.entities_mjcf]
        for point in entity_points:
            if self.container.contain(point, physics):
                return False
        return True
    
    def met_progress(self, physics=None):
        # TODO: return the progress of the condition
        return super().met_progress(physics)

@register.add_condition("is_grasped")
class IsGraspedCondition(Condition):
    """
    Check if the target entities are grasped
    """
    def __init__(self, entities, robot):
        self.entities = entities
        self.robot = robot
        
    def is_met(self, physics=None):
        for entity in self.entities:
            if not entity.is_grasped(physics, self.robot):
                return False
        return True

@register.add_condition("press_button")
class ButtonPressedCondition(Condition):
    """
    Check if the button is pressed
    """
    def __init__(self, target_button):
        self.button = target_button
        
    def is_met(self, physics=None):
        return self.button.is_pressed()
    
@register.add_condition("on")
class OnCondition(Condition):
    """
    Check if the target entities are on the target surface.
    In most of cases, only on entity in entities.
    """
    def __init__(self, entities, container):
        self.entities = entities
        self.container = container
        
    def is_met(self, physics=None):
        contacts = physics.data.contact
        
        container_geoms_id = [physics.bind(geom).element_id for geom in self.container.geoms]
        container_geoms_xpos = [physics.bind(geom).xpos for geom in self.container.geoms]
        max_xpos_z = max([xpos[-1] for xpos in container_geoms_xpos])
        for entity in self.entities:
            is_contacted = False
            entity_geom_ids = [physics.bind(geom).element_id for geom in entity.geoms]
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            # z position detection
            if entity_xpos[-1] <= max_xpos_z:
                return False
            # on contact detection
            for contact in contacts:
                if (contact.geom1 in container_geoms_id and contact.geom2 in entity_geom_ids) or \
                    (contact.geom2 in entity_geom_ids and contact.geom1 in container_geoms_id):
                    is_contacted = True
                    break
            if is_contacted is False:
                return False
        return True

@register.add_condition("above")
class AboveCondition(Condition):
    """
    Entity above the platform/container condition. Usually used for the entity is above the flat container. 
    params:
        target_entity: the target entity to be above the platform
        platform: the platform/flat container
    """
    def __init__(self, target_entity, platform):
        self.target_entity = target_entity
        self.platform = platform
    
    def is_met(self, physics):
        target_entity_xpos = physics.bind(self.target_entity.mjcf_model.worldbody).xpos
        z_platform = physics.bind(self.platform.mjcf_model.worldbody).xpos[-1]
        if target_entity_xpos[-1] < z_platform:
            return False
        else:
            point_to_check = target_entity_xpos
            point_to_check[-1] = z_platform + 0.01
            if self.platform.contain(point_to_check, physics): return True    
        return False

@register.add_condition("pour")
class PourCondition(Condition):
    """
    The cup/shaker/other_entity is poured. 
    As mujoco does not support the liquid simulation, use this condition to simplify. 
    The condition is the top site z pos is lower than the bottom site z pos.
    params:
        target_entity: the target entity to be poured
        threshold: the threshold of z_top - z_bottom, to confirm whether the entity is poured. 
    """
    def __init__(self, target_entity, threshold=0):
        self.target_entity = target_entity
        self.threshold = threshold
        
    def is_met(self, physics):
        top_site = self.target_entity.mjcf_model.worldbody.find("site", "top_site")
        bottom_site = self.target_entity.mjcf_model.worldbody.find("site", "bottom_site")
        top_site_xpos, bottom_site_xpos = physics.bind(top_site).xpos, physics.bind(bottom_site).xpos
        if (bottom_site_xpos[-1] - top_site_xpos[-1]) > self.threshold:
            return True
        else:
            return False

@register.add_condition("on_position")
class OnPositionCondition(Condition):
    """
    Entities should close to the target position within a certain distance
    params:
        entities: list, the target entities to be close to the target positions
        positions: list, the target positions
        tolerance_distance: the acceptable distance between the entities and the target positions
        dimension: the dimension of the target positions, 2 or 3
    """
    def __init__(self, entities, positions, tolerance_distance=0.03, dimension=2):
        self.entities = entities
        self.target_positions = positions
        self.tolerance_distance = tolerance_distance
        self.dimension = dimension
    
    def is_met(self, physics=None):
        for entity, target_pos in zip(self.entities, self.target_positions):
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            if distance(entity_xpos[:self.dimension], target_pos[:self.dimension]) > self.tolerance_distance:
                return False
        return True

@register.add_condition("contact")
class ContactCondition(Condition):
    """
    Check if the target entities are in contact with each other
    params:
        entity1: the first entity, Entity class of list of Entity
        entity2: the second entity, Entity class of list of Entity
    """
    def __init__(self, entity1, entity2):
        self.entity1 = entity1
        self.entity2 = entity2
        self.entity_geoms_id = None
        
    def is_met(self, physics=None):
        contacts = physics.data.contact
        if self.entity_geoms_id is None:
            self.entity_geoms_id = dict(
                entity1=[],
                entity2=[]
            )
            if isinstance(self.entity1, Entity):                
                self.entity_geoms_id["entity1"].extend([physics.bind(geom).element_id for geom in self.entity1.geoms])
            elif isinstance(self.entity1, list):    
                self.entity_geoms_id["entity1"].extend([physics.bind(geom).element_id for geom in self.entity1])
                
            if isinstance(self.entity2, Entity):                
                self.entity_geoms_id["entity2"].extend([physics.bind(geom).element_id for geom in self.entity2.geoms])
            elif isinstance(self.entity2, list):    
                self.entity_geoms_id["entity2"].extend([physics.bind(geom).element_id for geom in self.entity2])
            
        for contact in contacts:
            if (contact.geom1 in self.entity_geoms_id["entity1"] and contact.geom2 in self.entity_geoms_id["entity2"]) or \
                (contact.geom2 in self.entity_geoms_id["entity2"] and contact.geom1 in self.entity_geoms_id["entity1"]):
                return True
        return False
    
@register.add_condition("joint_in_range")
class JointInRangeCondition(Condition):
    """
    The target joint should be in specific range.
    Params:
        entities: the target entities, list of Entity with a single joint (hinge or slide)
        target_pos_range: the target position range
    """
    def __init__(self, entities, target_pos_range):
        self.entities = entities
        self.target_pos_range = target_pos_range
    
    def is_met(self, physics=None):
        for entity in self.entities:
            joints = entity.joints
            assert len(joints) == 1, "The number of joints should be equal to the target position range"
            if physics.bind(joints[-1]).qpos < self.target_pos_range[0] or physics.bind(joints[-1]).qpos > self.target_pos_range[1]:
                return False
        return True

@register.add_condition("lift")
class LiftCondition(Condition):
    """
    The entity should be lifted above the target height.
    params:
        entities: the target entities to be lifted
        target_height: the target height to achieve
    """
    def __init__(self, entities, target_height):
        self.entities = entities
        self.target_height = target_height
    
    def is_met(self, physics=None):
        for entity in self.entities:
            entity_xpos = physics.bind(entity.mjcf_model.worldbody).xpos
            if entity_xpos[-1] < self.target_height:
                return False
        return True

class ConditionSet:
    """
    A set of conditions, the condition set is met only when all the conditions are satisfied simutanously.
    """
    def __init__(self, conditions):
        self.conditions = conditions
    
    def __len__(self):
        return len(self.conditions)
    
    def is_met(self, physics=None):
        conditions_are_met = [condition.is_met(physics) for condition in self.conditions]
        return all(conditions_are_met)
    
    def add(self, condition):
        self.conditions.append(condition)
    
    def met_progress(self, physics=None):
        """
        compute the progress of the condition set. 
        Return the ratio of the conditions that are met and those conditions are met.
        """
        conditions_are_met = [condition.is_met(physics) for condition in self.conditions]
        met_conditions = []
        for condition, met in zip(self.conditions, conditions_are_met):
            if met: met_conditions.append(condition)
        return sum(conditions_are_met) / len(conditions_are_met), met_conditions 

@register.add_condition("asyn_sequence")
class AsynSequenceCondition(Condition):
    """
    A time sequence of conditions, the condition is met only when all the sub-conditions are satisfied asynchronously.
    Different with single condition set, the sub-conditions' met are maintained in a member value.
    params:
        condition_sets: a list of ConditionSet, each ConditionSet contains a list of conditions.   
    
    """
    def __init__(self, condition_sets):
        assert isinstance(condition_sets, list) and isinstance(condition_sets[0], ConditionSet), "condition_set should be a list of condition_set"
        self.condition_has_been_met = [False for _ in range(len(condition_sets))]
        self.condition_sets = condition_sets
        
    def is_met(self, physics=None):
        for i, condition in enumerate(self.condition_sets):
            if not self.condition_has_been_met[i]:
                if condition.is_met(physics):
                    self.condition_has_been_met[i] = True
        if all(self.condition_has_been_met): return True
        else: return False
    
    def met_progress(self, physics=None):
        met_scores = []
        # subcontion_num = 0
        for condition_set in self.condition_sets:
            progress_score, _ = condition_set.met_progress(physics)
            met_scores.append(progress_score)
        return np.mean(met_scores), None

@register.add_condition("or")
class OrCondition(Condition):
    """
    Any one of the conditions in the condition set is met.
    params:
        condition_sets: a list of ConditionSet, each ConditionSet contains a list of conditions.
    """
    def __init__(self, condition_sets):
        assert isinstance(condition_sets, list) and isinstance(condition_sets[0], ConditionSet), "condition_sets should be a list of condition sets"
        self.condition_sets = condition_sets
    
    def is_met(self, physics=None):
        return any([condition_set.is_met(physics) for condition_set in self.condition_sets])