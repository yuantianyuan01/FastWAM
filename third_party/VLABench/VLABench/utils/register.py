import collections

class Registration():
    def __init__(self):
        self._tasks = collections.OrderedDict()
        self._entities = collections.OrderedDict()
        self._robots = collections.OrderedDict()
        self._conditions = collections.OrderedDict()
        self._config_managers = collections.OrderedDict()
    
    def add_task(self, task_name):
        def wrap(cls):
            self._tasks[task_name] = cls
            return cls
        return wrap
    
    def add_entity(self, entity_name):
        def wrap(cls):
            self._entities[entity_name] = cls
            return cls
        return wrap
    
    def add_robot(self, robot_name):
        def wrap(cls):
            self._robots[robot_name] = cls
            return cls
        return wrap

    def add_condition(self, condition_name):
        def wrap(cls):
            self._conditions[condition_name] = cls
            return cls
        return wrap
    
    def add_config_manager(self, config_manager_name):
        def wrap(cls):
            self._config_managers[config_manager_name] = cls
            return cls
        return wrap
    
    def __getitem__(self, key):
        return self._tasks[key] or self._entities[key]
    
    def load_entity(self, key):
        return self._entities[key]
    
    def load_task(self, key):
        return self._tasks[key]
    
    def load_robot(self, key):
        return self._robots[key]
    
    def load_condition(self, key):
        return self._conditions[key]
    
    def load_config_manager(self, key):
        return self._config_managers[key]
    
    def keys(self):
        return self._tasks.keys()
    
    def __len__(self):
        return len(self._tasks)
    
    def __iter__(self):
        return iter(self._tasks)
    
    def get_robot_names(self):
        return self._robots.keys()
      
register = Registration()

