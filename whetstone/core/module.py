# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
from typing import Any, Dict, TypeVar, Generic, get_args
import yaml
from pathlib import Path

from omegaconf import DictConfig, ListConfig

from whetstone.utils import JSONType

class BaseModule(ABC):
    pass    

class BaseState(ABC):
    def __init__(self, instance: Any, persisted_data: JSONType | None = None):
        """Initialize state from a dictionary."""
        pass

    def persist(self) -> JSONType:
        """Convert state to a dictionary for serialization."""
        return {}
    

class StatefulModule[StateT: BaseState](BaseModule):
    _state: StateT | None = None

    def __init_subclass__(cls, **kwargs):
        gen_typevars = []
        for base in cls.__orig_bases__:
            gen_typevars.extend([t for t in get_args(base) if not isinstance(t, TypeVar) and issubclass(t, BaseState)])

        if len(gen_typevars) > 1:
            raise ValueError(f"StatefulModule {cls.__name__} must have exactly one BaseState type argument")
        elif len(gen_typevars) == 1:
            cls._state_type = gen_typevars[0]
        else:
            cls._state_type = BaseState

    def __init__(self, _state: StateT | None = None):
        super().__init__()
        self._state = _state
    
    @property
    def state(self) -> StateT:
        if self._state is None:
            self._state = self._init_state()
            
        if self._state is None:
            raise ValueError(f"State for class {self.__class__.__name__} could not be initialized")
            
        return self._state
    
    @state.setter
    def state(self, value: StateT):
        self._state = value
        
    def _init_state(self) -> StateT:
        return self._state_type(self)


class ModuleRegistry:
    _instance = None
    _modules = {}
    _instances = {}
    _state_cache = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModuleRegistry, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, module_class=None, *, name=None, make_dataclass=True):
        """Register a module class with the registry and convert it to a dataclass.
        
        Can be used as a decorator:
            @ModuleRegistry.register
            class MyModule(BaseModule):
                pass
                
        Or with a custom name:
            @ModuleRegistry.register(name="custom_name")
            class MyModule(BaseModule):
                pass
        """
        def decorator(module_class):
            if not issubclass(module_class, BaseModule):
                raise TypeError(f"{module_class.__name__} is not a subclass of BaseModule")
            
            module_name = name or module_class.__name__
            cls._modules[module_name] = module_class

            # Apply dataclass decorator
            if make_dataclass:
                module_class = dataclass(module_class)

            return module_class
            
        if module_class is not None:
            return decorator(module_class)
        return decorator
    
    @classmethod
    def instantiate(cls, config: DictConfig, module_class: type[BaseModule] | None = None):
        """Instantiate a module from a configuration.
        
        Handles both simple configs and configs with dependencies.
        Supports both direct class names and _target_ style configurations.
        """
        # Check if we've already instantiated this exact config
        import hashlib
        inst_id = hashlib.sha256(str(config).encode()).hexdigest()
        if inst_id in cls._instances:
            return cls._instances[inst_id]
            
        # Handle simple configs with name field
        elif isinstance(config, DictConfig) and (
            any(key in config for key in ["name", "_target_"])
            or (module_class is not None)
        ):
            if module_class is None:
                module_name = config["_target_"].split(".")[-1] if "_target_" in config else config["name"]
                if module_name not in cls._modules:
                    raise ValueError(f"Module {module_name} not registered")
                    
                module_class = cls._modules[module_name]
            
            # Process the rest of the config to instantiate dependencies
            kwargs = {}
            for key, value in config.items():
                if key not in ["name", "_target_"]:
                    kwargs[key] = cls.instantiate(value)
                    
            args = config.get("args", [])

            # If the module class is a StatefulModule, we need to rehydrate the state
            if isinstance(module_class, StatefulModule) and inst_id in cls._state_cache:
                kwargs["state"] = cls._state_cache[inst_id]

            instance = module_class(*args, **kwargs)
            
        # Handle lists or nested configs
        elif isinstance(config, ListConfig):
            return [cls.instantiate(item) for item in config]
        
        # Recurse into simple dicts
        elif isinstance(config, DictConfig):
            return {key: cls.instantiate(value) for key, value in config.items()}
        
        else:
            # This is a simple value, return it
            return config
            
        # Cache the instance
        cls._instances[inst_id] = instance
        return instance
    
    @classmethod
    def suspend_state(cls) -> dict:
        """Suspend the state of a stateful module instance."""
        all_states = {}

        for inst_id, instance in cls._instances.items():
            if isinstance(instance, StatefulModule):
                # We need to include a fully qualified name for the state class
                state_class_name = f"{instance.__class__.__module__}.{instance.state.__class__.__name__}"
                all_states[inst_id] = {
                    "state_class": state_class_name,
                    "state": instance.state.persist()
                }

        return all_states
    
    @classmethod
    def rehydrate_state(cls, state_data: dict):
        """Rehydrate the given module states. Applies state to existing instances and will apply it to future instances."""
        for instance_id, full_info in state_data.items():
            state_class_name = full_info["state_class"]

            # Split into module and class name
            module_name, class_name = state_class_name.rsplit('.', 1)
            state_class = None

            while state_class is None:
                # Import the module and get the class safely
                try:
                    state_class = getattr(importlib.import_module(module_name), class_name)
                    if not issubclass(state_class, BaseState):
                        raise ValueError(f"State class {state_class_name} is not a subclass of BaseState")
                except AttributeError:
                    if "." in module_name:
                        module_name = module_name.rsplit('.', 1)[0]
                    else:
                        break

            if state_class is None:
                raise ValueError(f"State class {state_class_name} not found")
            
            if full_info["state"] is None:
                raise ValueError(f"{state_class_name} state is None. Did you forget to implement to_json()?")

            state = state_class.init(full_info["state"])

            if state is None:
                raise ValueError(f"{state_class_name}.from_json returned None for {full_info['state']}")

            instance = cls._instances.get(instance_id)
            if instance:
                if not isinstance(instance, StatefulModule):
                    raise TypeError(f"Instance {instance_id} is not a StatefulModule")
                
                instance._state = state
            else:
                raise ValueError(f"Instance {instance_id} not found")
            
            cls._state_cache[instance_id] = state
    
    
    @classmethod
    def load_config(cls, config_path):
        """Load a configuration from a YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            
        return config
        
    @classmethod
    def instantiate_from_yaml(cls, config_path, module_class: type[BaseModule] | None = None):
        """Load a configuration from a YAML file and instantiate it."""
        config = cls.load_config(config_path)
        return cls.instantiate(config, module_class=module_class)
