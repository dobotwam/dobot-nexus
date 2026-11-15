"""Dimension contracts shared by standalone robot communication bridges."""
from dataclasses import dataclass
from collections.abc import Mapping

import numpy as np


def positive_dimension(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer, got {value!r}.')
    return int(value)


@dataclass(frozen=True)
class CommunicationSpec:
    state_dim: int = 14
    action_dim: int = 14
    action_horizon: int | None = None

    def __post_init__(self):
        positive_dimension(self.state_dim, 'state_dim')
        positive_dimension(self.action_dim, 'action_dim')
        if self.action_horizon is not None:
            positive_dimension(self.action_horizon, 'action_horizon')

    @classmethod
    def from_metadata(cls, metadata):
        if not isinstance(metadata, Mapping):
            raise TypeError('Dimension metadata must be an object.')
        sources = [metadata]
        nested = metadata.get('policy_spec')
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise TypeError('policy_spec must be an object.')
            sources.append(nested)
        shapes = {}
        for schema_name, field, width in (('input_schema', 'state', 1),
                                          ('output_schema', 'actions', 2)):
            schema = metadata.get(schema_name, {})
            if not isinstance(schema, Mapping):
                raise TypeError(f'{schema_name} must be an object.')
            shape = schema.get(field)
            if shape is not None:
                if not isinstance(shape, (list, tuple)) or len(shape) != width:
                    raise ValueError(f'{schema_name}.{field} has an invalid shape: {shape!r}.')
                if field == 'state':
                    shapes['state_dim'] = shape[0]
                else:
                    shapes.update(action_horizon=shape[0], action_dim=shape[1])
        sources.append(shapes)
        values = {}
        for name in ('state_dim', 'action_dim', 'action_horizon'):
            candidates = [positive_dimension(s[name], name) for s in sources
                          if s.get(name) is not None]
            if candidates:
                if len(set(candidates)) != 1:
                    raise ValueError(f'Conflicting {name} in communication metadata: {candidates}.')
                values[name] = candidates[0]
        if ('state_dim' in values) != ('action_dim' in values):
            raise ValueError('Metadata must declare both state_dim and action_dim.')
        # Metadata without dimensions belongs to the original 14D bridge.
        return cls(**values)

    def validate_robot(self, hello):
        robot = self.from_metadata(hello)
        if (robot.state_dim, robot.action_dim) != (self.state_dim, self.action_dim):
            raise ValueError(
                f'Robot dimensions state={robot.state_dim}, action={robot.action_dim} '
                f'do not match policy state={self.state_dim}, action={self.action_dim}. '
                'Custom robot servers must advertise their state_dim/action_dim and '
                'provide the matching hardware mapping.'
            )

    def validate_state(self, state):
        array = np.asarray(state, dtype=np.float32)
        if array.shape != (self.state_dim,) or not np.isfinite(array).all():
            raise ValueError(f'Expected finite state shape ({self.state_dim},), got {array.shape}.')
        return array

    def validate_actions(self, actions):
        array = np.asarray(actions, dtype=np.float32)
        if (array.ndim != 2 or array.shape[1] != self.action_dim or not len(array)
                or not np.isfinite(array).all()):
            raise ValueError(f'Expected finite actions [H,{self.action_dim}], got {array.shape}.')
        if self.action_horizon is not None and array.shape[0] != self.action_horizon:
            raise ValueError(f'Expected action_horizon={self.action_horizon}, got {array.shape[0]}.')
        return array
