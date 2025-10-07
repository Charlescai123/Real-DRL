import os
import enum
import logging
from typing import Dict

import numpy as np


class TriggerType(enum.Enum):
    SELF = 0
    EVENT = 1
    UNKNOWN = 2


class ActionMode(enum.Enum):
    STUDENT = 1
    TEACHER = 2


class PlotMode(enum.Enum):
    POSITION = 1
    VELOCITY = 2
