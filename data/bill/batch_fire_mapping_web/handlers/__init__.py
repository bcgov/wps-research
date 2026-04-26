"""HTTP handler mixins composed into FireHandler in app.py."""
from .base import BaseHandler
from .auth import AuthRoutes
from .fire_list import FireListRoutes
from .fire import FireRoutes
from .mapping import MappingRoutes
from .serial import SerialRoutes
from .rebrush import RebrushRoutes
from .batch import BatchRoutes
from .ops import OpsRoutes
from .static import StaticRoutes

