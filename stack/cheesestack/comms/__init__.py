"""Communication layer: vehicle networks, CAN, DBC, logging."""

from .can import CanFrame, VirtualBus, VirtualBusConfig, VirtualCanNetwork
from .canlog import CandumpReader, CandumpWriter, StructuredLogger
from .dbc import Database, load_dbc, load_default_database, parse_dbc
from .decoding import DecodedMessage, FrameDecoder
from .diagnostics import UdsClient, UdsServer
from .extraction import SignalExtractor, extract
from .monitoring import NetworkMonitor, TimingValidator
from .serialization import DriveCommandCodec, EncoderConfig, encode_frame
from .bus_health import BusHealthMonitor, BusLoadMonitor

__all__ = [
    "BusHealthMonitor",
    "BusLoadMonitor",
    "CanFrame",
    "CandumpReader",
    "CandumpWriter",
    "Database",
    "DecodedMessage",
    "DriveCommandCodec",
    "EncoderConfig",
    "FrameDecoder",
    "NetworkMonitor",
    "SignalExtractor",
    "StructuredLogger",
    "TimingValidator",
    "UdsClient",
    "UdsServer",
    "VirtualBus",
    "VirtualBusConfig",
    "VirtualCanNetwork",
    "encode_frame",
    "extract",
    "load_dbc",
    "load_default_database",
    "parse_dbc",
]
