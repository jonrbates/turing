from typing import Dict, Tuple

StateType = str
SymbolType = str
DirectionType = int
DeltaType = Dict[Tuple[StateType, SymbolType], Tuple[StateType,  SymbolType, DirectionType]]