import numpy as np

ft = 0.3048  # ft -> m
kts = 0.514444  # knot -> m/s

@ np.vectorize
def Pa_to_ft(value: float) -> float:
    """
    Convert Pa to ft given an altitude
    
    Parameters
    ----------
    value: float
        value in Pa
        
    Returns
    -------
    result: float
        value in ft
    """
    altitude_m: float = 0.0
    if value >= 22_632:
        altitude_m = 44_330.769 - 4_946.5463 * np.power(value, 0.19026311)
    
    elif value >= 5_464.87:
        altitude_m = 74_588.142 - 6_341.6156 * np.log(value)
    
    elif value >= 868.014:
        altitude_m = -196_650 + 278_731.18 * np.power(value, -0.029271247)
    
    elif value >= 110.906:
        altitude_m = -49_660.714 + 142_184.85 * np.power(value, -0.081959491)
    
    else:
        altitude = 84_303.425 - 7_922.2630 * np.log(value)
    
    altitude = altitude_m / ft
    
    return altitude