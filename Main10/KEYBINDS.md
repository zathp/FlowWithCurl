# 3D Visualization Keybinds

## Simulation Control
| Key | Action |
|-----|--------|
| **SPACE** | Toggle pause/play simulation |
| **N** | Step simulation once (when paused) |
| **ESC** | Exit viewer |

## Render Modes
| Key | Action |
|-----|--------|
| **M** | Cycle render modes: Points → Vectors → Density → Points |
| **V** | Toggle between Points and Vectors mode (legacy) |
| **B** | Toggle vector field display: Flow ↔ Curl |
| **F** | Toggle density field: Field 1 ↔ Field 2 |

## Camera Zoom
| Key | Action |
|-----|--------|
| **+** / **=** | Zoom in (decrease camera radius) |
| **-** | Zoom out (increase camera radius) |
| **Mouse Wheel** | Zoom in/out |

## Camera Movement
| Key | Action |
|-----|--------|
| **I** | Move forward along view direction |
| **K** | Move backward along view direction |
| **J** | Strafe left |
| **L** | Strafe right |
| **H** | Move up (along world up vector) |
| **N** | Move down (along world down vector) |

## Camera Rotation
| Key | Action |
|-----|--------|
| **Left Mouse + Drag** | Rotate camera (change yaw and pitch) |
| **Middle Mouse + Drag** | Pan camera (move center point) |
| **Right Mouse + Drag** | Pan camera (move center point) |

## Camera Reset
| Key | Action |
|-----|--------|
| **R** | Reset camera to initial position and orientation |

## Debugging
| Key | Action |
|-----|--------|
| **P** | Print random particle sample (position and color) |
| **D** | Dump current simulation state to temp file (.npz) |

## Notes
- Camera movement speed scales with current zoom level (camera radius)
- Camera center is clamped to bounds: [-200, 200] in X/Y, [-50, 50] in Z
- Camera radius is clamped between 1.0 and 2000.0
- Pitch rotation is clamped to ±1.49 radians to prevent camera flip
