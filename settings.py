from floorplanAnalysis import analysis

# epsilon value used for near perfect values
EPSILON = 1e-5

# get wall scale once and use globally
WALL_SCALE = analysis.get_analysis(search_for='scale')


# camera settings
FOV = 50  # deg
NEAR = 0.1
FAR = 100
SPEED = 0.005
SENSITIVITY = 0.04