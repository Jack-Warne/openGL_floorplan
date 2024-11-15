from model import *
from floorplanAnalysis import analysis
import glm
import numpy as np

class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        # Retrieve wall coordinates and scaling from the analysis
        wall_coordinates = analysis.get_analysis(search_for='walls')
        wall_scale = analysis.get_analysis(search_for='scale')

        # Convert 2D wall coordinates to 3D by adding a Y axis
        coordinates_3d = [(x, 0, y) for x, y in wall_coordinates]
        coordinates_3d.append(coordinates_3d[0])  # Close the loop

        #print("Wall coordinates (3D):", coordinates_3d)

        # Floor 
        n, s = 20, 2
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                add(Cube(app, pos=(x, -s, z)))

        # Add cubes along the perimeter only
        self.fill_perimeter_with_spaced_cubes(coordinates_3d, wall_scale)

    def fill_perimeter_with_spaced_cubes(self, wall_coordinates, wall_scale):
        """
        Fill only the perimeter defined by wall coordinates with cubes spaced 2 units apart.
        """
        app = self.app
        add = self.add_object

        # Define the length of each wall cube
        cube_length = wall_scale  # Assuming the cube size is equal to wall_scale

        # Loop through each wall segment
        for i in range(len(wall_coordinates) - 1):
            start = np.array(wall_coordinates[i])
            end = np.array(wall_coordinates[i + 1])

            # Calculate direction vector and distance between points
            direction = end - start
            distance = np.linalg.norm(direction)
            
            # Normalize the direction to unit length
            direction = direction / np.linalg.norm(direction)

            # Calculate the number of cubes needed for this segment, skipping every other cube explaned in documentation
            num_cubes = int(distance // (cube_length * 2))  # Spacing cubes by 2 units

            # Add cubes along the perimeter with spacing
            for j in range(num_cubes + 1):  # +1 to include the first cube
                pos = start + direction * (cube_length * 2) * j
                add(Cube(app, pos=tuple(pos), tex_id=3, scale=(wall_scale, wall_scale, wall_scale)))

    def update(self):
        pass
