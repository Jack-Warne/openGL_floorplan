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
        self.fill_perimeter_with_cubes(coordinates_3d, wall_scale)

    

    def fill_perimeter_with_cubes(self, wall_coordinates, wall_scale, max_y=240, height_step=1.0):
        """
        Fill the perimeter defined by wall coordinates with non-overlapping cubes stacked
        from y=0 to y=max_y with a step of height_step (which can now be a float).
        """
        app = self.app
        add = self.add_object
        height_step = height_step * wall_scale
        # Loop through each wall segment
        for i in range(len(wall_coordinates) - 1):
            start = np.array(wall_coordinates[i])
            end = np.array(wall_coordinates[i + 1])

            # Calculate direction vector and distance between points
            direction = end - start
            distance = np.linalg.norm(direction)
            direction = direction / distance  # Normalize the direction vector

            # Number of cubes to place along this wall segment (in the horizontal plane)
            num_cubes = int(distance // wall_scale)

            # Use np.arange to generate float steps for the Y-axis (from 0 to max_y, with height_step)
            y_values = np.arange(0, max_y + height_step, height_step)

            # Loop through each level of height from 0 to max_y with a step of height_step
            for y in y_values:
                # Place cubes along the perimeter at regular intervals and at height y
                for j in range(num_cubes):
                    # Calculate the horizontal position of the cube
                    pos = start + direction * wall_scale * j
                    # Add the vertical component to the position (set y-coordinate)
                    pos_with_y = (pos[0], y, pos[2])
                    # Add cube at the calculated position
                    add(Cube(app, pos=tuple(pos_with_y), tex_id=3, scale=(wall_scale, wall_scale, wall_scale)))




    def update(self):
        pass
