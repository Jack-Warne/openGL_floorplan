from model import *
from floorplanAnalysis import analysis
import glm
import numpy as np

class Scene:
    def __init__(self, app):
        self.app = app
        #analysis = self.analysis
        self.objects = []
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)



 

    def load(self):
        app = self.app
        add = self.add_object

        # Retrieve wall coordinates from the analysis
        wall_coordinates = analysis.get_analysis(search_for='walls')
        wall_scale = analysis.get_analysis(search_for='scale')

        # Convert to 3D coordinates
        coordinates_3d = [(x, 0, y) for x, y in wall_coordinates]
        coordinates_3d.append(coordinates_3d[0])
        print(coordinates_3d)

        # Define the length of a single wall segment
        wall_segment_length = 0.1  # You can adjust this value to the size of your wall segments

        # Add walls between each pair of coordinates
        for i in range(len(coordinates_3d) - 1):
            start = np.array(coordinates_3d[i])
            end = np.array(coordinates_3d[i + 1])

            # Calculate direction vector and distance between the two points
            direction = end - start
            distance = np.linalg.norm(direction)

            # Normalize the direction vector to move by wall_segment_length
            direction = direction / np.linalg.norm(direction)

            # Number of segments to place between the two points
            num_segments = int(distance // wall_segment_length)

            # Place walls along the line between the two points
            for j in range(num_segments):
                # Calculate the position for the wall segment
                pos = start + direction * wall_segment_length * j
                add(Wall(app, pos=tuple(pos),  tex_id=3, scale=(wall_scale, wall_scale, wall_scale)))

        # Floor (unchanged)
        n, s = 20, 2
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                add(Cube(app, pos=(x, -s, z)))




    def update(self): ...
        #self.moving_cube.rot.xyz = self.app.time
