from chunk_loader import Chunk
from frustrum import Frustrum
from model import *
import numpy as np
from floorplanAnalysis import analysis

class Scene:
    def __init__(self, app):
        self.app = app
        self.objects = []
        self.all_chunks = []  # Store all chunks
        self.load()
        # skybox
        self.skybox = AdvancedSkyBox(app)

    def add_object(self, obj):
        self.objects.append(obj)

    def load(self):
        app = self.app
        add = self.add_object

        # Wall data
        wall_coordinates = analysis.get_analysis(search_for='walls')
        wall_scale = analysis.get_analysis(search_for='scale')

        coordinates_3d = [(x, 0, y) for x, y in wall_coordinates]
        coordinates_3d.append(coordinates_3d[0])  # Close the loop

        # Floor
        n, s = 20, 2
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                add(Cube(app, pos=(x, -s, z)))

        # Add cubes along the perimeter and group them into chunks
        self.fill_perimeter_with_cubes(coordinates_3d, wall_scale)

    def fill_perimeter_with_cubes(self, wall_coordinates, wall_scale, max_y=240, height_step=1.0):
        """
        Fill the perimeter defined by wall coordinates with cubes and organize them into chunks.
        """
        app = self.app
        height_step = height_step * wall_scale
        chunk_size = 10  # Chunk size in world units
        chunks = {}  # Store chunks by their grid coordinates

        for i in range(len(wall_coordinates) - 1):
            start = np.array(wall_coordinates[i])
            end = np.array(wall_coordinates[i + 1])

            direction = end - start
            distance = np.linalg.norm(direction)
            direction = direction / distance
            num_cubes = int(distance // wall_scale)
            y_values = np.arange(0, max_y + height_step, height_step)

            for y in y_values:
                for j in range(num_cubes):
                    pos = start + direction * wall_scale * j
                    pos_with_y = (pos[0], y, pos[2])
                    chunk_coords = self.get_chunk_coords(pos_with_y, chunk_size)
                    chunk = self.get_or_create_chunk(chunks, chunk_coords, chunk_size, wall_scale)
                    chunk.cubes.append(Wall(app, pos=tuple(pos_with_y), tex_id=3, scale=(wall_scale, wall_scale, wall_scale)))

        # Save chunks to self.all_chunks
        self.all_chunks = list(chunks.values())

    def get_chunk_coords(self, position, chunk_size):
        """
        Get chunk grid coordinates for a given position.
        """
        return (
            int(position[0] // chunk_size),
            int(position[1] // chunk_size),
            int(position[2] // chunk_size),
        )

    def get_or_create_chunk(self, chunks, chunk_coords, chunk_size, cube_size):
        """
        Retrieve or create a chunk at the specified grid coordinates.
        """
        if chunk_coords not in chunks:
            chunks[chunk_coords] = Chunk(
                chunk_pos=(
                    chunk_coords[0] * chunk_size,
                    chunk_coords[1] * chunk_size,
                    chunk_coords[2] * chunk_size,
                ),
                chunk_size=chunk_size,
                cube_size=cube_size,
            )
        return chunks[chunk_coords]

    def update(self):
        """
        Cull chunks outside the view frustum and update visible objects.
        """
        proj_view_matrix = self.app.camera.m_proj * self.app.camera.m_view
        frustum_planes = Frustrum.extract_frustum_planes(proj_view_matrix)

        visible_chunks = []
        for chunk in self.all_chunks:
            if chunk.is_inside_frustum(frustum_planes):
                visible_chunks.append(chunk)

        self.objects = []
        for chunk in visible_chunks:
            self.objects.extend(chunk.cubes)

