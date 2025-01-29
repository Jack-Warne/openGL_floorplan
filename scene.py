from chunk_loader import Chunk
from frustrum import Frustrum
from model import *
import numpy as np
from floorplanAnalysis import analysis
from settings import *

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
        flushed_doors = analysis.get_analysis(search_for='doors')
        room_data = analysis.get_analysis(search_for='rooms')
        room_types = []
        vertices = []

        # Iterate through the data and separate room types and vertices
        for item in room_data:
            room_types.append(item[0])  # Append the room type
            vertices.append(item[1])    # Append the vertices

        # Map room types to texture IDs
        texture_map = {
            'default': 3,  # Default texture ID for walls
            'bedroom': 5,  # Texture ID for bedroom walls
        }

        # Ensure no duplicate door coordinates
        flushed_doors = self.filter_duplicate_doors(flushed_doors)
        closest_wall = [entry['closest_wall'] for entry in flushed_doors]

        # Convert doors to 3D coordinates
        self.doors3D = [
            (x1 * 0.1, 0, y1 * 0.1) for (x1, y1), (x2, y2) in closest_wall
        ] + [
            (x2 * 0.1, 0, y2 * 0.1) for (x1, y1), (x2, y2) in closest_wall
        ]

        # Convert walls to 3D coordinates
        coordinates_3d = [(x * 0.1, 0, y * 0.1) for x, y in wall_coordinates]
        coordinates_3d.append(coordinates_3d[0])  # Close the loop

        # Floor
        n, s = 20, 2
        for x in range(-n, n, s):
            for z in range(-n, n, s):
                add(Cube(app, pos=(x, -s, z)))

        # Add cubes along the perimeter and group them into chunks
        self.fill_perimeter_with_cubes(coordinates_3d)

        # Fill rooms with flat surfaces at y = 0 based on room type
        self.fill_rooms_with_flat_surfaces(room_types, vertices, texture_map)

    def fill_rooms_with_flat_surfaces(self, room_types, vertices, texture_map):
        """
        Fill the interior area defined by room vertices with flat surfaces at y = 0,
        using textures based on room type.
        """
        app = self.app
        chunk_size = 10  # Chunk size in world units
        chunks = {}  # Store chunks by their grid coordinates

        for room_type, room_vertices in zip(room_types, vertices):
            # Get the texture ID for the room type

            # Convert vertices to 3D coordinates (x, 0, z)
            room_vertices_3d = [(x * 0.1, 0, y * 0.1) for x, y in room_vertices]

            # Extract 2D points (x, z) for the polygon check
            room_vertices_2d = [(x, z) for x, _, z in room_vertices_3d]

            # Create a grid to fill the room area
            min_x = min(v[0] for v in room_vertices_3d)
            max_x = max(v[0] for v in room_vertices_3d)
            min_z = min(v[2] for v in room_vertices_3d)
            max_z = max(v[2] for v in room_vertices_3d)

            # Step size for grid (adjust based on WALL_SCALE)
            grid_step = WALL_SCALE

            # Iterate over the grid and place flat surfaces at y = 0
            x = min_x
            while x <= max_x:
                z = min_z
                while z <= max_z:
                    # Check if the point (x, z) is inside the room polygon
                    if self.is_point_in_polygon((x, z), room_vertices_2d):
                        # Place a flat surface at y = 0
                        pos_with_y = (x, 0, z)  # y is fixed at 0

                        # Get or create the chunk for this position
                        chunk_coords = self.get_chunk_coords(pos_with_y, chunk_size)
                        chunk = self.get_or_create_chunk(chunks, chunk_coords, chunk_size, WALL_SCALE)

                        # Add the flat surface to the chunk
                        chunk.cubes.append(Wall(app, pos=tuple(pos_with_y), tex_id=room_type, scale=(WALL_SCALE, WALL_SCALE, WALL_SCALE)))
                    z += grid_step
                x += grid_step

        # Save chunks to self.all_chunks
        self.all_chunks.extend(chunks.values())

    def is_point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using the ray-casting algorithm.
        """
        x, z = point
        n = len(polygon)
        inside = False
        for i in range(n):
            x1, z1 = polygon[i]
            x2, z2 = polygon[(i + 1) % n]
            if z > min(z1, z2):
                if z <= max(z1, z2):
                    if x <= max(x1, x2):
                        if z1 != z2:
                            xinters = (z - z1) * (x2 - x1) / (z2 - z1) + x1
                        if z1 == z2 or x <= xinters:
                            inside = not inside
        return inside

    def filter_duplicate_doors(self, doors):
        """
        Filter out duplicate consecutive door coordinates to ensure that doors
        are correctly represented with distinct start and end points.
        """
        unique_doors = []
        for i in range(1, len(doors)):
            if doors[i] != doors[i-1]:  # Ensure no consecutive coordinates are the same
                unique_doors.append(doors[i-1])
        return unique_doors

    def fill_perimeter_with_cubes(self, wall_coordinates, max_y=24, height_step=1.0):
        """
        Fill the perimeter defined by wall coordinates with cubes and organize them into chunks,
        ensuring walls are not placed where doors are.
        """
        app = self.app
        height_step = height_step * WALL_SCALE
        chunk_size = 10  # Chunk size in world units
        chunks = {}  # Store chunks by their grid coordinates

        for i in range(len(wall_coordinates) - 1):
            start = np.array(wall_coordinates[i])
            end = np.array(wall_coordinates[i + 1])

            direction = end - start
            distance = np.linalg.norm(direction)
            direction = direction / distance
            num_cubes = int(distance // WALL_SCALE)
            y_values = np.arange(0, max_y * height_step, height_step)

            for y in y_values:
                for j in range(num_cubes):
                    pos = start + direction * WALL_SCALE * j
                    pos_with_y = (pos[0], y, pos[2])

                    chunk_coords = self.get_chunk_coords(pos_with_y, chunk_size)
                    chunk = self.get_or_create_chunk(chunks, chunk_coords, chunk_size, WALL_SCALE)
                    chunk.cubes.append(Wall(app, pos=tuple(pos_with_y), tex_id=3, scale=(WALL_SCALE, WALL_SCALE, WALL_SCALE)))

        # Save chunks to self.all_chunks
        self.all_chunks.extend(chunks.values())

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