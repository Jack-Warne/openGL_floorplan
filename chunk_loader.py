from frustrum import Frustrum

class Chunk:
    def __init__(self, chunk_pos, chunk_size, cube_size):
        """
        Initializes a chunk.
        :param chunk_pos: 3D position of the chunk (world coordinates of its corner).
        :param chunk_size: Length of one edge of the chunk in world units.
        :param cube_size: The size of individual cubes in this chunk.
        """
        self.chunk_pos = chunk_pos  # (x, y, z) position of the chunk in world space
        self.chunk_size = chunk_size  # Size of the chunk (assumed cubic)
        self.cube_size = cube_size  # Size of individual cubes
        self.cubes = []  # List of cube objects within the chunk

    def is_inside_frustum(self, frustum_planes):
        """
        Check if this chunk's AABB intersects the view frustum.
        """
        aabb_min = self.chunk_pos
        aabb_max = (
            self.chunk_pos[0] + self.chunk_size,
            self.chunk_pos[1] + self.chunk_size,
            self.chunk_pos[2] + self.chunk_size,
        )
        return Frustrum.is_in_frustum(aabb_min, aabb_max, frustum_planes)
