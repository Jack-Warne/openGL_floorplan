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
        Check if this chunk's bounding sphere intersects the view frustum.
        :param frustum_planes: Frustum planes for culling.
        :return: True if inside the frustum, False otherwise.
        """
        chunk_center = (
            self.chunk_pos[0] + self.chunk_size / 2,
            self.chunk_pos[1] + self.chunk_size / 2,
            self.chunk_pos[2] + self.chunk_size / 2,
        )
        radius = self.chunk_size / 2
        return Frustrum.is_inside_frustum(chunk_center, frustum_planes, radius)
