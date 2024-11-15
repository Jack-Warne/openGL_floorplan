class SpatialHash:
    def __init__(self, cell_size=1):
        self.cell_size = cell_size
        self.grid = {}

    def hash_position(self, pos):
        # Convert 3D position to a cell coordinate
        return tuple(int(coord // self.cell_size) for coord in pos)

    def add(self, pos):
        hashed = self.hash_position(pos)
        if hashed not in self.grid:
            self.grid[hashed] = True
            return True  # Block was successfully added (no overlap)
        return False  # Block was already present

    def is_occupied(self, pos):
        hashed = self.hash_position(pos)
        return hashed in self.grid
