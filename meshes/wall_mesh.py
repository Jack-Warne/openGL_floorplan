import numpy as np
from floorplanAnalysis import analysis
from shader_program import ShaderProgram
from meshes.base_mesh import BaseMesh

class WallMesh(BaseMesh):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.ctx = app.ctx
        
        # Use the correct shader program for walls
        self.program = ShaderProgram(self.ctx)
        #self.program = self.shader_program.programs['walls']
        
        # Set the texture ID for walls (assuming texture ID 3 is for walls)
        self.texture = self.mesh.texture.textures[3]

        self.vbo_format = '2f 3f 3f'
        self.attrs = ['in_tex_coord_0', 'in_normal', 'in_position']
        
        # Create VAO using get_vao() method from BaseMesh
        self.vao = self.get_vao()

    @staticmethod
    def get_data(vertices, indices):
        data = [vertices[ind] for triangle in indices for ind in triangle]
        return np.array(data, dtype='f4')

    def get_vertex_data(self):
        # Define the wall vertices and indices
        vertices = [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
                    (-1, 1, -1), (-1, -1, -1), (1, -1, -1), (1, 1, -1)]
        
        # Get wall scale from analysis
        wall_scale = analysis.get_analysis(search_for='scale')
        vertices = [(x * wall_scale, y * wall_scale, z * wall_scale) for (x, y, z) in vertices]

        # Wall faces
        indices = [(0, 2, 3), (0, 1, 2),
                   (1, 7, 2), (1, 6, 7),
                   (6, 5, 4), (4, 7, 6),
                   (3, 4, 5), (3, 5, 0),
                   (3, 7, 4), (3, 2, 7),
                   (0, 6, 1), (0, 5, 6)]
        
        # Convert vertices and indices to an array format
        vertex_data = self.get_data(vertices, indices)

        # Define texture coordinates for each vertex
        tex_coord_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
        tex_coord_indices = indices
        tex_coord_data = self.get_data(tex_coord_vertices, tex_coord_indices)

        # Define normals for the faces
        normals = [(0, 0, 1)] * 6 + [(1, 0, 0)] * 6 + [(0, 0, -1)] * 6 + \
                  [(-1, 0, 0)] * 6 + [(0, 1, 0)] * 6 + [(0, -1, 0)] * 6
        normals = np.array(normals, dtype='f4').reshape(36, 3)

        # Combine texture coordinates, normals, and vertex data
        vertex_data = np.hstack([tex_coord_data, normals, vertex_data])
        return vertex_data

    def render(self):
        # Use the wall texture
        self.texture.use(location=0)
        
        # Bind the shader program
        self.program['u_texture_0'] = 0
        self.program['m_proj'].write(self.app.camera.m_proj)
        self.program['m_view'].write(self.app.camera.m_view)
        
        # Render the VAO
        self.vao.render()
