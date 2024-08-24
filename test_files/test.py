import glfw
import numpy as np
import pywavefront
import moderngl as mgl

# Initialize GLFW
if not glfw.init():
    raise Exception("GLFW can't be initialized")

# Create a windowed mode window and its OpenGL context
window = glfw.create_window(800, 600, "3D Contour OBJ", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window can't be created")

# Make the window's context current
glfw.make_context_current(window)

# Initialize ModernGL context
ctx = mgl.create_context()

# Load the OBJ file
scene = pywavefront.Wavefront('contour.obj', create_materials=True, collect_faces=True)

# Extract vertex data from the loaded OBJ file
vertices = []
for name, mesh in scene.meshes.items():
    for face in mesh.faces:
        for vertex_i in face:
            vertices.extend(scene.vertices[vertex_i])

vertices = np.array(vertices, dtype=np.float32)

# Create a vertex buffer
vbo = ctx.buffer(vertices.tobytes())

# Create a vertex array object
vao = ctx.simple_vertex_array(
    ctx.program(
        vertex_shader=open('shaders/default.vert').read(),
        fragment_shader=open('shaders/default.frag').read()
    ),
    vbo,
    'in_position'
)

# Enable depth test
ctx.enable(mgl.DEPTH_TEST)

# Define the perspective projection matrix
projection = np.array([
    [1.0, 0.0,  0.0,  0.0],
    [0.0, 1.0,  0.0,  0.0],
    [0.0, 0.0, -2.0, -3.0],
    [0.0, 0.0, -1.0,  1.0]
], dtype=np.float32)

# Render loop
while not glfw.window_should_close(window):
    # Poll for and process events
    glfw.poll_events()

    # Clear the color and depth buffer
    ctx.clear(0.2, 0.3, 0.3)
    ctx.clear_depth(1.0)

    # Set the view matrix
    view = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -3.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Set the model matrix
    angle = glfw.get_time() * 50.0
    cos_angle = np.cos(np.radians(angle))
    sin_angle = np.sin(np.radians(angle))
    model = np.array([
        [cos_angle, 0.0, sin_angle, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-sin_angle, 0.0, cos_angle, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # Pass the matrices to the shader
    vao.program['projection'].write(projection.tobytes())
    vao.program['view'].write(view.tobytes())
    vao.program['model'].write(model.tobytes())

    # Render the object
    vao.render(mgl.TRIANGLES)

    # Swap front and back buffers
    glfw.swap_buffers(window)

# Clean up and exit
glfw.terminate()
