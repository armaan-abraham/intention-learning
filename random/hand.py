import mujoco
import numpy as np
from mujoco import MjModel, MjData
from PIL import Image

# Load the Allegro Hand MJCF model
model_path = 'path/to/right_hand.xml'  # Update this path to your actual file location
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Set up the scene and context for rendering
pixels_width = 800
pixels_height = 600
camera = mujoco.MjvCamera()
option = mujoco.MjvOption()
scene = mujoco.MjvScene(model, maxgeom=2000)
context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

# Function to save an image of the current state
def save_current_state_image(model, data, scene, context, filename):
    # Update the scene
    mujoco.mjv_updateScene(
        model, data, option, None, camera,
        mujoco.mjtCatBit.mjCAT_ALL, scene
    )
    # Render the scene to an offscreen buffer
    viewport = mujoco.MjrRect(0, 0, pixels_width, pixels_height)
    mujoco.mjr_render(viewport, scene, context)
    # Create empty image buffer
    rgb = np.zeros((pixels_height, pixels_width, 3), dtype=np.uint8)
    # Read the pixels into the buffer
    mujoco.mjr_readPixels(rgb, None, viewport, context)
    # Flip the image vertically
    rgb = np.flipud(rgb)
    # Save the image using PIL
    img = Image.fromarray(rgb)
    img.save(filename)

# Set a random state
data.qpos = np.random.uniform(
    low=model.jnt_range[:, 0],
    high=model.jnt_range[:, 1],
    size=model.nq
)
data.qvel = np.random.randn(model.nv)

# Step the simulation to update the state
mujoco.mj_step(model, data)

# Save the image of the current state
save_current_state_image(
    model, data, scene, context,
    'allegro_hand_random_state.png'
)
