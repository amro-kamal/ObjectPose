from renderer import Renderer
from strike_utils import *

amb_int=0.7
dif_int=1.0
dirlight = [0.3, 0.1, 1.0]
x, y, z = 0, 0, -2
save = False
predict = False

if __name__ == "__main__":
    # Initialize neural network.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(device).to(device)

    # Initialize renderer.
    renderer = Renderer(
        "objects/fireengine/fireengine.obj", "objects/fireengine/fireengine.mtl"
    )

    # Render scene
    image = renderer.render()
    if save:
        image.save("pose_1.png")
    else:
        image.show()

    # Get neural network probabilities.
    if predict:
        with torch.no_grad():
            out = model(image)

        probs = torch.nn.functional.softmax(out, dim=1)
        target_class = 609
        print(probs[0][target_class].item())

    # Alter renderer parameters.
    R_obj = gen_rotation_matrix(np.pi / 4, np.pi / 4, 0)
    renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())
    renderer.prog["x"].value = x
    renderer.prog["y"].value = y
    renderer.prog["z"].value = z

    renderer.prog["amb_int"].value = amb_int
    renderer.prog["dif_int"].value = dif_int
    DirLight = np.array(dirlight)
    DirLight /= np.linalg.norm(DirLight)
    renderer.prog["DirLight"].value = tuple(DirLight)

    # Render new scene.
    image = renderer.render()
    if save:
        image.save("pose_2.png")
    else:
        image.show()

    # Get neural network probabilities.
    if predict:
        with torch.no_grad():
            out = model(image)

        probs = torch.nn.functional.softmax(out, dim=1)
        print(probs[0][target_class].item())

    # Get depth map.
    depth_map = renderer.get_depth_map()
    if save:
        depth_map.save("depth_map.png")
    else:
        depth_map.show()

    # Get normal map.
    norm_map = renderer.get_normal_map()
    if save:
        norm_map.save("normal_map.png")
    else:
        norm_map.show()

    # Get screen coordinates of vertices.
    (screen_coords, screen_img) = renderer.get_vertex_screen_coordinates()
    if save:
        screen_img.save("screen_coords.png")
    else:
        screen_img.show()

    # Use azimuth and elevation to generate rotation matrix.
    R_obj = gen_rotation_matrix_from_azim_elev(np.pi / 4, np.pi / 4)
    renderer.prog["R_obj"].write(R_obj.T.astype("f4").tobytes())

    image = renderer.render()
    if save:
        image.save("camera.png")
    else:
        image.show()


