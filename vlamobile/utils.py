import mujoco


def is_body_in_contact(model, data, body1_name, body2_name):
    """
    Checks whether two MuJoCo bodies are in contact.

    Args:
        model: mujoco.MjModel object
        data: mujoco.MjData object
        body1_name (str): Name of the first body
        body2_name (str): Name of the second body

    Returns:
        bool: True if body1 is in contact with body2, False otherwise
    """
    body1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body1_name)
    body2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body2_name)

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_body = model.geom_bodyid[contact.geom1]
        geom2_body = model.geom_bodyid[contact.geom2]
        if (geom1_body == body1_id and geom2_body == body2_id) or \
           (geom1_body == body2_id and geom2_body == body1_id):
            return True

    return False


def is_body_on_floor(model, data, body_name, floor_name="floor"):
    """
    Checks whether a MuJoCo body is in contact with the floor.

    Args:
        model: mujoco.MjModel object
        data: mujoco.MjData object
        body_name (str): Name of the body to check
        floor_name (str): Name of the floor body (default: "floor")

    Returns:
        bool: True if body is in contact with the floor, False otherwise
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, floor_name)

    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_body = model.geom_bodyid[contact.geom1]
        geom2_body = model.geom_bodyid[contact.geom2]
        if (geom1_body == body_id and geom2_body == floor_id) or \
           (geom1_body == floor_id and geom2_body == body_id):
            return True

    return False
