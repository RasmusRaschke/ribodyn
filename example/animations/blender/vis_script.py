import bpy
import numpy as np
import mathutils


def remove_old_objects(context):
    bpy.context.scene.frame_set(0)
    for ob in context.scene.objects:
        if ob.type in ["MESH", "EMPTY", "FONT", "CURVE"]:
            if "Arrow" in ob.name or "safe" in ob.name or "pole" in ob.name or "Placeholder" in ob.name:
                continue
            bpy.ops.object.select_all(action="DESELECT")
            ob.select_set(True)
            bpy.ops.object.delete(use_global=False)


def update_time_label(scene):
    text_obj = bpy.data.objects.get("TimeLabel")
    if text_obj is None:
        return
    traj = scene.get("traj_data")
    if traj is None:
        return
    traj = np.array(traj)
    frame_rate = scene.render.fps
    slow_motion_factor = scene.get("slow_motion_factor", 10.0)
    i = scene.frame_current
    t = i / frame_rate / slow_motion_factor
    idx = np.searchsorted(traj[:, 0], t)
    if idx >= len(traj):
        idx = len(traj) - 1
    text_obj.data.body = f"t = {t:.3f} s"


if __name__ == "__main__":
    remove_old_objects(bpy.context)

    # Hide the old magnetic-scene geometry, but do not delete it. Some objects
    # in the .blend may also be used as camera/rig references.
    for ob in bpy.context.scene.objects:
        if ob.type == "MESH" and (
            "safe" in ob.name
            or "pole" in ob.name
            or "Placeholder" in ob.name
        ):
            ob.hide_viewport = True
            ob.hide_render = True

    # Load input file
    with open("input", "r") as f:
        input_path = f.readline().strip()
        output_path = f.readline().strip()

    input_para = {}
    with open(input_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            print(line)
            if line.strip() and not line.startswith("#"):
                l = line.strip().split()
                input_para[l[0]] = l[1:]
                for i in range(len(input_para[l[0]])):
                    try:
                        input_para[l[0]][i] = float(input_para[l[0]][i])
                    except:
                        pass
                if len(input_para[l[0]]) == 1:
                    input_para[l[0]] = input_para[l[0]][0]

    show_time = False

    # Plane normal from the solver input. Blender's primitive plane has local
    # +z as its normal, so rotate +z onto the physical plane normal.
    plane_normal = mathutils.Vector((
        input_para["normal"][0],
        input_para["normal"][1],
        input_para["normal"][2],
    ))
    plane_normal.normalize()

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    plane = bpy.context.active_object
    plane.rotation_mode = "QUATERNION"
    plane.rotation_quaternion = (
        mathutils.Vector((0, 0, 1))
        .rotation_difference(plane_normal)
    )
    plane.data.materials.append(bpy.data.materials.get("Plane"))

    # Add ball object
    ball_radius = input_para["radius"] * 100
    print(ball_radius)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_radius, location=(0, 0, 0), segments=64, ring_count=64)
    ball = bpy.context.active_object
    ball.name = "Ball"
    ball.data.materials.append(bpy.data.materials.get("Ball"))

    start_magnetization = np.array([
        input_para["magneticMoment"][0],
        input_para["magneticMoment"][1],
        input_para["magneticMoment"][2],
    ])

    # Keep the original carrier object, because the trajectory and quaternion
    # animation use it. The magnetic arrow is only created for a nonzero moment.
    mangetization_obj = bpy.data.objects.new("magnetisation", None)
    mangetization_obj.rotation_mode = "QUATERNION"
    bpy.context.scene.collection.objects.link(mangetization_obj)

    rotation_quat = mathutils.Quaternion((1.0, 0.0, 0.0, 0.0))

    if np.linalg.norm(start_magnetization) > 0.0:
        start_magnetization = (
            start_magnetization
            / np.linalg.norm(start_magnetization)
        )

        arrow = bpy.data.objects.get("Arrow")

        if arrow is None:
            print("Error: Arrow object not found in the scene.")
        else:
            arrow_copy = arrow.copy()
            arrow_copy.data = arrow.data.copy()

            arrow_copy.name = "arrow"
            arrow_copy.hide_render = False

            bpy.context.scene.collection.objects.link(arrow_copy)
            arrow_copy.parent = mangetization_obj
            arrow_copy.location = (0, 0, 0)
            arrow_copy.rotation_euler = (0, 0, 0)
            arrow_copy.scale = (
                ball_radius * 1.2,
                ball_radius * 1.2,
                ball_radius * 1.2,
            )

            default_dir = mathutils.Vector((0, 0, 1))
            target_dir = mathutils.Vector(start_magnetization)
            rotation_quat = default_dir.rotation_difference(target_dir)

            arrow_copy.rotation_mode = "QUATERNION"
            arrow_copy.rotation_quaternion = rotation_quat

    ball.parent = mangetization_obj
    ball.rotation_mode = "QUATERNION"
    ball.rotation_quaternion = rotation_quat

    if show_time:
        bpy.ops.object.text_add(enter_editmode=False, align="VIEW", location=(0, 0, 0))
        text_obj = bpy.context.active_object
        text_obj.name = "TimeLabel"
        text_obj.data.size = ball_radius * 0.8
        text_obj.data.align_x = "CENTER"
        text_obj.rotation_euler = (np.radians(90), 0, np.radians(90))

        text_offset_z = ball_radius * 2.5
        text_obj.location = (0, 0, text_offset_z)
        loc_constraint = text_obj.constraints.new(type="COPY_LOCATION")
        loc_constraint.target = mangetization_obj
        loc_constraint.use_offset = True

    # Load trajectory
    traj = np.loadtxt(output_path, delimiter=",", skiprows=1)
    frame_rate = 24
    slow_motion_factor = 0.6
    bpy.context.scene.render.fps = int(frame_rate)
    total_time = traj[-1, 0]
    total_frames = int(total_time * frame_rate * slow_motion_factor) + 1
    bpy.context.scene.frame_end = total_frames

    # Store traj and slow_motion_factor in the scene for the handler to access
    bpy.context.scene["traj_data"] = traj.tolist()
    bpy.context.scene["slow_motion_factor"] = slow_motion_factor

    bpy.context.view_layer.update()
    world_to_plane = plane.matrix_world.inverted()

    plane_points = np.array([
        (world_to_plane @ mathutils.Vector((
            row[1] * 100,
            row[2] * 100,
            row[3] * 100,
        )))[:2]
        for row in traj
    ])

    max_x, max_y = np.max(plane_points[:, 0]), np.max(plane_points[:, 1])
    min_x, min_y = np.min(plane_points[:, 0]), np.min(plane_points[:, 1])

    mesh = plane.data
    o = ball_radius * 2
    mesh.vertices[1].co = (max_x + o, min_y - o, 0)
    mesh.vertices[0].co = (min_x - o, min_y - o, 0)
    mesh.vertices[2].co = (min_x - o, max_y + o, 0)
    mesh.vertices[3].co = (max_x + o, max_y + o, 0)

    # Use the existing camera exactly as stored in the .blend.
    # Do not change its lens, rotation or initial position.
    scene = bpy.context.scene
    camera = scene.camera

    if camera is None:
        cameras = [
            ob for ob in scene.objects
            if ob.type == "CAMERA"
        ]

        if not cameras:
            raise RuntimeError(
                "No camera object found in the current Blender scene."
            )

        if len(cameras) > 1:
            print(
                "Scene has no active camera. Available cameras:",
                [ob.name for ob in cameras],
            )

        camera = cameras[0]
        scene.camera = camera

    camera_start_world = camera.matrix_world.translation.copy()
    com_start_world = mathutils.Vector((
        traj[0, 1] * 100,
        traj[0, 2] * 100,
        traj[0, 3] * 100,
    ))

    empty = bpy.data.objects.new("Empty", None)
    empty_obj = bpy.context.active_object

    # Create a curve for the trajectory that follows empty
    curve = bpy.ops.curve.primitive_bezier_curve_add(radius=1, enter_editmode=False, align="WORLD", location=(0, 0, 0), scale=(1, 1, 1))
    curve_obj = bpy.context.active_object
    curve_obj.name = "TrajectoryCurve"
    curve_obj.data.bevel_depth = ball_radius * 0.1
    curve_obj.data.bevel_resolution = 4
    curve_obj.data.materials.append(bpy.data.materials.get("Blue"))
    spline = curve_obj.data.splines[0]
    positions = []
    ball_center = []

    # bpy.ops.mesh.primitive_uv_sphere_add(radius=ball_radius * 0.1, location=(0, 0, 0), segments=16, ring_count=16)
    # magnetization_indicator = bpy.context.active_object
    # magnetization_indicator.name = "MagnetizationIndicator"

    frames_to_skip = 2
    sampled_frames = list(range(0, total_frames, frames_to_skip))

    if sampled_frames[-1] != total_frames - 1:
        sampled_frames.append(total_frames - 1)

    spline.bezier_points.add(len(sampled_frames) - 1)

    for j, i in enumerate(sampled_frames):

        t = i / frame_rate / slow_motion_factor
        idx = np.searchsorted(traj[:, 0], t)
        if idx >= len(traj):
            idx = len(traj) - 1

        mangetization_obj.location[0] = traj[idx, 1] * 100
        mangetization_obj.location[1] = traj[idx, 2] * 100
        mangetization_obj.location[2] = traj[idx, 3] * 100
        mangetization_obj.keyframe_insert(data_path="location", frame=i)

        # Translate the camera by the same COM displacement as the sphere.
        # This preserves the original camera angle, distance, lens and framing.
        current_com_world = mathutils.Vector((
            traj[idx, 1] * 100,
            traj[idx, 2] * 100,
            traj[idx, 3] * 100,
        ))

        desired_camera_world = (
            camera_start_world
            + current_com_world
            - com_start_world
        )

        if camera.parent is None:
            camera.location = desired_camera_world
        else:
            camera.location = (
                camera.parent.matrix_world.inverted()
                @ desired_camera_world
            )

        camera.keyframe_insert(
            data_path="location",
            frame=i,
        )
        mx, my, mz = traj[idx, -3], traj[idx, -2], traj[idx, -1]
        mag = np.sqrt(mx**2 + my**2 + mz**2)

        if mag > 0.0:
            mx /= mag
            my /= mag
            mz /= mag

        mangetization_obj.rotation_quaternion[0] = traj[idx, 7]
        mangetization_obj.rotation_quaternion[1] = traj[idx, 8]
        mangetization_obj.rotation_quaternion[2] = traj[idx, 9]
        mangetization_obj.rotation_quaternion[3] = traj[idx, 10]
        mangetization_obj.keyframe_insert(data_path="rotation_quaternion", frame=i)

        if i == 0:
            bpy.context.scene.collection.objects.link(empty)
            empty.location = (0, 0, ball_radius)
            empty.parent = mangetization_obj

        # Force the dependency graph to recalculate all world matrices
        bpy.context.view_layer.update()

        # Now matrix_world reflects the parent's current rotation/location
        world_pos = empty.matrix_world.translation
        print(f"Frame {i}: World Position = {world_pos}")
        positions.append((world_pos[0], world_pos[1], world_pos[2]))
        ball_pos = ball.matrix_world.translation
        ball_center.append((ball_pos[0], ball_pos[1], ball_pos[2]))
        spline.bezier_points[j].co = world_pos
        spline.bezier_points[j].handle_left_type = "AUTO"
        spline.bezier_points[j].handle_right_type = "AUTO"

        curve_obj.data.bevel_factor_end = i / float(total_frames - 1)
        curve_obj.data.keyframe_insert(data_path="bevel_factor_end", frame=i)

    np.savetxt("trajectory_points.csv", np.array(positions), delimiter=",", header="x,y,z", comments="")
    np.savetxt("ball_center.csv", np.array(ball_center), delimiter=",", header="x,y,z", comments="")

    if show_time:
        # Register the frame change handler (remove old one first to avoid duplicates)
        for handler in bpy.app.handlers.frame_change_post:
            if handler.__name__ == "update_time_label":
                bpy.app.handlers.frame_change_post.remove(handler)

        bpy.app.handlers.frame_change_post.append(update_time_label)

        # Also register for render so the text updates during rendering
        for handler in bpy.app.handlers.render_pre:
            if handler.__name__ == "update_time_label":
                bpy.app.handlers.render_pre.remove(handler)

        bpy.app.handlers.render_pre.append(update_time_label)

        # Set initial text
        update_time_label(bpy.context.scene)

        print("Done. Handler registered for live text updates.")
