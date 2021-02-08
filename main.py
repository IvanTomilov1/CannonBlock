import numpy as np
from objects_file import CannonBall, CatchPlate, canon_settings
from draw_file import draw_borders, draw_cannonball, draw_plate, draw_cannon
import cv2


#######################################################################################################
plane_size = (1500, 800)  # Size of a plane
radius = 20  # Radius of a ball
plate_velocity = 125  # Velocity of a plate
dt = 0.3  # Time discretization value
acceleration = np.array([0, -9.81])  # Acceleration in gravitational field on the Earth near the ground
########################################################################################################
# Define plate object
plate = CatchPlate(center_height=plane_size[1] // 2,
                   plate_x=plane_size[0] - 10,
                   width=2 * radius,
                   plate_velocity=plate_velocity,
                   min_height=0,
                   max_height=plane_size[1],
                   dt=dt)
#########################################################################################################
# Set cannon initial parameters
cannon_coord = np.array([50., 128.])
cannon_length = radius * 2
cannon_width = radius
#########################################################################################################

while True:
    plate.free_track()  # Free the memory of a plane

    x0_ball = cannon_coord.copy()  # Cannonball must start his motion inside the cannon
    v0_ball, angle = canon_settings(mean_ball_velocity=150,
                                    velocity_spread=15,
                                    angle=35,
                                    angle_spread=15)

    ball = CannonBall(x0=x0_ball, v0=v0_ball, radius=radius, dt=dt)  # Define cannonball object

    while True:
        image = 255 - np.zeros((plane_size[1], plane_size[0], 3), np.uint8)  # Define the plane
        image = draw_borders(image)

        # Handle collisions with deflecting plate
        if any(map(lambda x: ball.is_collision(x), plate.get_borders())):
            ball.velocity[0] *= -1

        ball.step(acceleration)
        ball_pos = ball.get_position()

        # Plate can see the cannonball only in the middle of a plane
        if (ball_pos[0] > plane_size[0] // 3) and (ball_pos[0] < 2 * plane_size[0] // 3):
            plate.append_pos(ball_pos)
            plate.approximation()
        # But plate can move whenever it need to
        plate.move_to_destination()

        # Drawing part
        image = draw_cannon(image, cannon_coord, cannon_length, cannon_width, angle)
        image = draw_cannonball(image, ball_pos, ball.radius)
        image = draw_plate(image, [plate.plate_x, plate.center_height], plate.width)

        # Show a frame
        cv2.imshow('Experiment', image)

        # Set some frequency
        k = cv2.waitKey(10)
        if k == 27:
            # If the user presses Esc, stop the experiment step
            break

        # If ball smash into the floor or into the ceil of a plane, experiment step is over
        # Experiment is also over if ball flies over the side boundaries
        if (ball_pos[0] > plane_size[0] + 100) or (ball_pos[0]) < -100 or (ball_pos[1] < 1) \
                or ball.position[1] >= plane_size[1]:
            break
    # For the better view, it would be better to make pauses between the experiment's steps
    k = cv2.waitKey(1000)
    if k == 27:
        # If the user presses Esc, stop the experiment step
        break
