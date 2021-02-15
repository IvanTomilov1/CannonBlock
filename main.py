import numpy as np
import cv2
from objects import CannonBall, CatchPlate, canon_settings
from draw import draw_borders, draw_cannonball, draw_plate, draw_cannon


def main():
    #######################################################################################################
    plane_size = (1500, 800)
    radius = 20
    plate_velocity = 125
    dt = 0.3
    acceleration = np.array([0, -9.81])  # Acceleration in gravitational field on the Earth near the ground
    ########################################################################################################
    plate = CatchPlate(center_height=plane_size[1] // 2,
                       plate_x=plane_size[0] - plane_size[0] // 3 - 10,
                       width=2 * radius,
                       plate_velocity=plate_velocity,
                       min_height=0,
                       max_height=plane_size[1])
    #########################################################################################################
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

        ball = CannonBall(x0=x0_ball, v0=v0_ball, radius=radius)

        while True:
            image = 255. - np.zeros((plane_size[1], plane_size[0], 3), np.uint8)
            image = draw_borders(image)
            ################################################################################################
            # Ball's part
            # Handle collisions with deflecting plate
            if any(map(lambda x: ball.is_collision(x, dt), plate.get_borders(plane_size[0] // 3))):
                ball.velocity[0] *= -1

            ball.velocity += acceleration * dt
            ball.move(dt)
            ball_pos = ball.get_position()
            image = draw_cannon(image, cannon_coord, cannon_length, cannon_width, angle)
            image = draw_cannonball(image, ball_pos, ball.radius)
            #################################################################################################
            # Plate's part
            plate.find_ball(image[:, plane_size[0] // 3:2 * plane_size[0] // 3, :])
            plate.approximation()
            plate.move_to_destination(dt)
            image = draw_plate(image, [plate.plate_x + plane_size[0] // 3, plate.center_height], plate.width)
            #################################################################################################

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


if __name__ == '__main__':
    main()
