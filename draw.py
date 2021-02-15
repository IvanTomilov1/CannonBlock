import numpy as np
import cv2


def draw_borders(image):
    '''
    Draw borders of a plane and draw two grey lines that divides a plane to three equal parts
    :param image: plane
    :return: plane with borders and grey lines
    '''
    room_shape = (image.shape[1], image.shape[0])
    cv2.line(image, (room_shape[0] // 3, 0), (room_shape[0] // 3, room_shape[1]), (128, 128, 128), 3)
    cv2.line(image, (2 * room_shape[0] // 3, 0), (2 * room_shape[0] // 3, room_shape[1]), (128, 128, 128), 3)

    cv2.line(image, (0, 0), (room_shape[0], 0), (0, 0, 0), 5)
    cv2.line(image, (0, room_shape[1] - 1), (room_shape[0], room_shape[1] - 1), (0, 0, 0), 5)
    cv2.line(image, (0, 0), (0, room_shape[1]), (0, 0, 0), 5)
    cv2.line(image, (room_shape[0], 0), (room_shape[0], room_shape[1]), (0, 0, 0), 5)

    cv2.putText(image, "Press Esc twice to exit", (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
    image = image.astype(np.uint8)
    return image


def draw_cannon(image, cannon_coord, cannon_length, cannon_width, angle):
    '''
    :param image: plane
    :param cannon_coord: coordinates of an axis of rotation of a cannon
    :param cannon_length: length of a cannon
    :param cannon_width: width of a cannon
    :param angle: angle on which cannon is rotate relatively to horizon
    :return: plane with a cannon on it
    '''
    plane_size = [image.shape[1], image.shape[0]]
    canon_x2 = int(cannon_coord[0] + cannon_length * np.cos(angle))
    canon_y2 = int(cannon_coord[1] + cannon_length * np.sin(angle))

    cv2.line(image,
             (int(cannon_coord[0]), plane_size[1] - int(cannon_coord[1])),
             (canon_x2, plane_size[1] - canon_y2),
             (100, 100, 100), cannon_width)
    cv2.circle(image, (int(cannon_coord[0]), plane_size[1] - int(cannon_coord[1])), int(cannon_width), (0, 0, 0), -1)
    image = image.astype(np.uint8)
    return image


def draw_cannonball(image, ball_position, radius):
    '''
    :param image: the plane
    :param ball_position: two-dimensional array that represents position of a cannonball
    :param radius: radius of a cannonball
    :return: plane with a cannonball on it
    '''
    plane_size = [image.shape[1], image.shape[0]]
    cv2.circle(image, (int(ball_position[0]), plane_size[1] - int(ball_position[1])),
               radius, (0, 0, 255), -1)
    image = image.astype(np.uint8)
    return image


def draw_plate(image, plate_pos, plate_width):
    '''
    :param image: the plane
    :param plate_pos: plate position in format [x_coord, height]
    :param plate_width: width of a plate
    :return: plane with a plate on it
    '''
    plane_size = [image.shape[1], image.shape[0]]
    cv2.line(image, (int(plate_pos[0]), plane_size[1] - int(plate_pos[1] + plate_width)),
             (int(plate_pos[0]), plane_size[1] - int(plate_pos[1] - plate_width)), (0, 0, 0), 3)

    cv2.line(image, (int(plate_pos[0]), plane_size[1] - int(plate_pos[1])),
             (plane_size[0], plane_size[1] - int(plate_pos[1])), (100, 100, 100), 6)
    image = image.astype(np.uint8)
    return image
