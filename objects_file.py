import numpy as np


class CannonBall:
    def __init__(self, x0, v0, radius, dt):
        '''
        Object that represents a cannonball in the experiment
        :param x0: initial position - two-dimensional numpy array
        :param v0: initial velocity - two-dimensional numpy array
        :param radius: radius of a ball
        :param dt: time discretization value
        '''
        self.velocity = v0
        self.position = x0
        self.radius = radius
        self.dt = dt

    def acceleration(self, a):
        # Acceleration's action. a is acceleration vector
        self.velocity += a*self.dt

    def move(self):
        # Move ball with current velocity
        self.position += self.velocity*self.dt

    def step(self, a):
        # Accelerate and move - one step in the modelling
        self.acceleration(a)
        self.move()

    def is_collision(self, point):
        # Collision handling
        return np.sqrt(np.sum((self.position - point)**2)) <= self.radius + np.sqrt(np.sum(self.velocity**2))*self.dt/2

    def get_position(self):
        # Returns position of a ball
        return self.position.copy()


def canon_settings(mean_ball_velocity, velocity_spread, angle, angle_spread):
    '''
    This function is required to organize the process of a random initial speed and angle initialization.
    Velocity is random, but it is distrubuted normally. Angle fluctuations are choosed from an uniform distribution.
    :param mean_ball_velocity: mean of a normal distribution for a speed
    :param velocity_spread: std of a normal distribution for a speed
    :param angle: mean value of an angle.
    :param angle_spread: how wide must be uniform distribution for angle fluctuation (width is twice bigger)
    :return: initial speed for a ball and angle
    '''
    ball_velocity = np.random.randn() * velocity_spread + mean_ball_velocity
    angle = angle + np.random.uniform(-angle_spread, angle_spread)
    angle = angle * np.pi / 180
    v0_ball = np.array([ball_velocity * np.cos(angle), ball_velocity * np.sin(angle)])

    return v0_ball, angle


class CatchPlate:
    def __init__(self, center_height, plate_x, width, plate_velocity, min_height, max_height, dt):
        '''
        Object that represents plate that must deflect canonballs
        :param center_height: initial height of a center of a plate
        :param plate_x: x coordination of a line on which plate is displaced
        :param width: Width of a plate. Plate is width//2 up and width//2 down from the center
        :param plate_velocity: plate isn't teleporting so it has some velocity
        :param min_height: Down border for a plate motion
        :param max_height: Up border for a plate motion
        :param dt:
        '''
        self.center_height = center_height
        self.plate_x = plate_x
        self.width = width
        self.track_list = list()  # List for a tracking of a ball in the middle of a plane
        self.num_obs = 0  # Number of observations
        self.plate_velocity = plate_velocity
        self.required_position = center_height
        self.height_borders = [min_height, max_height]  # Organize borders in one
        self.dt = dt

    def append_pos(self, ball_pos):
        if [i[0] for i in self.track_list].count(ball_pos[0]) == 0:
            # To avoid some problems with non-parametric trajectory drawing, we exclude positions
            # with equal x coordinates from tracking list
            self.track_list.append(ball_pos)
            self.num_obs += 1

    def free_track(self):
        # When tracking of one ball is finished, it would be better to clean memory.
        self.track_list = list()
        self.num_obs = 0

    def move_to_destination(self):
        # Plate must to move as fast as it can, but if it's steps are bigger that required,
        # it would be better to set step to required size
        change = min(self.plate_velocity*self.dt, abs(self.required_position - self.center_height))

        # Two conditions 'if' that are located below guarantee that plate will be in borders.
        if ((self.center_height - self.width//2) >= self.height_borders[0]) and \
                (self.required_position < self.center_height):
            self.center_height -= change

        if ((self.center_height + self.width // 2) <= self.height_borders[1]) and \
                (self.required_position > self.center_height):
            self.center_height += change

    def approximation(self):
        if self.num_obs == 0:
            # If there aren't any observations, it would be better to not move
            self.required_position = self.center_height
        elif self.num_obs == 1:
            # If there are only one observation, it would be better to extrapolate trajectory as a constant
            self.required_position = self.track_list[0][1]
        elif self.num_obs == 2:
            # If there are two observations, we can use more accurate linear extrapolation
            x0, y0 = self.track_list[-2]
            x1, y1 = self.track_list[-1]
            if x0 == x1:
                self.required_position = y1
            else:
                a = (y1-y0)/(x1-x0)
                b = (y0*x1-y1*x0)/(x1-x0)
                self.required_position = a*self.plate_x + b   # linear approximation
        else:
            # If there are three or more observations, we can use quadratic extrapolation
            x0, y0 = self.track_list[-3]
            x1, y1 = self.track_list[-2]
            x2, y2 = self.track_list[-1]

            a = ((y2-y0)/(x2-x0) - (y1-y0)/(x1-x0))/(x2-x1)
            b = (y1-y0)/(x1-x0) - a*(x1+x0)
            c = y0 - a*(x0**2) - b*x0

            self.required_position = a*(self.plate_x**2) + b*self.plate_x + c   # quadratic approximation

    def get_borders(self):
        # Returns borders and center coordinates. It is required to handle collisions
        return [[self.plate_x, self.center_height+i] for i in [-self.width//2, 0, self.width//2]]