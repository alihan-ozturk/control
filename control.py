import numpy as np
from scipy.linalg import expm
import cvxpy as cp
import pygame
import math

# Initialize Pygame
pygame.init()

# Screen Dimensions
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MPC vs User-Controlled Vehicle")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Fonts
pygame.font.init()
font = pygame.font.SysFont('Arial', 16)

# Parameters
dt = 0.1
num_steps = 1000
horizon = 10
friction_coefficient = 1

Q = np.diag([0.1, 0.1, 1, 0.001, 0.001])
R = np.diag([0.0001, 0.01])

x_mpc = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
x_user = np.array([0.0, 0.0, 0.0, 1.0, 0.0])
x_ref = np.array([np.random.uniform(-20, 20), np.random.uniform(-20, 10),
                  np.random.uniform(-2*np.pi, 2*np.pi), np.random.uniform(-1, 1), np.random.uniform(-1, 1)])

x_mpc_history = [x_mpc[0]]
y_mpc_history = [x_mpc[1]]
x_user_history = [x_user[0]]
y_user_history = [x_user[1]]
state_mpc_history = [x_mpc.copy()]
state_user_history = [x_user.copy()]

u_user = np.array([0.0, 0.0])
u_guess = np.array([0.0, 0.0])

# Functions
def state_space_model(x, u, friction_coefficient):
    X, Y, theta, vx, vy = x
    u1, u2 = u
    dx = np.zeros_like(x)
    dx[0] = vx
    dx[1] = vy
    dx[2] = u2
    dx[3] = u1 * np.cos(theta) - friction_coefficient * vx
    dx[4] = u1 * np.sin(theta) - friction_coefficient * vy
    return dx

def get_linear_matrices(theta, vx, vy, u1, friction_coefficient):
    A = np.array([
        [0,    0,       0,                    1,                   0],
        [0,    0,       0,                    0,                   1],
        [0,    0,       0,                    0,                   0],
        [0,    0, -u1*np.sin(theta), -friction_coefficient,        0],
        [0,    0,  u1*np.cos(theta),          0,        -friction_coefficient]
    ])
    B = np.array([
        [0,           0],
        [0,           0],
        [0,           1],
        [np.cos(theta),0],
        [np.sin(theta),0]
    ])
    return A, B

def draw_grid():
    for x in range(0, WIDTH, 50):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, 50):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

def to_screen_coords(x, y, offset):
    return int(WIDTH / 2 + offset + x * 10), int(HEIGHT / 2 - y * 10)

def draw_text(surface, text, pos, color=BLACK):
    label = font.render(text, True, color)
    surface.blit(label, pos)

# Pygame Main Loop
running = True
clock = pygame.time.Clock()
step = 0

while running and step < num_steps:
    # Handle Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # User Input Control
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        u_user[0] += 0.1  # Increment forward
    if keys[pygame.K_s]:
        u_user[0] -= 0.1  # Increment backward
    if keys[pygame.K_a]:
        u_user[1] += 0.05  # Increment left turn
    if keys[pygame.K_d]:
        u_user[1] -= 0.05  # Increment right turn

    # Clip user inputs
    u_user = np.clip(u_user, [-10, -np.pi], [10, np.pi])

    # MPC Control
    A, B = get_linear_matrices(x_mpc[2], x_mpc[3], x_mpc[4], u_guess[0], friction_coefficient)
    A_d = expm(A * dt)
    B_d = B * dt

    x_var = cp.Variable((5, horizon + 1))
    u_var = cp.Variable((2, horizon))

    cost = 0
    constraints = [x_var[:, 0] == x_mpc]
    for t in range(horizon):
        cost += cp.quad_form(x_var[:, t] - x_ref, Q) + cp.quad_form(u_var[:, t], R)
        constraints += [x_var[:, t + 1] == A_d @ x_var[:, t] + B_d @ u_var[:, t]]
        constraints += [cp.abs(u_var[0, t]) <= 10]
        constraints += [cp.abs(u_var[1, t]) <= np.pi]

    cost += cp.quad_form(x_var[:, horizon] - x_ref, Q)
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve()

    u_mpc = u_var[:, 0].value
    if u_mpc is None:
        u_mpc = np.array([0.0, 0.0])

    # Update States
    x_mpc = x_mpc + dt * state_space_model(x_mpc, u_mpc, friction_coefficient)
    x_user = x_user + dt * state_space_model(x_user, u_user, friction_coefficient)
    x_ref = x_user
    x_mpc_history.append(x_mpc[0])
    y_mpc_history.append(x_mpc[1])
    x_user_history.append(x_user[0])
    y_user_history.append(x_user[1])

    state_mpc_history.append(x_mpc.copy())
    state_user_history.append(x_user.copy())

    u_guess = u_mpc

    # Draw Scene
    screen.fill(WHITE)

    draw_grid()

    # Draw Reference Point
    # ref_x, ref_y = to_screen_coords(x_ref[0], x_ref[1], 0)
    # pygame.draw.circle(screen, GREEN, (ref_x, ref_y), 5)

    # Draw MPC Vehicle Path
    for i in range(max(0, len(state_mpc_history) - 20), len(state_mpc_history)):
        x, y = to_screen_coords(state_mpc_history[i][0], state_mpc_history[i][1], 0)
        pygame.draw.circle(screen, BLUE, (x, y), 2)

    # Draw User-Controlled Vehicle Path
    for i in range(max(0, len(state_user_history) - 20), len(state_user_history)):
        x, y = to_screen_coords(state_user_history[i][0], state_user_history[i][1], 0)
        pygame.draw.circle(screen, RED, (x, y), 2)

    # Draw MPC Vehicle Heading
    mpc_x, mpc_y = to_screen_coords(x_mpc[0], x_mpc[1], 0)
    pygame.draw.circle(screen, BLUE, (mpc_x, mpc_y), 5)
    mpc_heading_x = mpc_x + 15 * math.cos(x_mpc[2])
    mpc_heading_y = mpc_y - 15 * math.sin(x_mpc[2])
    pygame.draw.line(screen, BLUE, (mpc_x, mpc_y), (mpc_heading_x, mpc_heading_y), 2)

    # Draw User-Controlled Vehicle Heading
    user_x, user_y = to_screen_coords(x_user[0], x_user[1], 0)
    pygame.draw.circle(screen, RED, (user_x, user_y), 5)
    user_heading_x = user_x + 15 * math.cos(x_user[2])
    user_heading_y = user_y - 15 * math.sin(x_user[2])
    pygame.draw.line(screen, RED, (user_x, user_y), (user_heading_x, user_heading_y), 2)

    # Display States
    draw_text(screen, f"MPC State:", (10, 10))
    draw_text(screen, f"X: {x_mpc[0]:.2f}, Y: {x_mpc[1]:.2f}", (10, 30))
    draw_text(screen, f"Theta: {x_mpc[2]:.2f}, VX: {x_mpc[3]:.2f}, VY: {x_mpc[4]:.2f}", (10, 50))

    draw_text(screen, f"User State:", (WIDTH - 200, 10))
    draw_text(screen, f"X: {x_user[0]:.2f}, Y: {x_user[1]:.2f}", (WIDTH - 200, 30))
    draw_text(screen, f"Theta: {x_user[2]:.2f}, VX: {x_user[3]:.2f}, VY: {x_user[4]:.2f}", (WIDTH - 200, 50))

    pygame.display.flip()
    clock.tick(10)
    step += 1

pygame.quit()