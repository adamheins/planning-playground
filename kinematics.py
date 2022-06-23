"""Example of computing joint locations in space using rigid body transformations."""
import numpy as np
import matplotlib.pyplot as plt


# link lengths
L1 = 0.75
L2 = 0.5

# joint angles
θ1 = 0.25 * np.pi
θ2 = 0.25 * np.pi


def transformation_matrix(θ, x, y):
    """2D tranformation matrix consisting of a rotation by angle θ and translation by (x, y)."""
    c = np.cos(θ)
    s = np.sin(θ)
    return np.array([[c, -s, x], [s, c, y], [0, 0, 1]])


def invert_transformation_matrix(T):
    """Inverse of a 2D transformation matrix.

    This is equivalent to just calling np.linalg.inv(T), but is potentially faster.
    """
    C = T[:2, :2]
    r = T[:2, 2]

    T_inv = np.eye(3)
    T_inv[:2, :2] = C.T
    T_inv[:2, 2] = -C.T @ r
    return T_inv


def main():
    # Transformation between fixed world frame F_0 and the frame F_1 located at
    # joint 1 and pointing along link 1. This maps points expressed in F_1 to
    # the equivalent point in F_0.
    T_01 = transformation_matrix(θ1, 0, 0)

    # Transformation between joint frame F_1 and F_2
    T_12 = transformation_matrix(θ2, L1, 0)

    # Transformation between joint frame F_2 and F_3
    T_23 = transformation_matrix(0, L2, 0)

    # Compound transforms to get the location of each point in the world frame
    T_02 = T_01 @ T_12
    T_03 = T_02 @ T_23

    # We can easily invert transforms, such that T @ inv(T) == np.eye(3)
    T_30 = transformation_matrix(T_03)

    # Homogeneous points in 2D have the form (x, y, 1)
    r0 = np.array([0, 0, 1])  # origin

    # Location of each joint, expressed in the world frame
    # Here we are transforming the origin of each frame to its location in the
    # world frame F_0
    r1 = T_01 @ r0
    r2 = T_02 @ r0
    r3 = T_03 @ r0

    # Plot the points
    plt.figure()
    plt.plot([r1[0], r2[0], r3[0]], [r1[1], r2[1], r3[1]], "-o")
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
