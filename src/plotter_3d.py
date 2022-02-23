# pylint: disable=C0103,E1101,R1728
"""
Module responsible for saving animation
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import animation

from mediapipe.framework.formats import landmark_pb2

WHITE_COLOR = (224, 224, 224)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
_PRESENCE_THRESHOLD = 0.5
_VISIBILITY_THRESHOLD = 0.5


class Plotter3D:
    """
       Class to plot 3d graph
    """

    def __init__(self):
        """
            Initializes Plotter_3D object.
        """
        self.connections = None

        self.imgs = None
        self.landmarks = None

        self.fig = plt.figure()
        gs = GridSpec(1, 2)

        self.ax1 = self.fig.add_subplot(gs[0, 0])
        self.ax2 = self.fig.add_subplot(gs[0, 1], projection='3d')

        self.ani = None

    def plot_landmarks(self, landmark_list: landmark_pb2.NormalizedLandmarkList,
                       connections: Optional[List[Tuple[int, int]]] = None,
                       elevation: int = 10,
                       azimuth: int = 10):
        """Plot the landmarks and the connections in matplotlib 3d.
        Args:
          landmark_list: A normalized landmark list proto message to be plotted.
          connections: A list of landmark index tuples that specifies how landmarks to
            be connected.
          elevation: The elevation from which to view the plot.
          azimuth: the azimuth angle to rotate the plot.
        Raises:
          ValueError: If any connetions contain invalid landmark index.
        """
        if not landmark_list:
            return
        self.ax2.view_init(elev=elevation, azim=azimuth)
        plotted_landmarks = {}
        for idx, landmark in enumerate(landmark_list.landmark):
            if ((landmark.HasField('visibility') and
                 landmark.visibility < _VISIBILITY_THRESHOLD) or
                    (landmark.HasField('presence') and
                     landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            self.ax2.plot(
                xs=[-landmark.z],
                ys=[landmark.x],
                zs=[-landmark.y],
                linestyle="", marker="o")
            plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
        if connections:
            num_landmarks = len(landmark_list.landmark)
            # Draws the connections if the start and end landmarks are both visible.
            for connection in connections:
                start_idx = connection[0]
                end_idx = connection[1]
                if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                    raise ValueError(f'Landmark index is out of range. Invalid connection '
                                     f'from landmark #{start_idx} to landmark #{end_idx}.')
                if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                    landmark_pair = [
                        plotted_landmarks[start_idx], plotted_landmarks[end_idx]
                    ]
                    self.ax2.plot3D(
                        xs=[landmark_pair[0][0], landmark_pair[1][0]],
                        ys=[landmark_pair[0][1], landmark_pair[1][1]],
                        zs=[landmark_pair[0][2], landmark_pair[1][2]],
                        color=BLACK_COLOR,
                        linewidth=1)

    def init(self):
        """
        Initialize a plot.
        """
        self.ax2.set_xlabel('X-axis', fontweight='bold')
        self.ax2.set_ylabel('Y-axis', fontweight='bold')
        self.ax2.set_zlabel('Z-axis', fontweight='bold')
        self.ax2.set_xlim([-1, 1])
        self.ax2.set_ylim([-1, 1])
        self.ax2.set_zlim([-1, 1])

    def create_animation(self, result_landmarks, result_images):
        """
        Save 3d animation
        """
        print("Processing animation ....")
        self.init()
        self.landmarks = result_landmarks
        self.imgs = result_images
        self.ani = animation.FuncAnimation(self.fig, self.animate,
                                           interval=len(result_landmarks) / 30,
                                           frames=len(result_landmarks), repeat=False)
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        self.ani.save('./animation.mp4', writer)
        print("Animation saved")

    def animate(self, i):
        """
            Animation loop.
        """
        self.ax2.clear()
        self.ax1.clear()
        self.init()
        self.plot_landmarks(self.landmarks[i], self.connections)
        self.ax1.imshow(self.imgs[i], interpolation='nearest')
        self.ax2.set_title("Iteration: " + str(i))

    def set_connections(self, CONNECTIONS):
        """
            Set relations between joints.
        """
        self.connections = CONNECTIONS
