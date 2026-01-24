import os
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

rtm_connections = {
    'hand': [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ],
    'body': [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [3, 5],
        [5, 7],
        [4, 6],
        [6, 8],
    ],
    'face': [
        # No connections for face in RTM format
    ]
}

mp_connections = {
    'hand': [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 4],
        [0, 5],
        [5, 6],
        [6, 7],
        [7, 8],
        [0, 9],
        [9, 10],
        [10, 11],
        [11, 12],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 17],
        [17, 18],
        [18, 19],
        [19, 20],
    ],
    'body': [
        [0, 1],
        [0, 2],
        [0, 3],
        [0, 4],
        [3, 5],
        [5, 7],
        [4, 6],
        [6, 8],
    ],
    'face': [
        # TODO: Define face connections for MediaPipe if needed
    ]
}

def viz_skeletons(skeletons, save_path: Path = None, title="Skeleton Visualization", outfile=None, pose_type='rtm'):
    """
    Visualize skeletons using matplotlib.

    Args:
        skeletons (list of np.array): List of skeleton keypoints.
        save_path (str, optional): Path to save the visualization. If None, shows the plot.
        title (str): Title of the plot.
    """
    body = skeletons['body']
    left = skeletons['left']
    right = skeletons['right']
    face = skeletons.get('face', None)
    
    BODY_CONNECTIONS = rtm_connections['body'] if pose_type == 'rtm' else mp_connections['body']
    HAND_CONNECTIONS = rtm_connections['hand'] if pose_type == 'rtm' else mp_connections['hand']
    FACE_CONNECTIONS = rtm_connections['face'] if pose_type == 'rtm' else mp_connections['face']

    os.makedirs(str(save_path), exist_ok=True) 
    for t in range(body.shape[0]):
        
        _body  = body[t]   
        _left  = left[t] 
        _right = right[t]
        _face = face[t] if face is not None else None
        
        def plot_skeleton(ax, keypoints, connections, color='b'):
            for connection in connections:
                p1 = keypoints[connection[0]]
                p2 = keypoints[connection[1]]
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, marker='o')

        fig, ax = plt.subplots(1, 3, figsize=(15, 5)) if _face is None else plt.subplots(1, 4, figsize=(20, 5))
        ax[0].set_title('Body')
        plot_skeleton(ax[0], _body, BODY_CONNECTIONS, color='b')
        ax[1].set_title('Left Hand')
        plot_skeleton(ax[1], _left, HAND_CONNECTIONS, color='g')
        ax[2].set_title('Right Hand')
        plot_skeleton(ax[2], _right, HAND_CONNECTIONS, color='r')
        
        if _face is not None:
            ax[3].set_title('Face')
            ax_face = ax[3]
            plot_skeleton(ax_face, _face, FACE_CONNECTIONS, color='m')
        
        plt.suptitle(title)
        

        # invert y-axis
        for a in ax:
            a.invert_yaxis()

        if save_path:
            plt.savefig(save_path / f"skeleton_{t:04d}.png")
            plt.close()

    images = []
    for t in range(body.shape[0]):
        img_path = save_path / f"skeleton_{t:04d}.png"
        images.append(imageio.imread(img_path))

    imageio.mimsave(save_path / outfile, images, fps=25, loop=0)

