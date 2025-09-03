
from typing import List, Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def export_timeline_gif(timeline: List[Optional[str]], save_path: str, fps: int = 6):
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.set_axis_off()
    txt = ax.text(0.02, 0.5, "", va="center", ha="left" )

    def update(frame):
        txt.set_text(" → ".join([t for t in timeline[:frame+1] if t]))
        return (txt,)

    ani = FuncAnimation(fig, update, frames=len(timeline), interval=1000//fps, blit=True)
    ani.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
