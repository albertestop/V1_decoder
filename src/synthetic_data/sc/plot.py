import numpy as np
import matplotlib.pyplot as plt


def plot_random_cell_activity(trial_resp, trial, i, output_dir):
    first_volume = np.asarray(trial_resp[0])
    neuron_ids = np.asarray(first_volume[:, 0], dtype=float).astype(int)
    chosen_row = int(np.random.randint(len(neuron_ids)))
    chosen_neuron_id = int(neuron_ids[chosen_row])

    times = np.asarray(trial_resp[:, chosen_row, 1], dtype=float)
    amplitudes = np.asarray(trial_resp[:, chosen_row, 2], dtype=float)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, amplitudes, linewidth=1.8, label="processed")
    ax.set_title(
        f"Trial {trial} - Neuron {chosen_neuron_id}"
    )
    ax.set_xlabel("record_time")
    ax.set_ylabel("record_amplitude")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_path = f"{output_dir}/trial_{i}_single_neuron.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_trial_raster(trial_resp, trial, output_dir):
    trial_resp

    first_volume = np.asarray(trial_resp[0])
    neuron_ids = np.asarray(first_volume[:, 0], dtype=float).astype(int)
    amp_matrix = np.asarray(trial_resp[:, :, 2], dtype=float).T

    fig, ax = plt.subplots(figsize=(10, 8))
    img = ax.imshow(
        amp_matrix,
        aspect="auto",
        vmin=-0.5,
        vmax=np.max(amp_matrix)/5,
        origin="lower",
        cmap="viridis"
    )
    cbar = fig.colorbar(img, ax=ax)
    cbar.set_label("record_amplitude")

    ax.set_title(f"Raster - Trial {trial}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Neuron index")

    n_ticks = 8
    tick_pos = np.linspace(0, len(neuron_ids) - 1, n_ticks, dtype=int)
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(neuron_ids[tick_pos])

    fig.tight_layout()
    out_path = f"{output_dir}/raster_postpr.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
