import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorflow.python.framework import tensor_util

# Reward plot
rtlf_model_logdir = "./logs/train_ddpg_uneven_teacher_gamma_0.1/2025_02_27_03_10_21"
# phydrl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.4/2025_02_27_13_13_32"
phydrl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.45/2025_02_28_12_12_49"

# drl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.8/2025_02_27_19_07_51"
drl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_1/2025_02_28_01_31_47"


#
# Falls plot
# rtlf_model_logdir = "./logs/train_ddpg_uneven_teacher_gamma_0.1/2025_02_25_12_18_17"
# phydrl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.4/2025_02_27_13_13_32"
# drl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_1/2025_02_26_17_58_41"


def load_event_file(event_log_folder: str):
    """Find and load the first event file from the folder."""
    for filename in os.listdir(event_log_folder):
        if "events.out.tfevents" in filename:
            event_path = os.path.join(event_log_folder, filename)
            ea = event_accumulator.EventAccumulator(event_path)
            ea.Reload()
            return ea
    raise FileNotFoundError("No event file found in the specified folder.")


def event2csv(event_log_folder: str, output_dir="exported_data"):
    """Convert TensorBoard tensor-type scalar events to CSV files."""
    ea = load_event_file(event_log_folder)
    import pdb
    pdb.set_trace()
    os.makedirs(output_dir, exist_ok=True)

    scalar_tags = ea.Tags().get("tensors", [])
    if not scalar_tags:
        print("No tensor tags found.")
        return

    for tag in scalar_tags:
        events = ea.Tensors(tag)
        steps = []
        values = []
        wall_times = []

        for e in events:
            try:
                value = tensor_util.MakeNdarray(e.tensor_proto).item()
            except Exception as ex:
                print(f"Failed to parse tensor for tag '{tag}' at step {e.step}: {ex}")
                continue

            steps.append(e.step)
            values.append(value)
            wall_times.append(e.wall_time)

        df = pd.DataFrame({
            "step": steps,
            "value": values,
            "wall_time": wall_times
        })

        safe_tag = tag.replace("/", "_")
        file_path = os.path.join(output_dir, f"{safe_tag}.csv")
        df.to_csv(file_path, index=False)
        print(f"Saved tag '{tag}' to '{file_path}'")


def plot_for_event(event_acc, tag, color, label, plot_distribution=True):
    if tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        # plot with distribution
        if plot_distribution:
            # Compute the moving average and standard deviation.
            window_size = 120  # window size
            rewards_mean = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
            rewards_std = np.array([np.std(values[max(0, i - window_size):i]) for i in range(len(values))])

            plt.plot(steps[:len(rewards_mean)], rewards_mean, label=label, color=color, linewidth=5)
            plt.fill_between(steps[:len(rewards_mean)],
                             rewards_mean - rewards_std[:len(rewards_mean)],
                             rewards_mean + rewards_std[:len(rewards_mean)],
                             color=color, alpha=0.2)  # range of standard
        else:
            plt.plot(steps, values, label=label, color=color, linewidth=5)


def summary_fig_plot(tag="Train/mean_reward", plot_distribution=True):
    trlf_event = load_event_file(rtlf_model_logdir)
    phydrl_event = load_event_file(phydrl_model_logdir)
    drl_event = load_event_file(drl_model_logdir)

    # tag = "Train/mean_reward"

    # Figure plot
    fig = plt.figure(figsize=(13, 12))
    plot_for_event(trlf_event, tag, color='green', label="Ours", plot_distribution=plot_distribution)
    plot_for_event(phydrl_event, tag, color='blue', label="Phy-DRL", plot_distribution=plot_distribution)
    plot_for_event(drl_event, tag, color='red', label="CLF-DRL", plot_distribution=plot_distribution)

    plt.xlabel("Training Episode", fontsize=30)
    plt.ylabel("Times of Falls", fontsize=30)
    # plt.title("Training Reward with Variance")
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.legend(loc="lower right", fontsize=25)
    # plt.grid()
    plt.show()
    fig_name = tag.split('/')[-1] if '/' in tag else tag
    fig.savefig(f'{fig_name}.pdf', dpi=300)


if __name__ == '__main__':
    event2csv(f"results/logs/train/Real-DRL/training")
    # summary_fig_plot()      # For reward plot
    # summary_fig_plot(tag="Perf/failed_times", plot_distribution=False)      # For failed times plot
