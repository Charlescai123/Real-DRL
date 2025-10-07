import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Reward plot
realdrl_model_logdir = "./logs/src-assurance/2025_05_08_11_50_44"
phydrl_model_logdir = "./logs/phy-drl/2025_05_06_15_14_31"
rtassurance_model_logdir = "./logs/src-assurance/2025_05_07_22_47_04"
drl_model_logdir = "./logs/drl/2025_05_05_15_43_34"
neuralspx_model_logdir = "./logs/neural-simplex/2025_05_05_01_31_27"


#
# Falls plot
# rtlf_model_logdir = "./logs/train_ddpg_uneven_teacher_gamma_0.1/2025_02_25_12_18_17"
# phydrl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.4/2025_02_27_13_13_32"
# drl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_1/2025_02_26_17_58_41"

def exponential_smoothing(data, smoothing=0.99):
    """TensorBoard-style exponential smoothing"""
    smoothed = []
    last = data[0]
    for point in data:
        last = smoothing * last + (1 - smoothing) * point
        smoothed.append(last)
    return np.array(smoothed)


def load_event_file(log_dir):
    """Load the event file in the log dir"""
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f]
    event_file = sorted(event_files, key=os.path.getmtime)[-1]

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    return event_acc


def plot_for_event(event_acc, tag, color, label, plot_distribution=True, use_ema=False, smoothing=0.999):
    if tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        # Use exponential moving average
        if use_ema:

            # Compute exponential moving average
            smoothed_values = exponential_smoothing(values, smoothing=smoothing)

            # Compute EMA-based standard deviation (optional & approximate)
            if plot_distribution:
                rewards_std = np.std(values) * np.ones_like(smoothed_values)  # constant std shading
                plt.plot(steps, smoothed_values, label=label, color=color, linewidth=5)
                plt.fill_between(steps,
                                 smoothed_values - rewards_std,
                                 smoothed_values + rewards_std,
                                 color=color, alpha=0.1)
            else:
                plt.plot(steps, smoothed_values, label=label, color=color, linewidth=5)

        else:
            # plot with distribution
            if plot_distribution:
                # Compute the moving average and standard deviation.
                window_size = 100  # window size
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
    realdrl_event = load_event_file(realdrl_model_logdir)
    neuralspx_event = load_event_file(neuralspx_model_logdir)
    rtassurance_event = load_event_file(rtassurance_model_logdir)
    phydrl_event = load_event_file(phydrl_model_logdir)
    drl_event = load_event_file(drl_model_logdir)

    # tag = "Train/mean_reward"

    # Figure plot
    fig = plt.figure(figsize=(19, 15))
    plot_for_event(realdrl_event, tag, color='forestgreen', label="Real-DRL", plot_distribution=plot_distribution)
    plot_for_event(neuralspx_event, tag, color='royalblue', label="Neural Simplex", plot_distribution=plot_distribution)
    plot_for_event(rtassurance_event, tag, color='crimson', label="Runtime Assurance",
                   plot_distribution=plot_distribution)
    plot_for_event(phydrl_event, tag, color='goldenrod', label="Phy-DRL", plot_distribution=plot_distribution)
    plot_for_event(drl_event, tag, color='#9467bd', label="CLF-DRL", plot_distribution=plot_distribution)

    plt.xlabel("Episode Numbers", fontsize=35)
    plt.ylabel("Episode Return", fontsize=35)
    # plt.title("Training Reward with Variance")
    plt.ylim(-9000, 15000)
    plt.xlim(1, 5000)
    # plt.legend(ncol=2, loc='lower center')

    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)

    plt.legend(ncol=2, loc="lower right", fontsize=35)
    plt.grid()
    plt.show()
    fig_name = tag.split('/')[-1] if '/' in tag else tag
    fig.savefig(f'{fig_name}.pdf', dpi=300)


if __name__ == '__main__':
    # tag = "Train/avg_episode_return"
    tag = "Train/mean_reward"
    summary_fig_plot(tag=tag)  # For reward plot
    # summary_fig_plot(tag="Perf/failed_times", plot_distribution=False)      # For failed times plot
