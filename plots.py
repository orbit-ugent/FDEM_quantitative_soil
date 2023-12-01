import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def bars_plot(feature_sets, test_errors_summary, train_errors_summary, title):
    fig, ax = plt.subplots(figsize=[7, 4])
    width = 0.35  # the width of the bars

    x = np.arange(len(feature_sets))
    rects1 = ax.bar(x - width/2, test_errors_summary, width, color = 'red', label='Test')
    rects2 = ax.bar(x + width/2, train_errors_summary, width, color = 'blue', label='Train')

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_ylabel('Scores '+title)
    ax.set_xticks(range(len(test_errors_summary)), feature_sets, rotation = 90)
    ax.set_ylim(-0.5, 1)
    ax.legend()
    ax.set_title('features vs '+title)
    fig.tight_layout()

    plt.show()
    #plt.savefig("stochastic_"+title, dpi=200)


def plot_results(df, actual, predicted, r2_val, rmse_val, scale, title):
    
    fig, axes = plt.subplots(figsize=[7, 4])
    ss = 100
    
    # Create a colormap
    cmap = plt.cm.Reds

    # Loop through each point and plot with appropriate marker
    for i, (x, y, s) in enumerate(zip(actual, predicted, scale)):
        if ~(np.isnan(x) or np.isnan(y) or np.isnan(s)):
            if df['depth'].iloc[i] == 50:
                marker_style = 'D'
            else:
                marker_style = 'o'
            axes.scatter(x, y, s=ss, alpha=0.8, c=[s], cmap=cmap, vmin=0, vmax=65, marker=marker_style)

    # Create a dummy scatter plot to use its colormap
    dummy_scatter = plt.scatter([], [], c=[], cmap=cmap, vmin=0, vmax=65)
    cbar = plt.colorbar(dummy_scatter, ax=axes)

    axes.plot([0, 0.6], [0, 0.6], color='black', label=f'R2 = {r2_val}; RMSE = {rmse_val}')
    axes.set_xlabel("Observed wat")
    axes.set_ylabel("Predicted wat")
    axes.set_title(title)
    axes.legend()
    axes.grid(True)
    axes.set_xlim([0, 0.6])
    axes.set_ylim([0, 0.6])
    plt.show()


def plot_det(results, feature_set, target, profile_prefix, em_intype, cal, s_site, indicator, color_lt, color_ls, color_id):
    # Create subplots
    fig, axes = plt.subplots(nrows=len(feature_set), ncols=1, figsize=(10, 3 * len(feature_set)), sharex=True, sharey=True)

    for idx, (feature, scores) in enumerate(results.items()):
        ax = axes[idx]
        sns.histplot(scores['LT'][indicator], ax=ax, color=color_lt, label='layers together', kde=True, stat="density", linewidth=0)
        sns.histplot(scores['ID'][indicator], ax=ax, color=color_id, label='ideal', kde=True, stat="density", linewidth=0)
        sns.histplot(scores['LS'][indicator], ax=ax, color=color_ls, label='layers separate', kde=True, stat="density", linewidth=0)

        # Highlight median and mean
        ax.axvline(x=np.median(scores['LT'][indicator]), color=color_lt, linestyle='--')
        ax.axvline(x=np.median(scores['ID'][indicator]), color=color_id, linestyle='--')
        ax.axvline(x=np.median(scores['LS'][indicator]), color=color_ls, linestyle='--')
        ax.axvline(x=np.mean(scores['LT'][indicator]), color=color_lt, linestyle=':', alpha=0.7)
        ax.axvline(x=np.mean(scores['ID'][indicator]), color=color_id, linestyle=':', alpha=0.7)
        ax.axvline(x=np.mean(scores['LS'][indicator]), color=color_ls, linestyle=':', alpha=0.7)

        # Set x-axis limits
        if indicator == 'R2':
            ax.set_xlim(-10, 1)
            ax.set_ylim(0, 0.75)
        else:
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 25)
        
        ax.set_ylabel(f'{feature} {indicator} density')

        ax.legend()

    # Set the overall figure title
    plt.suptitle(f'Deterministic {target} {indicator} distribution at {profile_prefix}, {em_intype}', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Save the figure with a filename that includes s_site and em_intype
    filename = f"{target}_{indicator}det_{s_site}_{cal}_{em_intype}.png"
    plt.savefig(filename)

    # Show the plot
    plt.show()


def plot_stoch(results, feature_set, target, profile_prefix, em_intype, cal, s_site, indicator, color_lt, color_ls, color_LS2):
    # Create subplots
    fig, axes = plt.subplots(nrows=len(feature_set), ncols=1, figsize=(10, 3 * len(feature_set)), sharex=True, sharey=True)

    for idx, (feature, scores) in enumerate(results.items()):
        ax = axes[idx]
        sns.histplot(scores['LT'][indicator], ax=ax, color=color_lt, label='LT, n ='+str(scores['LT']['n']), kde=True, stat="density", linewidth=0)
        sns.histplot(scores['LS2'][indicator], ax=ax, color=color_LS2, label='LS2, n ='+str(scores['LS2']['n']), kde=True, stat="density", linewidth=0)
        sns.histplot(scores['LS'][indicator], ax=ax, color=color_ls, label='LS, n ='+str(scores['LS']['n']), kde=True, stat="density", linewidth=0)

        # Highlight median and mean

        ax.axvline(x=np.median(scores['LT'][indicator]), color=color_lt, linestyle='--')
        ax.axvline(x=np.median(scores['LS2'][indicator]), color=color_LS2, linestyle='--')
        ax.axvline(x=np.median(scores['LS'][indicator]), color=color_ls, linestyle='--')

        # Set x-axis limits
        if indicator == 'R2':
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 5)
        else:
            ax.set_xlim(0, 0.5)
            ax.set_ylim(0, 25)
        
        ax.set_ylabel(f'{feature} {indicator} density')

        ax.legend()

    # Set the overall figure title
    plt.suptitle(f'Stochastic {target} {indicator} distribution at {profile_prefix}, {em_intype}', fontsize=16)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    # Save the figure with a filename that includes s_site and em_intype
    filename = f"{target}_{indicator}stoch_{s_site}_{cal}_{em_intype}.png"
    plt.savefig(filename)

    # Show the plot
    plt.show()
