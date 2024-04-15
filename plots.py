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
    file_name = f"{target}_{indicator}det_{s_site}_{cal}_{em_intype}.pdf"
    folder_path = 'output_images/'

    plt.savefig(folder_path + file_name)

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
    file_name = f"{target}_{indicator}stoch_{s_site}_{cal}_{em_intype}.pdf"
    folder_path = 'output_images/'
    plt.savefig(folder_path + file_name)
    # Show the plot
    plt.show()

    return file_name


def plot_stoch_implementation(df, Y, Ypred, r2, profile_prefix):
    fig, axes = plt.subplots(figsize=[7, 7])
    ss = 200

    label_for_depth_50_added = False
    label_for_depth_10_added = False

    axes.set_xlim(0, 60)
    axes.set_ylim(0, 60)

    # Plot a line and label for R2
    axes.plot([0, 60], [0, 60], color='black', label=f'R2 = {r2}')
    
    axes.set_ylabel(r'Predicted observed $\theta$ [%]', fontsize=20)
    axes.set_xlabel(r'Observed $\theta$ [%]', fontsize=20)

    for i, (x, y) in enumerate(zip(Y, Ypred)):
        if df['depth'].iloc[i] == 50:
            marker_style = 'D'
            c = 'orange'
            if not label_for_depth_50_added:
                axes.scatter(x*100, y*100, s=ss, alpha=0.8, color=c, marker=marker_style, label='Layer 50 cm')
                label_for_depth_50_added = True
            else:
                axes.scatter(x*100, y*100, s=ss, alpha=0.8, color=c, marker=marker_style)
        else:
            marker_style = 'o'
            c = 'blue'
            if not label_for_depth_10_added:
                axes.scatter(x*100, y*100, s=ss, alpha=0.8, color=c, marker=marker_style, label='Layer 10 cm')
                label_for_depth_10_added = True
            else:
                axes.scatter(x*100, y*100, s=ss, alpha=0.8, color=c, marker=marker_style)

    # Add the legend after all plotting is done
    axes.legend(fontsize=14)

    plt.suptitle(f'{profile_prefix}', fontsize=24)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    axes.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    # Save the figure with a filename that includes profile_prefix
    file_name_pdf = f"Stoch_VWC_{profile_prefix}.pdf"
    file_name_png = f"Stoch_VWC_{profile_prefix}.png"

    folder_path = 'output_images/'
    plt.savefig(folder_path + file_name_pdf)
    plt.savefig(folder_path + file_name_png)

    # Show the plot
    plt.show()


def plot_bars(uncal_LIN_M, cal_LIN_M, cal_rECa_M, uncal_LIN_P, cal_LIN_P, cal_rECa_P, target_set, approaches):
    # Setup the subplots
    fig, axes = plt.subplots(2, len(approaches), figsize=(15, 10), sharey=True, sharex=True)

    for idx, approach in enumerate(approaches):
        bars_data = {
            'Uncalibrated LIN': {'P': uncal_LIN_P, 'M': uncal_LIN_M},
            'Calibrated LIN': {'P': cal_LIN_P, 'M': cal_LIN_M},
            'Calibrated rECa': {'P': cal_rECa_P, 'M': cal_rECa_M}
        }

        lab = np.arange(len(target_set))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0

        colors = {'Uncalibrated LIN': 'grey', 'Calibrated LIN': 'orange', 'Calibrated rECa': 'blue'}

        for i, soil_type in enumerate(['P', 'M']):
            ax = axes[i, idx]

            for attribute, data in bars_data.items():
                offset = width * multiplier

                if i == 0:
                    rects = ax.bar(lab + offset, data[soil_type][approach][:5], width, label=attribute, color=colors[attribute])
                else: 
                    rects = ax.bar(lab + offset, data[soil_type][approach][:5], width, color=colors[attribute])
                    
                # Adjust bar labels for readability and apply inclination
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                rotation=60)  # Rotate the annotations for readability
                multiplier += 1

            # Add some text for labels, title and custom x-axis tick labels, etc.
            if idx == 0:
                ax.set_ylabel('${R^2}$', fontweight='bold', fontsize=14)
                ax.legend(loc='upper left', ncol=1)

            ax.set_xticks(lab + width / 2)
            ax.set_xticklabels(target_set, rotation=60, fontsize=14)
            ax.set_ylim(-0.25, 0.8)  # Adjust as per your data's range

            multiplier = 0

    # Adding labels to the plot
    fig.text(0.99, 0.9, 'Proefhoeve', horizontalalignment='left', verticalalignment='center', fontweight='bold', fontsize=14, rotation=-90)
    fig.text(0.99, 0.45, 'Middlekerke', horizontalalignment='left', verticalalignment='center', fontweight='bold', fontsize=14, rotation=-90)

    fig.text(0.20, 0.95, 'LS',  fontweight='bold', fontsize=14)
    fig.text(0.50, 0.95, 'LT',  fontweight='bold', fontsize=14)
    fig.text(0.85, 0.95, 'LT2', fontweight='bold', fontsize=14)

    plt.tight_layout()

    folder_path = 'output_images/'
    file_name_png = f"S_Results_bars.png"
    file_name_pdf = f"S_Results_bars.pdf"

    plt.savefig(folder_path + file_name_pdf, bbox_inches='tight')  # Adjusted bbox_inches
    plt.savefig(folder_path + file_name_png, bbox_inches='tight')  # Adjusted bbox_inches

    plt.show()