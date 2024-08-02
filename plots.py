import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.linear_model import LinearRegression


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

    axes.plot([0, 0.6], [0, 0.6], color='black', label=r'$R^2$'+f'= {r2_val}; RMSE = {rmse_val}')
    axes.set_xlabel("$θ$* [%]")
    axes.set_ylabel("Predicted $θ$* [%]")
    axes.set_title(title)
    axes.legend()
    axes.grid(True)
    axes.set_xlim([0, 0.6])
    axes.set_ylim([0, 0.6])
    plt.show()


def plot_det(results, feature_set, target, profile_prefix, em_intype, cal, s_site, indicator, color_lt, color_ls, color_id):
    # Create subplots
    fig, axes = plt.subplots(nrows=len(feature_set), ncols=1, figsize=(10, 3 * len(feature_set)), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]  # Make it iterable even if there's only one subplot

    for idx, (feature, scores) in enumerate(results.items()):
        ax = axes[idx]
        sns.histplot(scores['LT'][indicator], ax=ax, color=color_lt, label='layers together', kde=True, stat="density", linewidth=0)
        sns.histplot(scores['ID'][indicator], ax=ax, color=color_id, label='ideal', kde=True, stat="density", linewidth=0)
        sns.histplot(scores['LS'][indicator], ax=ax, color=color_ls, label='layers separate', kde=True, stat="density", linewidth=0)

        # Highlight median and mean
        ax.axvline(x=np.median(scores['LT'][indicator]), color=color_lt, linestyle='--')
        ax.axvline(x=np.median(scores['ID'][indicator]), color=color_id, linestyle='--')
        ax.axvline(x=np.median(scores['LS'][indicator]), color=color_ls, linestyle='--')

        # Set x-axis and y-axis limits based on indicator
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
    plt.savefig(folder_path + file_name, dpi=300)

    # Show the plot
    plt.show()


def plot_stoch(results, feature_set, target, profile_prefix, em_intype, cal, s_site, indicator, color_lt, color_ls, color_LS2):
    # Create subplots
    fig, axes = plt.subplots(nrows=len(feature_set), ncols=1, figsize=(10, 3 * len(feature_set)), sharex=True, sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = [axes]  # Make it iterable even if there's only one subplot

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
    plt.savefig(folder_path + file_name, dpi=300)
    # Show the plot
    plt.show()

    return file_name


def f8(df, Y, Ypred, r2, profile_prefix):
    fig, axes = plt.subplots(figsize=[7, 7])
    ss = 200

    label_for_depth_50_added = False
    label_for_depth_10_added = False

    axes.set_xlim(0, 60)
    axes.set_ylim(0, 60)

    # Plot a line and label for R2
    axes.plot([0, 60], [0, 60], color='black', label=r'$R^2$'+f' = {r2}')
    
    if profile_prefix == 'proefhoeve':
        axes.set_ylabel('Site 2. Stochastic prediction $θ$ [%]', fontsize=18)

    elif profile_prefix == 'middelkerke':
        axes.set_ylabel('Site 1. Stochastic prediction $θ$ [%]', fontsize=18)

    axes.set_xlabel('Observed $θ$ [%]', fontsize=18)

    for i, (x, y) in enumerate(zip(Y, Ypred)):
        if df['depth'].iloc[i] == 50:
            marker_style = '^'
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

    #plt.suptitle(f'{profile_prefix}', fontsize=24)
    axes.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    axes.grid(True)

    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])

    # Save the figure with a filename that includes profile_prefix
    file_name_png = f"Stoch_VWC_{profile_prefix}.png"

    folder_path = 'output_images/'
    plt.savefig(folder_path + file_name_png, dpi=300)

    # Show the plot
    plt.show() 


def f7(uncal_LIN_M, cal_LIN_M, cal_rECa_M, inv_M, uncal_LIN_P, cal_LIN_P, cal_rECa_P, inv_P, target_set, approaches):
    # Setup the subplots
    fig, axes = plt.subplots(2, len(approaches), figsize=(15, 10), sharey=True, sharex=True)
    width = 0.2  # the width of the bars

    for idx, approach in enumerate(approaches):
        bars_data = {
            'Uncalibrated LIN': {'P': uncal_LIN_P, 'M': uncal_LIN_M},
            'Calibrated LIN': {'P': cal_LIN_P, 'M': cal_LIN_M},
            'Calibrated rECa': {'P': cal_rECa_P, 'M': cal_rECa_M},
            'FDEM Inverted': {'P': inv_P, 'M': inv_M}
        }

        lab = np.arange(len(target_set))  # the label locations
        offset_multiplier = -1.5 * width  # Start offset to center bars

        colors = {'Uncalibrated LIN': 'red', 'Calibrated LIN': 'navy', 'Calibrated rECa': 'navy', 'FDEM Inverted': 'green'}
        hatches = {'Uncalibrated LIN': '///', 'Calibrated LIN': '///', 'Calibrated rECa': '', 'FDEM Inverted': ''}  # Hatching for LIN bars

        for i, soil_type in enumerate(['M', 'P']):
            ax = axes[i, idx]

            for attribute, data in bars_data.items():
                # Add hatch to LIN bars only
                hatch = hatches.get(attribute, '')
                hatch_color = 'white' if hatch else None  # Set hatch color to white for LIN bars
                rects = ax.bar(lab + offset_multiplier, data[soil_type][approach][:5], width, label=attribute, color=colors[attribute], hatch=hatch, edgecolor=hatch_color)

                # Adjust bar labels for readability and apply inclination
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),  # 3 points vertical offset
                                textcoords="offset points",
                                ha='center', va='bottom',
                                rotation=60)  # Rotate the annotations for readability

                offset_multiplier += width

            offset_multiplier = -1.5 * width  # Reset for next group

            # Customized Y-axis label based on the row index
            if idx == 0:
                if i == 0:
                    ax.set_ylabel(r'Site 1. Median $R^2$', fontsize=18)
                else:
                    ax.set_ylabel(r'Site 2. Median $R^2$', fontsize=18)
                ax.legend(loc='upper left', ncol=1)

            ax.set_xticks(lab + width / 2)
            ax.set_xticklabels(target_set, fontsize=16)
            ax.set_ylim(-0.25, 0.8)  # Adjust as per your data's range

    # Adding labels to the plot
    fig.text(0.20, 0.95, 'LT',  fontweight='bold', fontsize=14)
    fig.text(0.50, 0.95, 'LS',  fontweight='bold', fontsize=14)
    fig.text(0.85, 0.95, 'LS2', fontweight='bold', fontsize=14)

    plt.tight_layout()
    folder_path = 'output_images/'
    file_name_png = f"f7.png"

    plt.savefig(folder_path + file_name_png, bbox_inches='tight', dpi=300)  # Adjusted bbox_inches
    plt.show()


def f5(df, preds, targets, site):
        # TODO remove subtitles and add top/sub soil to the x axes legend. 

    global_max_x = float('-inf')

    # Adjusting the figure size and axes
    fig, axes = plt.subplots(len(targets), len(preds), figsize=(10, 3 * len(targets)))  # Increased width to accommodate text

    axes = np.atleast_2d(axes)  # Ensure axes is always 2D
    ss = 25  # Reduced size for scatter points

    pred_titles = {
        'bulk_ec_hp': 'Soil probe EC',
        'bulk_ec_inv_mS': 'FDEM inverted EC'
    }

    for target_index, target in enumerate(targets):
        for pred_index, pred in enumerate(preds):
            ax = axes[target_index, pred_index]
            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # General correlation coefficient
            general_corr = np.corrcoef(df[pred].dropna(), df[target].dropna())[0, 1]
            ax.text(0.05, 0.95, f'All: r={general_corr:.2f}', transform=ax.transAxes, va="center")

            for layer_cm in [10, 50]:
                x_data = df[df['depth'] == layer_cm][pred]
                y_data = df[df['depth'] == layer_cm][target]

                label = 'Topsoil' if layer_cm == 10 else 'Subsoil'
                ax.scatter(x_data, y_data * 100 if target == 'vwc' else y_data, s=ss, alpha=0.8, marker='o' if layer_cm == 10 else 'D', label=label)

            #if target_index == 0 and site != 'P':
                #ax.set_title(pred_titles.get(pred, pred), fontweight='bold', fontsize=14)
            if target_index != len(targets) - 1:
                ax.set_xticklabels([])
            if pred_index == 0:
                ax.set_ylabel('$θ$* [%]' if target == 'vwc' else target, fontsize=14)
            if pred_index != 0:
                ax.set_yticklabels([])

            if site != 'M':
                ax.set_xlabel('EC [S/m]', fontsize=12)

            global_max_x = max(global_max_x, x_data.max())

        if site != 'M':
            ax.legend(loc='lower right', prop={'weight': 'normal'})

            ax.set_ylim(0, 50)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, global_max_x)

    # Define the site name based on the site variable and position it
    site_name = 'Middelkerke' if site == 'M' else 'Proefhoeve' if site == 'P' else ''
    # Text is added vertically at the right outside the plot
    fig.text(1.005, 0.7, site_name, horizontalalignment='left', verticalalignment='center', fontsize=12, fontweight='bold', rotation=270, transform=fig.transFigure)

    # Adjust margins slightly to fit the text
    plt.subplots_adjust(right=0.95)  # Slightly decrease the right margin to fit text
    plt.tight_layout(pad=0.5, w_pad=0.4, h_pad=0.8)

    plt.savefig('output_images/'+str(site)+'f5.png', dpi=300)
    plt.show()


def SA_plot(file_path_all_, SA_results, indicator):
    # Load your DataFrame
    dtM = pd.read_csv(SA_results + 'dt' + 'M' + '_' + file_path_all_ + '.csv')
    dtP = pd.read_csv(SA_results + 'dt' + 'P' + '_' + file_path_all_ + '.csv')

    # Convert boolean columns to string for better plotting
    dtP['remove_coil'] = dtP['remove_coil'].astype(str)
    dtP['start_avg'] = dtP['start_avg'].astype(str)
    dtP['constrain'] = dtP['constrain'].astype(str)
    dtM['remove_coil'] = dtM['remove_coil'].astype(str)
    dtM['start_avg'] = dtM['start_avg'].astype(str)
    dtM['constrain'] = dtM['constrain'].astype(str)

    # Ordering 'Det' categories if needed
    if 'Det' in dtP.columns:
        dtP['Det'] = pd.Categorical(dtP['Det'], categories=["LT", "LS", "ID"], ordered=True)

    if 'Det' in dtM.columns:
        dtM['Det'] = pd.Categorical(dtM['Det'], categories=["LT", "LS", "ID"], ordered=True)

    # Set the overall font size for the plots
    plt.rcParams.update({'font.size': 15})

    # List of variables for which to create boxplots
    variables = ["Extract", 'Samples location', 'Interface', 'Forward_Model', 'Minimization_Method', 'Alpha', "remove_coil", "start_avg", "constrain", "Det"]

    # Determine the number of rows and columns needed
    num_vars = len(variables)
    num_rows = 2  # Two rows: 
    num_cols = num_vars  # Each column represents a variable

    # Initialize the figure with calculated rows and columns
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(2*num_vars, 12), sharex='col', sharey='row')
    axs = axs.ravel()  # Flatten the array of axes

    # Loop through each variable to create boxplots for dtM
    for i, var in enumerate(variables):
        sns.boxplot(x=var, y=indicator, data=dtM, ax=axs[i])
        axs[i].grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        axs[i].tick_params(axis='x', labelsize=16, rotation=40)
        axs[i].xaxis.set_label_position('top')
        axs[i].set_xlabel(var, fontsize=14)  # Set xlabel to the top
        axs[i].axhline(y=0.161, color='red', linestyle='--')  # Add red line at y=0.161
        if i == 0:  # Add the label only on the first plot of each row
            axs[i].text(-0.5, 0.161, r'$\theta_0$', verticalalignment='bottom', horizontalalignment='right')
        if i % num_cols != 0:
            axs[i].set_ylabel('')
            axs[i].tick_params(axis='y', left=False, labelleft=False)
        else:
            axs[i].set_ylabel(r'Site 1. Median $RMSE$ of $\theta$', fontsize=16)
    #ax2t = axs[-num_vars-1].twinx()
    #ax2t.set_ylabel(r'Site 1. Median $R^2$ of $\theta$')
    #ax2t.set_ylim(max(dtM.R2), min(dtM.R2))

    # Loop through each variable to create boxplots for dtP, remove x-axis labels on the bottom row
    for i, var in enumerate(variables):
        sns.boxplot(x=var, y=indicator, data=dtP, ax=axs[i + num_cols])
        axs[i + num_cols].grid(True, which='both', linestyle='-', linewidth=0.5, color='gray')
        axs[i + num_cols].tick_params(axis='x', labelsize=14, rotation=40)
        axs[i + num_cols].set_xlabel('')  # Remove x-labels on the bottom row
        axs[i + num_cols].axhline(y=0.274, color='red', linestyle='--')  # Add red line at y=0.274
        if (i + num_cols) == num_cols:  # Add the label only on the first plot of the bottom row
            axs[i + num_cols].text(-0.5, 0.274, r'$\theta_0$', verticalalignment='bottom', horizontalalignment='right')
        if (i + num_cols) % num_cols != 0:
            axs[i + num_cols].set_ylabel('')
            axs[i + num_cols].tick_params(axis='y', left=False, labelleft=False)
        else:
            axs[i + num_cols].set_ylabel(r'Site 2. Median $RMSE$ of $\theta$', fontsize=16)

    #ax2b = axs[-1].twinx()
    #ax2b.set_ylabel(r'Site 2. Median $R^2$ of $\theta$')
    #ax2b.set_ylim(max(dtP.R2), min(dtP.R2))

    # Adjust layout and add titles based on the site
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, hspace=0.1)

    # Save and show the plot
    plt.savefig('output_images/' + file_path_all_ + indicator + '.png', dpi=300)
    plt.show()


def f6(d1, d2, targets, preds):
    global_max_x = float('-inf')

    # Use constrained_layout for better handling of space
    fig, axes = plt.subplots(2, len(preds), figsize=(12, 6), layout='compressed')
    axes = np.atleast_2d(axes)  # Ensure axes is always 2D
    ss = 50  # Increased scatter point size for better visibility

    dataframes = [d1, d2]
    site_labels = ['Site 1 $θ$ [%]', 'Site 2 $θ$ [%]']
    x_labels = ['Probe EC [mS/m]', 'FDEM inverted EC [mS/m]', 'Ideal EC [mS/m]']

    scatter_plots = []

    for row_index, df in enumerate(dataframes):
        for pred_index, pred in enumerate(preds):
            ax = axes[row_index, pred_index]
            ax.grid(True, which='both', linestyle='--')
            ax.set_box_aspect(1)

            for layer_cm, marker in zip([10, 50], ['o', '^']):
                x_data = df[df['depth'] == layer_cm][pred] * 1000
                y_data = df[df['depth'] == layer_cm][targets[0]]
                color_data = df[df['depth'] == layer_cm]['clay']

                scatter = ax.scatter(x_data, y_data * 100 if targets[0] == 'vwc' else y_data, 
                                     c=color_data, cmap='copper_r', s=ss, alpha=0.7, marker=marker,
                                     label='Topsoil' if layer_cm == 10 else 'Subsoil',
                                     vmin=0, vmax=50)
                scatter_plots.append(scatter)

                if not x_data.empty:
                    global_max_x = max(global_max_x, x_data.max())

            if row_index == 0:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(x_labels[pred_index], fontsize=12)

            if pred_index == 0:
                ax.set_ylabel(site_labels[row_index], fontsize=14)
            else:
                ax.set_yticklabels([])

            ax.set_ylim(0, 50)

            # Add legend only to the upper left subplot
            if row_index == 0 and pred_index == 0:
                ax.legend(loc='upper right')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_xlim(0, global_max_x)

    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    fig.colorbar(scatter_plots[0], pad=0.2, cax=cbar_ax, label='Clay content [%]', shrink=0.6)
    for scatter in scatter_plots:
        scatter.set_clim(0, 50)

    plt.savefig('output_images/f6.png', dpi=300)
    plt.show()


def f10(d1, d2, feat):
    # Create a figure and axes
    fig, axs = plt.subplots(3, 1, figsize=(8, 15))
    corr1 = np.corrcoef(d1.res_ideal_hp, d1[feat])[0, 1]
    corr2 = np.corrcoef(d2.res_ideal_hp, d2[feat])[0, 1]

    # Define global x and y limits
    x_min = min(d1.res_ideal_hp.min(), d2.res_ideal_hp.min())
    x_max = max(d1.res_ideal_hp.max(), d2.res_ideal_hp.max())
    y_max = max(d1[feat].max(), d2[feat].max())

    # Top row plot: diff1 vs d1.clay
    axs[0].scatter(d1.res_ideal_hp, d1[feat], marker='o')
    axs[0].set_title('Diff1 vs D1.Clay')
    axs[0].set_xlabel('Diff1')
    axs[0].set_ylabel('D1.Clay')
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(0, y_max)
    # Annotate with correlation
    axs[0].annotate(f'Correlation: {corr1:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Middle row plot: diff2 vs d2.clay
    axs[1].scatter(d2.res_ideal_hp, d2[feat], marker='o')
    axs[1].set_title('Diff2 vs D2.Clay')
    axs[1].set_xlabel('Diff2')
    axs[1].set_ylabel('D2.Clay')
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(0, y_max)
    axs[1].annotate(f'Correlation: {corr2:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Combine data
    combined_x = np.concatenate((d1.res_ideal_hp, d2.res_ideal_hp))
    combined_y = np.concatenate((d1[feat], d2[feat]))
    corr_combined = np.corrcoef(combined_x, combined_y)[0, 1]

    # Fit a linear function to the combined data
    slope, intercept, r_value, p_value, std_err = stats.linregress(combined_x, combined_y)
    fit_line = slope * combined_x + intercept

    # Bottom row plot: combined data
    axs[2].scatter(combined_x, combined_y, marker='o')
    axs[2].plot(combined_x, fit_line, color='red', linestyle='--', linewidth=2)
    axs[2].set_title('Combined Diff vs Clay with Fit')
    axs[2].set_xlabel('Diff')
    axs[2].set_ylabel('Clay')
    axs[2].set_xlim(x_min, x_max)
    axs[2].set_ylim(0, y_max)
    axs[2].annotate(f'Correlation: {corr_combined:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                    fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Adjust layout
    plt.tight_layout()
    # Show plot
    plt.show()


def fig3(comb_dfM, comb_dfP, extract):
    # Setup figure
    num_rows = 2
    num_cols = 3

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

    # Define configurations to plot
    column_configs1 = ['HCP1QP', 'HCP2QP', 'HCP4QP', 'PRP1QP', 'PRP2QP', 'PRP4QP']
    column_configs2 = ['HCPHQP', 'HCP1QP', 'HCP2QP', 'PRPHQP', 'PRP1QP', 'PRP2QP']

    # Store regression parameters
    regression_params1 = {}
    regression_params2 = {}

    # Fixed axis limits
    left_min, left_max = 50, 350
    right_min, right_max = 20, 115

    # Plotting logic for Site 1 (comb_dfM)
    for i in range(len(column_configs1)):

        measured_col = f'{column_configs1[i]}'
        modeled_col = f'forward_{column_configs1[i]}'  # Assumes naming convention
        X1 = comb_dfM[measured_col].values.reshape(-1, 1)
        y1 = comb_dfM[modeled_col].values
        regression1 = LinearRegression().fit(X1, y1)
        regression_params1[measured_col] = (regression1.coef_[0], regression1.intercept_)
        reg_line1 = regression1.predict(X1)

        if i<=2:
            ax = axes[0, i]
            ax.grid(True)

            if ax == axes[0, 0]:
                scatter = ax.scatter(X1, y1, edgecolor='k', alpha=0.7, label='Data Points')
                ax.plot([left_min, left_max], [left_min, left_max], 'r--', label='1:1 Line')
                ax.plot(X1, reg_line1, 'g-', label='Regression Line')
            else:
                scatter = ax.scatter(X1, y1, edgecolor='k', alpha=0.7)
                ax.plot([left_min, left_max], [left_min, left_max], 'r--')
                ax.plot(X1, reg_line1, 'g-')
                
            ax.set_xlim([left_min, left_max])
            ax.set_ylim([left_min, left_max])
            ax.set_aspect('equal', adjustable='box')  # Setting aspect ratio to equal
            axes[0, 0].legend(title='HCP1.0 QP', loc='upper left')
            axes[0, 1].legend(title='HCP2.0 QP', loc='upper left')
            axes[0, 2].legend(title='HCP4.0 QP', loc='upper left')

            ax.set_ylabel('')

        if i > 0:  # Remove y-axis labels for center and right column subplots
            ax.set_yticklabels([])

    axes[0, 0].set_ylabel('Site 1. FW Modeled ERT LIN ECa [mS/m]', fontsize=14)

    # Plotting logic for Site 2 (comb_dfP)
    for i in range(len(column_configs2)):

        measured_col = f'{column_configs2[i]}'
        modeled_col = f'forward_{column_configs2[i]}'  # Assumes naming convention
        X2 = comb_dfP[measured_col].values.reshape(-1, 1)
        y2 = comb_dfP[modeled_col].values
        regression2 = LinearRegression().fit(X2, y2)
        regression_params2[measured_col] = (regression2.coef_[0], regression2.intercept_)
        reg_line2 = regression2.predict(X2)
        
        if i<=2:
            ax = axes[1, i]
            ax.grid(True)
            scatter = ax.scatter(X2, y2, edgecolor='k', alpha=0.7)
            ax.plot([right_min, right_max], [right_min, right_max], 'r--')
            ax.plot(X2, reg_line2, 'g-')
            ax.set_xlim([right_min, right_max])
            ax.set_ylim([right_min, right_max])
            ax.set_aspect('equal', adjustable='box')  # Setting aspect ratio to equal
            axes[1, 0].legend(title='HCP0.5 QP', loc='upper left')
            axes[1, 1].legend(title='HCP1.0 QP', loc='upper left')
            axes[1, 2].legend(title='HCP2.0 QP', loc='upper left')
            
            ax.set_ylabel('')

        if i > 0:  # Remove y-axis labels for center and right column subplots
            ax.set_yticklabels([])

    axes[1, 0].set_ylabel('Site 2. FW Modeled ERT LIN ECa [mS/m]', fontsize=14)

    # Set common xlabel
    for ax in axes[1, :]:
        ax.set_xlabel(f'FDEM LIN ECa [mS/m]', fontsize=14)

    # Adjust layout to reduce white spaces
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    # Save and show the plot
    cal_metaname = os.path.join('output_images', f'Calibration_plots_'+str(extract)+'.png')
    plt.savefig(cal_metaname, dpi=300)
    plt.show()

    return regression_params1, regression_params2