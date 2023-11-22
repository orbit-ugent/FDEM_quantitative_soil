import matplotlib.pyplot as plt
import numpy as np

def plot3(ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12, ax13, ax14, ax15):
    ax1.legend(loc='upper right', fontsize = 8)
    #ax1.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax1.tick_params(axis='y', labelsize=12) 
    ax1.tick_params(axis='x', labelsize=12) 
    #ax1.set_xlabel('Clay [%]', fontsize = 16) 
    ax1.set_ylabel('Kfd [%]', fontsize = 16) 
    ax1.grid(True) 
    ax1.set_ylim(0, 5e-3)  
    ax1.set_xlim(0, 80) 
    ax1.legend(loc='upper right', fontsize = 10)

    #ax2.legend(loc='upper right', fontsize = 8)
    #ax2.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax2.tick_params(axis='y', labelsize=12) 
    ax2.tick_params(axis='x', labelsize=12) 
    #ax2.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax2.set_ylabel('Kfd [%]', fontsize = 16) 
    ax2.grid(True) 
    ax2.set_ylim(0, 5e-3)  
    ax2.set_xlim(0, 45) 
    ax2.legend(loc='upper right', fontsize = 10)

    ax3.legend(loc='upper right', fontsize = 8)
    #ax3.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax3.tick_params(axis='y', labelsize=12) 
    ax3.tick_params(axis='x', labelsize=12) 
    #ax3.set_xlabel('Clay [%]', fontsize = 16) 
    ax3.set_ylabel('Kfd [%]', fontsize = 16) 
    ax3.grid(True) 
    ax3.set_ylim(0, 5e-3)  
    ax3.set_xlim(0, 80) 
    ax3.legend(loc='upper right', fontsize = 10)

    #ax4.legend(loc='upper right', fontsize = 8)
    #ax4.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax4.tick_params(axis='y', labelsize=12) 
    ax4.tick_params(axis='x', labelsize=12) 
    #ax4.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax4.set_ylabel('Kfd [%]', fontsize = 16) 
    ax4.grid(True) 
    ax4.set_ylim(0, 5e-3)  
    ax4.set_xlim(0, 45) 
    ax4.legend(loc='upper right', fontsize = 10)

    #ax5.legend(loc='upper right', fontsize = 8)
    #ax5.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax5.tick_params(axis='y', labelsize=12) 
    ax5.tick_params(axis='x', labelsize=12) 
    ax5.set_xlabel('Clay [%]', fontsize = 16) 
    ax5.set_ylabel('Kfd [%]', fontsize = 16) 
    ax5.grid(True) 
    ax5.set_ylim(0, 5e-3)  
    ax5.set_xlim(0, 80) 
    ax5.legend(loc='upper right', fontsize = 10)

    #ax6.legend(loc='upper right', fontsize = 8)
    #ax6.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax6.tick_params(axis='y', labelsize=12) 
    ax6.tick_params(axis='x', labelsize=12) 
    ax6.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax6.set_ylabel('Kfd [%]', fontsize = 16) 
    ax6.grid(True) 
    ax6.set_ylim(0, 5e-3)  
    ax6.set_xlim(0, 45) 
    ax6.legend(loc='upper right', fontsize = 10)

    #ax7.legend(loc='upper right', fontsize = 8)
    #ax7.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax7.tick_params(axis='y', labelsize=12) 
    ax7.tick_params(axis='x', labelsize=12) 
    ax7.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax7.set_ylabel('Kfd [%]', fontsize = 16) 
    ax7.grid(True) 
    ax7.set_ylim(0, 5e-3)  
    ax7.set_xlim(0, 45) 
    ax7.legend(loc='upper right', fontsize = 10)

    #ax8.legend(loc='upper right', fontsize = 8)
    #ax8.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax8.tick_params(axis='y', labelsize=12) 
    ax8.tick_params(axis='x', labelsize=12) 
    ax8.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax8.set_ylabel('Kfd [%]', fontsize = 16) 
    ax8.grid(True) 
    ax8.set_ylim(0, 5e-3)  
    ax8.set_xlim(0, 45) 
    ax8.legend(loc='upper right', fontsize = 10)

    #ax9.legend(loc='upper right', fontsize = 8)
    #ax9.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax9.tick_params(axis='y', labelsize=12) 
    ax9.tick_params(axis='x', labelsize=12) 
    ax9.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax9.set_ylabel('Kfd [%]', fontsize = 16) 
    ax9.grid(True) 
    ax9.set_ylim(0, 5e-3)  
    ax9.set_xlim(0, 45) 
    ax9.legend(loc='upper right', fontsize = 10)

    #ax10.legend(loc='upper right', fontsize = 8)
    #ax10.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax10.tick_params(axis='y', labelsize=12) 
    ax10.tick_params(axis='x', labelsize=12) 
    ax10.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax10.set_ylabel('Kfd [%]', fontsize = 16) 
    ax10.grid(True) 
    ax10.set_ylim(0, 5e-3)  
    ax10.set_xlim(0, 45) 
    ax10.legend(loc='upper right', fontsize = 10)

    #ax11.legend(loc='upper right', fontsize = 8)
    #ax11.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax11.tick_params(axis='y', labelsize=12) 
    ax11.tick_params(axis='x', labelsize=12) 
    ax11.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax11.set_ylabel('Kfd [%]', fontsize = 16) 
    ax11.grid(True) 
    ax11.set_ylim(0, 5e-3)  
    ax11.set_xlim(0, 45) 
    ax11.legend(loc='upper right', fontsize = 10)

    #ax12.legend(loc='upper right', fontsize = 8)
    #ax12.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax12.tick_params(axis='y', labelsize=12) 
    ax12.tick_params(axis='x', labelsize=12) 
    ax12.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax12.set_ylabel('Kfd [%]', fontsize = 16) 
    ax12.grid(True) 
    ax12.set_ylim(0, 5e-3)  
    ax12.set_xlim(0, 45) 
    ax12.legend(loc='upper right', fontsize = 10)

    #ax13.legend(loc='upper right', fontsize = 8)
    #ax13.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax13.tick_params(axis='y', labelsize=12) 
    ax13.tick_params(axis='x', labelsize=12) 
    ax13.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax13.set_ylabel('Kfd [%]', fontsize = 16) 
    ax13.grid(True) 
    ax13.set_ylim(0, 5e-3)  
    ax13.set_xlim(0, 45) 
    ax13.legend(loc='upper right', fontsize = 10)

    #ax14.legend(loc='upper right', fontsize = 8)
    #ax14.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax14.tick_params(axis='y', labelsize=12) 
    ax14.tick_params(axis='x', labelsize=12) 
    ax14.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax14.set_ylabel('Kfd [%]', fontsize = 16) 
    ax14.grid(True) 
    ax14.set_ylim(0, 5e-3)  
    ax14.set_xlim(0, 45) 
    ax14.legend(loc='upper right', fontsize = 10)

    #ax15.legend(loc='upper right', fontsize = 8)
    #ax15.set_title("Susceptibility vs clay" , fontweight='bold', fontsize=20) 
    ax15.tick_params(axis='y', labelsize=12) 
    ax15.tick_params(axis='x', labelsize=12) 
    ax15.set_xlabel('CEC [meq/100g]', fontsize = 16) 
    #ax15.set_ylabel('Kfd [%]', fontsize = 16) 
    ax15.grid(True) 
    ax15.set_ylim(0, 5e-3)  
    ax15.set_xlim(0, 45) 
    ax15.legend(loc='upper right', fontsize = 10)


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