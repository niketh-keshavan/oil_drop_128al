"""
Generate all figures for the Millikan Oil Drop Experiment lab report.
This script creates publication-quality plots showing:
1. Charge quantization pattern
2. Ionization effect across trials
3. Error budget breakdown
4. Experimental vs. CODATA comparison
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle

# Set up plotting style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['axes.linewidth'] = 1.2

# ============================================================================
# Figure 1: Charge Quantization Pattern
# ============================================================================

def create_quantization_plot():
    """Create scatter plot showing charge quantization at n=1 and n=2 levels."""
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Measured charges (in 10^-18 C) with uncertainties
    trials = ['T1 Pre', 'T1 Post', 'T2 Pre', 'T2 Post', 'T3 Pre', 'T3 Post']
    charges = np.array([2.80, 1.86, 3.78, 2.08, 3.12, 1.75])  # 10^-18 C
    uncertainties = np.array([0.21, 0.17, 0.27, 0.18, 0.24, 0.16])
    
    # Convert to elementary charge units
    e_codata = 1.60217663e-19  # C
    n_values = charges * 1e-18 / e_codata
    n_errors = uncertainties * 1e-18 / e_codata
    
    # X positions for plotting
    x_pos = np.arange(len(trials))
    
    # Pre-ionization (circles) and post-ionization (squares)
    colors = ['steelblue', 'coral', 'steelblue', 'coral', 'steelblue', 'coral']
    markers = ['o', 's', 'o', 's', 'o', 's']
    
    for i, (x, n, err, color, marker) in enumerate(zip(x_pos, n_values, n_errors, colors, markers)):
        ax.errorbar(x, n, yerr=err, fmt=marker, color=color, markersize=8, 
                    capsize=5, capthick=1.5, elinewidth=1.5, alpha=0.8)
    
    # Add quantization level reference lines
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.6, label='n = 1')
    ax.axhline(y=2.0, color='green', linestyle='--', linewidth=2, alpha=0.6, label='n = 2')
    
    # Formatting
    ax.set_xlabel('Measurement', fontsize=11)
    ax.set_ylabel('Charge (elementary charge units)', fontsize=11)
    ax.set_title('Charge Quantization Pattern', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(trials, rotation=45, ha='right')
    ax.set_ylim([0.5, 2.8])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Legend
    pre_patch = mpatches.Patch(color='steelblue', label='Pre-ionization', alpha=0.8)
    post_patch = mpatches.Patch(color='coral', label='Post-ionization', alpha=0.8)
    ax.legend(handles=[pre_patch, post_patch], loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('charge_quantization.png', dpi=300, bbox_inches='tight')
    print("✓ Created: charge_quantization.png")
    plt.close()

# ============================================================================
# Figure 2: Ionization Effect Across Trials
# ============================================================================

def create_ionization_effect_plot():
    """Create bar chart showing charge reduction due to ionization."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Data: pre and post ionization charges
    trials = ['Trial 1', 'Trial 2', 'Trial 3']
    pre_ionization = np.array([2.80, 3.78, 3.12])  # 10^-18 C
    post_ionization = np.array([1.86, 2.08, 1.75])  # 10^-18 C
    
    x = np.arange(len(trials))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, pre_ionization, width, label='Pre-ionization', 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, post_ionization, width, label='Post-ionization', 
                   color='coral', alpha=0.8, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Formatting
    ax.set_xlabel('Trial', fontsize=11)
    ax.set_ylabel('Charge ($10^{-18}$ C)', fontsize=11)
    ax.set_title('Ionization-Induced Charge Changes', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(trials)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_ylim([0, 4.5])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('ionization_effect.png', dpi=300, bbox_inches='tight')
    print("✓ Created: ionization_effect.png")
    plt.close()

# ============================================================================
# Figure 3: Error Budget Pie Chart
# ============================================================================

def create_error_budget_plot():
    """Create pie chart showing contribution of each error source."""
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Error contributions (percentages)
    error_sources = ['Rising velocity\n(4.8%)', 
                     'Falling velocity\n(5.3%)', 
                     'Voltage + Plate sep.\n(0.3%)']
    contributions = [4.8, 5.3, 0.3]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    explode = (0.05, 0.05, 0)
    
    # Create pie chart
    wedges, texts, autotexts = ax.pie(contributions, labels=error_sources, autopct='%1.1f%%',
                                        colors=colors, explode=explode, startangle=90,
                                        textprops={'fontsize': 10})
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(11)
    
    ax.set_title('Uncertainty Budget: Sources of Error\n(Velocity-Limited Measurement)', 
                 fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('error_budget_pie.png', dpi=300, bbox_inches='tight')
    print("✓ Created: error_budget_pie.png")
    plt.close()

# ============================================================================
# Figure 4: Experimental vs. CODATA Comparison
# ============================================================================

def create_result_comparison_plot():
    """Create bar chart comparing experimental and CODATA values."""
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Values
    e_experimental = 1.62e-19  # C
    e_codata = 1.602176634e-19  # C
    uncertainty = 0.14e-19  # C
    
    # Create bar for experimental value
    bar_width = 0.4
    x_exp = 0
    
    # Bar with uncertainty band
    ax.bar(x_exp, e_experimental, bar_width, 
           color='steelblue', alpha=0.7, edgecolor='black', linewidth=1.5,
           label='Experimental')
    
    # Add error bar
    ax.errorbar(x_exp, e_experimental, yerr=uncertainty, 
               fmt='none', ecolor='steelblue', elinewidth=3, capsize=8, capthick=2)
    
    # Add CODATA reference line
    ax.axhline(y=e_codata, color='red', linestyle='-', linewidth=2.5, 
              label='CODATA 2018', zorder=3)
    
    # Formatting
    ax.set_ylabel('Elementary Charge (C)', fontsize=11)
    ax.set_title('Experimental Result vs. Accepted Value', fontsize=12, fontweight='bold')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([1.55e-19, 1.65e-19])
    ax.set_xticks([])
    
    # Format y-axis in scientific notation
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add text annotation
    ax.text(x_exp, e_experimental + uncertainty + 0.005e-19, 
           f'Measured:\n{e_experimental:.2e} C\n±{uncertainty:.2e} C',
           ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Agreement percentage
    agreement = abs(e_experimental - e_codata) / e_codata * 100
    ax.text(0.25, 1.605e-19, f'Agreement:\n{agreement:.1f}%',
           fontsize=10, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('result_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Created: result_comparison.png")
    plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == '__main__':
    print("Generating publication-quality figures for lab report...\n")
    
    create_quantization_plot()
    create_ionization_effect_plot()
    create_error_budget_plot()
    create_result_comparison_plot()
    
    print("\n✅ All figures generated successfully!")
    print("\nGenerated files:")
    print("  1. charge_quantization.png    - Charge quantization at n=1 and n=2")
    print("  2. ionization_effect.png      - Charge reduction due to ionization")
    print("  3. error_budget_pie.png       - Error source contributions")
    print("  4. result_comparison.png      - Experimental vs. CODATA comparison")
