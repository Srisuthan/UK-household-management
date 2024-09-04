import matplotlib.pyplot as plt
import numpy as np

def generate_comparison_graphs(season, avg_electricity_usage_trained, total_electricity_usage_trained, avg_cost_trained, total_cost_trained,
                               avg_electricity_usage_random, total_electricity_usage_random, avg_cost_random, total_cost_random):
    # Data for bar charts
    categories = ['Average Electricity Usage', 'Total Electricity Usage', 'Average Cost', 'Total Cost']
    trained_values = [avg_electricity_usage_trained, total_electricity_usage_trained, avg_cost_trained, total_cost_trained]
    random_values = [avg_electricity_usage_random, total_electricity_usage_random, avg_cost_random, total_cost_random]

    # Plotting the comparison
    plt.figure(figsize=(12, 6))
    x = np.arange(len(categories))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width/2, trained_values, width, label='Trained Agent')
    bars2 = ax.bar(x + width/2, random_values, width, label='Random Policy')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(f'{season.capitalize()} Season - Comparison of Trained Agent vs Random Policy')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Adding labels to the bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()
    # Save the figure to a file
    plt.savefig(f"comparison_{season}.png")
    plt.close()  # Close the plot to free memory
