import matplotlib.pyplot as plt
from pathlib import Path

# path to save plots
save_plot_path = Path(__file__).parent / 'plots'

# plot 1(bar plot)

fig = plt.figure(figsize=(12,6))
x_points = ['Random Forest','XG Boost','Decision Trees']
y_points = [0.89, 0.94, 0.75]
colors= ['red','green','blue']

plt.bar(x=x_points,height=y_points,label=x_points,color=colors)
plt.ylim((0,1))
plt.legend()
plt.savefig(save_plot_path / 'bar_plot.png')
