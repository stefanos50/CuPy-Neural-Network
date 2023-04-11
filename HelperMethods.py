from matplotlib import pyplot as plt

def plot_result(x,y,title,y_label,x_label,x_legend,y_legend):
    plt.plot(x)
    plt.plot(y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend([x_legend, y_legend], loc='upper left')
    plt.show()