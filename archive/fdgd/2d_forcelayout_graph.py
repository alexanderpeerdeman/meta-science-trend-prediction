import forcelayout as fl
import matplotlib.pyplot as plt
import numpy as np

iris_data = None
with open(f'iris.csv', encoding='utf8') as iris_file:
    iris_data = np.loadtxt(
        iris_file,
        skiprows=1,
        usecols=(0, 1, 2, 3),
        delimiter=',',
        comments='#'
    )

ani = fl.draw_spring_layout_animated(dataset=iris_data,
                                     algorithm=fl.NeighbourSampling,
                                     alpha=0.5,
                                     algorithm_highlights=True)

plt.savefig('iris.png')

plt.show()