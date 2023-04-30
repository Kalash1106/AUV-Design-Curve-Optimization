import numpy as np
import drag
import hpc_design_curve as auv
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.optimize import minimize
from pymoo.factory import get_problem
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt

#This function uses each parameter xi* scaled between 0 and 1 and extrapolates it to its actual value. x = ax* + b
def extract_params(r_max, nn, nt, ln, lt):
    r_max = 0.25 * r_max + 0.25
    nn = 2.5 * nn + 2
    nt = 2.5 * nt + 2
    ln = 0.4 * ln + 0.25
    lt = 0.4 * lt + 0.25

    return [r_max, nn, nt, ln, lt]

#Choice between Scatter, PCP, Radviz
def plot_stack(F, choice):

    if(choice == 'Scatter'):
        from pymoo.visualization.scatter import Scatter
        plot = Scatter(legend=(True, {'loc': "upper left", 'bbox_to_anchor': (-0.1, 1.08, 0, 0)}),
                       labels = ['Drag', 'Negative of Mid-Section Volume', 'Wake Fraction'])
        plot.add(F, color = 'blue', label = 'Pareto-Front')
        plot.show()

    if(choice == 'PCP'):
        from pymoo.visualization.pcp import PCP
        plot = PCP(legend=(True, {'loc': 'lower right'}),
                   labels = ['Drag', 'Negative of Mid-Section Volume', 'Wake Fraction'])
        plot.set_axis_style(color="grey", alpha=0.5)
        plot.add(F, color="black", alpha=0.3)
        plot.add(F[1], linewidth=5, color="red", label = 'Least Drag')
        plot.add(F[7], linewidth=5, color="blue", label = 'Greatest Mid-Section Volume')
        plot.add(F[18], linewidth=5, color="green", label = 'Least Wake Fraction')
        plot.legend
        plot.show()

    if(choice == 'Radviz'):
        from pymoo.visualization.radviz import Radviz
        Radviz().add(F).show()

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []

    def notify(self, algorithm):
        drag = algorithm.pop.get("F")[:,0]
        self.data["best"].append(drag.min())

#The actual class containing the parameters and objective, constraint functions
class auv_problem(ElementwiseProblem):
    def __init__(self,**kwargs):
        super().__init__(n_var = 5, n_obj = 3,n_ieq_constr=1, xl=np.zeros(5),
                         xu=np.ones(5), **kwargs)

    def _evaluate(self, x, out, **Vol_min):
        r_max, nn, nt, ln, lt = extract_params(x[0], x[1], x[2], x[3], x[4])
        total_volume = auv.get_total_volume(r_max, nn, nt, ln, lt)

        drag_val = drag.total_drag(r_max, nn, nt, ln, lt)
        cylindrical_volume = auv.volume_of_midsection(r_max, 3 - ln - lt)
        wake_fraction = auv.calc_wake_fraction(r_max, auv.get_block_coefficient(r_max, total_volume), total_volume)


        out["F"] = [drag_val, -cylindrical_volume, wake_fraction]
        out["G"] = 0.6 - total_volume


if __name__ == '__main__':

    problem = auv_problem()
    # create the reference directions to be used for the optimization
    ref_dirs = get_reference_directions("energy", 3, 250, seed = 1)

    algorithm = NSGA3(
    pop_size = 250,
    ref_dirs= ref_dirs
    )

F_list = []
res = minimize(problem,
               algorithm,
               seed=1,
               termination=('n_gen', 150),
               callback=MyCallback(),
               verbose=False)

xl, xu = problem.bounds()

val = res.algorithm.callback.data["best"]
plt.plot(np.arange(len(val)), val)
plt.xlabel("Number of Generations")
plt.ylabel("Minimum Drag of that Generation (N)")
plt.title("Evolution of Drag in NSGA-3")
plt.show()

# X = res.opt.get("X")
# X[:,0] = 0.25 * (1 + X[:,0])
# X[:,1] = 2.5 * X[:,1] + 2
# X[:,2] = 2.5 * X[:,2] + 2
# X[:,3] = 0.4 * X[:,3] + 0.25
# X[:,4] = 0.4 * X[:,4] + 0.25

F = res.opt.get("F")
print(F)
# plot_stack(F, 'Scatter')
