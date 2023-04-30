from numpy import *
import numba
import drag
import hpc_design_curve as auv
from traits.api import  Array, HasTraits, Range, Instance, observe
from traitsui.api import View, Item, Group
from mayavi import mlab
from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, SceneEditor, \
                MlabSceneModel

def print_quantities(r_max, nn, nt, ln, lt):
    vol = auv.get_total_volume(r_max, nn, nt, ln, lt)
    print("\n\n")
    print("For r_max = {} m, nn = {}, nt = {}, ln = {} m, lt = {} m ".format(r_max, nn, nt, ln, lt))
    print("Total Volume of ", vol, "m\u00b3")
    print("Total Drag of " ,drag.total_drag(r_max, nn, nt, ln, lt), "N")
    print("Total Surface area of ", auv.get_total_surface_area(r_max, nn, nt, ln, lt), "m\u00b2")
    print("Wake fraction = ", auv.calc_wake_fraction(r_max, 
                                                     auv.get_block_coefficient(r_max, vol), vol))
    
@numba.njit
def auv_hull(r_max, nn, nt, ln, lt):
    #Params
    n_x = 100
    l = 3

    x = linspace(0, l, n_x)
    y = zeros_like(x)
    z = zeros_like(x)
    s = zeros_like(x) #Radius section

    #Nose Section
    for i in numba.prange(int(ln*n_x/l)):
        s[i] = auv.radius_of_nose(r_max, ln, nn, x[i])

    #The mid section
    for m in numba.prange(int((l - lt)*n_x/l) - int(ln*n_x/l)):
        idx = int(ln*n_x/l) + m
        s[idx] = r_max
    
    #The tail section
    for b in numba.prange(int(lt*n_x/l)):
        idx = int((l - lt)*n_x/l) + b
        s[idx] = auv.radius_of_tail(r_max, lt, nt, x[b])

    return x, y, z, s


class MyModel(HasTraits):
    r_max = Range(0.25, 0.5, 0.3)
    nn = Range(2., 4.5, 2)
    nt = Range(2., 4.5, 2)
    ln = Range(low = 0.25, high= 0.7, value = 0.25)
    lt = Range(0.25, 0.7, 0.25)
    x = Array(dtype=float, shape=(None,))
    y = Array(dtype=float, shape=(None,))
    z = Array(dtype=float, shape=(None,))
    s = Array(dtype=float, shape=(None,))
    #Warmup
    auv_hull(0.5, 2, 2, 0.5, 0.5)
    auv.warmup_numba_funcs()

    scene = Instance(MlabSceneModel, ())
    plot = Instance(PipelineBase)


    # When the scene is activated, or when the parameters are changed, we
    # update the plot.
    @observe('r_max, nn, nt, ln, lt, scene.activated')
    def update_plot(self, event=None):
        self.x, self.y, self.z, self.s = auv_hull(self.r_max, self.nn, self.nt, self.ln, self.lt)
        print_quantities(self.r_max, self.nn, self.nt, self.ln, self.lt)
        if self.plot is None:
            # First we need to make a line source from our data
            line = mlab.pipeline.line_source(self.x, self.y, self.z, self.s)
            # Then we apply the "tube" filter to it, and vary the radius by "s"
            tube = mlab.pipeline.tube(line, tube_sides=5, tube_radius=0.25)
            tube.filter.vary_radius = 'vary_radius_by_scalar'
            self.plot = self.scene.mlab.pipeline.surface(tube, extent = [0,3, 
                                                                        -self.r_max, self.r_max,
                                                                        -self.r_max, self.r_max,])
        else:
            mlab.clf()
            line = mlab.pipeline.line_source(self.x, self.y, self.z, self.s)
            # Then we apply the "tube" filter to it, and vary the radius by "s"
            tube = mlab.pipeline.tube(line, tube_sides=5, tube_radius=0.25)
            tube.filter.vary_radius = 'vary_radius_by_scalar'
            self.plot = self.scene.mlab.pipeline.surface(tube, extent = [0,3, 
                                                                        -self.r_max, self.r_max,
                                                                        -self.r_max, self.r_max,])

    # The layout of the dialog created
    view =  View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                Group(
                        '_', 'r_max', 'nn', 'nt', 'ln', 'lt',
                     ),
                resizable=True,
                )


my_model = MyModel()
my_model.configure_traits()