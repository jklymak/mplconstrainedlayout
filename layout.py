# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import kiwisolver as kiwi
import matplotlib
matplotlib.use('qt5agg')
from matplotlib import pyplot as plt

Variable = kiwi.Variable


# plt.close('all')


class Box(object):
    """
    Basic rectangle representation using variables
    """

    def __init__(self, parent=None, name='',
                 lower_left=(0, 0), upper_right=(1, 1)):
        self.parent = parent
        self.name = name
        sn = self.name + '_'
        if parent is None:
            self.solver = kiwi.Solver()
        else:
            self.solver = parent.solver
        self.top = Variable(sn + 'top')
        self.bottom = Variable(sn + 'bottom')
        self.left = Variable(sn + 'left')
        self.right = Variable(sn + 'right')

        self.width = Variable(sn + 'width')
        self.height = Variable(sn + 'height')
        self.h_center = Variable(sn + 'h_center')
        self.v_center = Variable(sn + 'v_center')

        self.min_width = Variable(sn + 'min_width')
        self.min_height = Variable(sn + 'min_height')
        self.pref_width = Variable(sn + 'pref_width')
        self.pref_height = Variable(sn + 'pref_height')

        right, top = upper_right
        left, bottom = lower_left
        self.add_constraints()

    def add_constraints(self):
        sol = self.solver
        for i in [self.min_width, self.min_height]:
            sol.addEditVariable(i, 1e9)
            sol.suggestValue(i, 0)
        self.hard_constraints()
        self.soft_constraints()
        sol.updateVariables()

    def hard_constraints(self):
        hc = [self.width == self.right - self.left,
              self.height == self.top - self.bottom,
              self.h_center == (self.left + self.right) * 0.5,
              self.v_center == (self.top + self.bottom) * 0.5,
              self.width >= self.min_width,
              self.height >= self.min_height]
        for c in hc:
            self.solver.addConstraint(c)

    def soft_constraints(self):
        sol = self.solver
        for i in [self.pref_width, self.pref_height]:
            sol.addEditVariable(i, 'strong')
            sol.suggestValue(i, 0)
        c = [(self.pref_width == self.width),
             (self.pref_height == self.height)]
        for i in c:
            sol.addConstraint(i|0.000001)


    def set_geometry(self, left, bottom, right, top, strength=1e9):
        sol = self.solver
        for i in [self.top, self.bottom,
                  self.left, self.right]:
            if not sol.hasEditVariable(i):
                sol.addEditVariable(i, strength)

        sol.suggestValue(self.top, top)
        sol.suggestValue(self.bottom, bottom)
        sol.suggestValue(self.left, left)
        sol.suggestValue(self.right, right)
        sol.updateVariables()

    def get_mpl_rect(self):
        return (self.left.value(), self.bottom.value(),
                self.width.value(), self.height.value())

    def __repr__(self):
        args = (self.name, self.left.value(), self.bottom.value(),
                self.right.value(), self.top.value())
        return 'Rect: %s, (left: %d) (bot: %d)  (right: %d) (top: %d)'%args


class GridLayout(object):

    def __init__(self, rows, cols, width=100, height=100):
        self.rows, self.cols = rows, cols
        self.calc_borders(width, height)

    def calc_borders(self, width, height):
        cols, rows = self.cols, self.rows
        self.left_borders = [width / cols * i for i in range(cols)]
        self.right_borders = [width / cols * i for i in range(cols + 1)]
        self.top_borders = [height - height / rows * i for i in range(rows)]
        self.bottom_borders = [height - height / rows * i for i in range(rows+1)]

    def place_rect(self, rect, pos, colspan=1, rowspan=1):
        start_row, start_col = pos
        end_col = start_col + colspan
        end_row = start_row + rowspan

        left, right = self.left_borders[start_col], self.right_borders[end_col]
        top, bottom = self.top_borders[start_row], self.bottom_borders[end_row]
        print(left)
        print(right,top,bottom)

        rect.set_geometry(left, bottom, right, top)


def align(items, attr, strength='weak'):
    """
    Helper function to generate alignment constraints

    Parameters
    ----------
    items: a list of rects to align.
    attr: which attribute to align.

    Returns
    -------
    cons: list of constraints describing the alignment
    """
    cons = []
    for i in items[1:]:
        cons.append((getattr(items[0], attr) == getattr(i, attr)) | strength)
    return cons


def stack(items, direction):
    """
    Helper generating constraints for stacking items in a direction.
    """
    constraints = []

    if direction == 'left':
        first_item, second_item = 'left', 'right'
    elif direction == 'right':
        first_item, second_item = 'right', 'left'
    elif direction == 'top':
        first_item, second_item = 'top', 'bottom'
    elif direction == 'bottom':
        first_item, second_item = 'bottom', 'top'

    for i in range(1, len(items)):
        c = getattr(items[i-1], first_item) <= getattr(items[i], second_item)
        constraints.append(c)
    return constraints

def hstack(items, padding=0):
    constraints = []
    for i in range(1, len(items)):
        constraints.append(items[i-1].right+padding <= items[i].left)
    return constraints

def vstack(items, padding=0):
    constraints = []
    for i in range(1, len(items)):
        constraints.append(items[i-1].bottom-padding >= items[i].top)
    return constraints

def vstacktight(items, padding=0):
    constraints = []
    for i in range(1, len(items)):
        constraints.append((items[i-1].bottom-padding == items[i].top)| 2.e8)
        print(constraints[-1])
    return constraints

def get_text_size(mpl_txt, renderer):
    bbox = mpl_txt.get_window_extent(renderer)
    return bbox.width, bbox.height

class SpaceContainer(Box):
    def __init__(self, parent, name):
        # parent is the axis usually
        super(SpaceContainer, self).__init__(parent, name)
        self.solver.suggestValue(self.pref_width, 1.)
        self.solver.suggestValue(self.pref_height, 1.)

    def place(self):
        print("Space!", self)

class TextContainer(Box):
    def __init__(self, parent, name):
        # parent is the axis usually
        super(TextContainer, self).__init__(parent, name)
        self.mpl_text = None
        self.figure = parent.figure

    def set_mpl_text(self, txt):
        print("TXT:",txt)
        self.mpl_text = txt
        txt.set_figure(self.parent.figure)
        self.parent.figure.texts.append(txt)
        renderer = self.figure.canvas.get_renderer()
        bbox = self.figure.transFigure.inverted().transform_bbox(
            txt.get_window_extent(renderer))

        print("Text_ex",self,bbox)
        self.solver.suggestValue(self.min_width, bbox.width)
        self.solver.suggestValue(self.min_height, bbox.height)

    def place(self):
        txt = self.mpl_text
        print("Text being placed",self)
        if txt is not None:
            txt.set_position((self.left.value(),
                              self.bottom.value()))


def contains(parent, child):
    c = [parent.left <= child.left,
         parent.bottom <= child.bottom,
         parent.right >= child.right,
         parent.top >= child.top,
        ]
    return c

class RawAxesContainer(Box):
    def __init__(self, parent, name):
        super(RawAxesContainer, self).__init__(parent, name)
        self.adjusted_axes_box = Box()

        # this axis is added at init.  The plotted, and then used to layout.
        # we need a re-calc step after draw..
        self.axes = parent.parent.figure.add_axes([0.15, 0.15, 0.45, 0.45], label=str(id(self)))
        # constrain to be inside parent?
        constraints = [self.left  >= self.parent.left,
                        self.right <= self.parent.right,
                        self.top   <= self.parent.top,
                        self.bottom >= self.parent.bottom]
        for c in constraints:
            self.solver.addConstraint(c)
        #print('Added Axes')
        #print(self.axes)

    def place(self):
        print("Placing: ")
        #        print (self.solver.dump())
        figure = self.parent.parent.figure
        invTransFig = figure.transFigure.inverted().transform_bbox
        # set the axes position based on the info in our box.  Need to
        # transform into relative co-ordinates.
        print(self)
        print(self.width.value())
        box = matplotlib.transforms.Bbox.from_bounds(*self.get_mpl_rect())
        print('Raw axis box',self.name, box)
        bbox = invTransFig(box)
        print('Raw axis bbox',self.name, bbox)
        print('figure extents',figure.canvas.get_width_height())

        # This is the rectangle given by out box.left, box.bottom etc.
        self.axes.set_position(box)

class AxesTickContainer(Box):
    def __init__(self, parent, name):
        super(AxesTickContainer, self).__init__(parent, name)

        self.raw_axes =  RawAxesContainer(self, name+'ax')
        self.parent =  parent
        self.renderer = parent.renderer

        self.solver.suggestValue(self.raw_axes.pref_width, 1.)
        self.solver.suggestValue(self.raw_axes.pref_height, 1.)
        self.update_constraints = []

    def update(self):
        # this only gets called once, but prob should be called numerous times.
        # ... .but, how do we get rid of old constraints?
        # We need to get how different the raw_axes bbox is from the position so we know what the offset is
        figure = self.parent.figure
        invTransFig = figure.transFigure.inverted().transform_bbox

        raa = self.raw_axes.axes
        abox = raa.get_position()
        tbox = invTransFig(raa.get_tightbbox(self.renderer))
        print('abox')
        print(abox)
        print('tbox')
        print(tbox)

        dl = (abox.x0-tbox.x0)
        dr = (tbox.x1-abox.x1)
        db = (abox.y0-tbox.y0)
        dt = (tbox.y1-abox.y1)
        print('Update dl,dr,db,dt',self.name,dl,dr,db,dt)
        # remove the old constraints if any
        for c in self.update_constraints:
            self.solver.removeConstraint(c)

        ## set constraints.
        self.update_constraints = [self.left +dl  == self.raw_axes.left,
                        self.right - dr == self.raw_axes.right,
                        self.top   == self.raw_axes.top+dt,
                        self.bottom +db == self.raw_axes.bottom,
                        self.left >= 0,
                        self.bottom >= 0]
        for c in self.update_constraints:
            self.solver.addConstraint(c)
        print("Updating AxesTick",self)

    def place(self):
        print('Axes Container:',self)
        # in here we need to get a bbox for the axis,
        self.raw_axes.place()

class AxesContainer(Box):
    def __init__(self, parent, name='ac'):
        super(AxesContainer, self).__init__(parent, name)
        self.children = []

        self.parent = parent
        self.figure = parent.figure
        self.renderer = parent.renderer
        self.solver = parent.solver

        self.axes_tick = AxesTickContainer(self, name+'at') # contains full plot w/o labels.
        self.top_title = TextContainer(self, name+'tt')
        self.left_label = TextContainer(self, name+'ll')
        self.right_label = TextContainer(self, name+'rl')
        self.top_label = TextContainer(self, name+'tl')
        self.bottom_label = TextContainer(self, name+'bl')
        # set up dummy ticklables

        self.children = [self.top_title, self.top_label, self.bottom_label,
                         self.left_label, self.right_label, self.axes_tick,
                         ]
        self.padding = Variable(name + '_padding')

        self.solver.addEditVariable(self.padding, 'strong')
        self.solver.suggestValue(self.padding, 0.005)

        constraints = vstacktight([ self.top_title,
                                self.top_label,
                              self.axes_tick, self.bottom_label],
                              padding=0.005)
        constraints += hstack([self.left_label, self.axes_tick,
                                self.right_label], padding=0.005)
        for c in constraints:
            print(c)
            self.solver.addConstraint(c)
        #constraints += [self.left_label.right  <= self.raw_axes.left]

        pad = self.padding
        # need to save these because we want to remove them and redo them
        # if we add another box in some direction.
        self.outer_left = self.left_label
        self.outer_right = self.right_label
        self.outer_bottom = self.bottom_label
        self.outer_top = self.top_title
        if 0:
            self.outer_constraints = [self.left + pad   <= self.left_label.left,
                        self.right - pad >= self.right_label.right,
                        self.top_title.top + pad <= self.top,
                        self.bottom + pad <= self.bottom_label.bottom,
                        self.left >= 0,
                        self.bottom >= 0]
        self.outer_constraints = [self.left + pad   <= self.outer_left.left,
                        self.right - pad >= self.outer_right.right,
                        self.outer_top.top + pad <= self.top,
                        self.bottom + pad <= self.outer_bottom.bottom,
                        self.left >= 0,
                        self.bottom >= 0]
        for c in self.outer_constraints:
            print(c)
            self.solver.addConstraint((c|1e5))

        if 1:
            constraints = align([self.top_title, self.top_label,
                                  self.axes_tick.raw_axes, self.bottom_label], 'h_center')
            constraints += align([self.left_label, self.axes_tick.raw_axes,
                                  self.right_label], 'v_center')

        for c in constraints:
            print(c)
            self.solver.addConstraint((c|1e5))

        self.solver.addConstraint((self.top_title.bottom == self.top_label.top))
        self.solver.addConstraint((self.bottom_label.top ==
                    self.axes_tick.bottom))

        # these end up setting the maximum size of the axis.  Not sure why we
        # want this to be that way, so set to big numbers.
        #        self.solver.suggestValue(self.axes_tick.raw_axes.pref_width, 100000)
        #        self.solver.suggestValue(self.axes_tick.raw_axes.pref_height, 100000)
        self.solver.suggestValue(self.axes_tick.pref_width, 1.)
        self.solver.suggestValue(self.axes_tick.pref_height, 1.00000)
        self.solver.updateVariables()

    def append_right(self, box):
        print(type(self.children[-1]))
        print(type(box))
        self.children += [box]
        # constraints on the leftside of box.
        # constraints = hstack([self.outer_right, box])
        constraints = [self.outer_right.right == box.left]
        constraints += align([self.outer_right, box],'v_center')
        for c in constraints:
            print(c)
            self.solver.addConstraint((c|1e5))
        # now box is the outer right
        self.outer_right = box

        # remove the old outer constraints....
        for c in self.outer_constraints:
            try:
                self.solver.removeConstraint(c)
            except:
                print('failed to remove',c)
        # redo the outer constraints.
        pad = 0.00
        self.outer_constraints = [self.left + pad   <= self.outer_left.left,
                        self.right - pad >= self.outer_right.right,
                        self.outer_top.top + pad <= self.top,
                        self.bottom + pad <= self.outer_bottom.bottom,
                        self.left >= 0,
                        self.bottom >= 0]
        for c in self.outer_constraints:
            print(c)
            self.solver.addConstraint((c|1e5))


    def add_label(self, text, where='bottom'):
        d = {'left': self.left_label,
             'right': self.right_label,
             'top': self.top_label,
             'bottom': self.bottom_label,
             'title': self.top_title
             }
        r = d[where]
        if where == 'title':
            fs = plt.rcParams['axes.titlesize']
        else:
            fs = plt.rcParams['xtick.labelsize']

        if where in ('left', 'right'):
            rotation = 'vertical'
            txt = plt.Text(0, 0, text=text, transform=self.figure.transFigure,
                            rotation=rotation,
                           va='bottom', ha='left', fontsize=fs)
        else:
            rotation = 'horizontal'
            txt = plt.Text(0, 0, text=text, transform=self.figure.transFigure,
                           rotation=rotation,
                           va='bottom', ha='left', fontsize=fs)

        r.set_mpl_text(txt)


    def do_layout(self):
        print("Do Layout")
        print(self)
        print(self.left_label)
        print(self.axes_tick.raw_axes)
        print(self.axes_tick)
        print(self.bottom_label)
        print(self.right_label)
        print("Do Layout")
        self.axes_tick.update()
        self.solver.updateVariables()

        for c in self.children:
            # children can either be AxesContainer (parasitic axis usually)
            # or children can be objects with a place method.
            if isinstance(c,AxesContainer):
                c.do_layout()
            else:
                c.place()


def find_renderer(fig):
    if hasattr(fig.canvas, "get_renderer"):
        renderer = fig.canvas.get_renderer()
    else:
        import io
        fig.canvas.print_pdf(io.BytesIO())
        renderer = fig._cachedRenderer
    return(renderer)


class FigureLayout(Box):
    def __init__(self, mpl_figure):
        Box.__init__(self, None, 'fl')
        self.children = []
        self.figure = mpl_figure
        self.renderer = find_renderer(mpl_figure)
        self.set_geometry(0, 0, 1.,1.)
        self.parent = None

    def grid_layout(self, size, hspace=0.1):
        width = self.width.value
        height = self.height.value
        rows, cols = size

        col_splits = [width/cols *i for i in range(cols)]
        row_splits = [height/rows * i for i in range(rows)]

if __name__ == '__main__':
    import numpy as np
    fig2 = plt.figure(0, dpi=100)
    ax = fig2.add_axes([0.,0.,1.,1.])
    ax.set_xticks(np.arange(0,1.,0.1))
    ax.set_yticks(np.arange(0,1.,0.1))
    ax.grid('on')
    print(fig2.canvas.get_width_height())
    fl = FigureLayout(fig2)
    ac1 = AxesContainer(fl,name='ac1')
    ac2 = AxesContainer(fl,name='ac2')
    ac3 = AxesContainer(fl,name='ac3')
    print(ac3.width)
    # make a colorbar for ac3 only...
    if 1:
        cb3 = AxesContainer(fl,name='cb3')
        # set the size of the actual colorbar inside this axis...
        fl.solver.addConstraint(cb3.axes_tick.raw_axes.width ==
                0.03*ac3.axes_tick.width)
        fl.solver.addConstraint(cb3.axes_tick.raw_axes.height ==
                0.6*ac3.axes_tick.raw_axes.height)
        # append this axis to the right of ac3:
        ac3.append_right(cb3)
        print(fl.solver.dump())
    #print(boo)

    gl = GridLayout(2, 2, fig2.bbox.width, fig2.bbox.height)

    #fl.solver.addConstraint(ac1.top_label.v_center == ac2.top_label.v_center)
    if 1:
        fl.solver.addConstraint(ac1.axes_tick.raw_axes.right ==
            ac3.axes_tick.raw_axes.right)
        fl.solver.addConstraint(ac1.axes_tick.raw_axes.left ==
            ac3.axes_tick.raw_axes.left)
    #fl.solver.addConstraint(ac1.raw_axes.left == ac3.raw_axes.left)
    #fl.solver.addConstraint(ac1.left_label.right == ac3.left_label.right)

    if 1:
        fl.solver.addConstraint(ac1.axes_tick.raw_axes.height ==
                            ac3.axes_tick.raw_axes.height)
        fl.solver.addConstraint(ac1.axes_tick.raw_axes.left ==
                            ac3.axes_tick.raw_axes.left)
        fl.solver.addConstraint(ac1.axes_tick.raw_axes.right ==
                                                ac3.axes_tick.raw_axes.right)
    if 0:
        fl.solver.addConstraint(ac2.axes_tick.raw_axes.width ==
                            ac3.axes_tick.raw_axes.width)
    if 0:
        # contraint to align bottom and tops...
        fl.solver.addConstraint(ac2.axes_tick.raw_axes.bottom ==
                        ac3.axes_tick.raw_axes.bottom)
        fl.solver.addConstraint(ac2.axes_tick.raw_axes.top ==
                        ac1.axes_tick.raw_axes.top)
    #fl.solver.addConstraint(ac2.raw_axes.top == ac1.raw_axes.top)
    # ac.add_label('title', 'title')
    #ac1.add_label('hallo', 'top')
    ac1.add_label('Abo\nYay', 'left')
    ac1.add_label('Abo', 'bottom')
    ac1.add_label('title1', 'title')

    if 1:
        ac2.add_label('sda\n ha ha h', 'bottom')
        ac2.add_label('title2', 'title')
        ac2.add_label('right', 'right')

        ac2.axes_tick.raw_axes.axes.plot(np.arange(0,10000,1000),
            np.arange(0,10000,1000))

        ac3.add_label('title3', 'title')
        ac3.add_label('Wavenumbers / [cm]', 'left')
        ac3.add_label('Ac3 bottom label', 'bottom')

    ac1.axes_tick.raw_axes.axes.xaxis.set_ticks_position('bottom')
    ac2.axes_tick.raw_axes.axes.yaxis.set_ticks_position('right')
    ac3.axes_tick.raw_axes.axes.yaxis.set_ticks_position('left')
    ac1.axes_tick.raw_axes.axes.plot([1000, 3000, 6000, 7000])

    pcm = ac3.axes_tick.raw_axes.axes.pcolormesh(np.random.rand(20,20)*500)
    fig2.colorbar(pcm,cax=cb3.axes_tick.raw_axes.axes)
    def do_lay(ev):
        print('Resize geom',fig2.canvas.get_width_height())
        gl.calc_borders(1.,1.)
        gl.place_rect(ac1, (0, 0))
        gl.place_rect(ac2, (0, 1), rowspan=2)
        gl.place_rect(ac3, (1, 0))
        for a in [ac1, ac2, ac3]:
            print(a)
            a.do_layout()
            # print(fl.solver.dump())

    do_lay(None)
    cid = fig2.canvas.mpl_connect('resize_event', do_lay)

    # print(ac)
    plt.show()
    #print('Starting Print')
    #matplotlib.use('PDF',warn=False, force=True)
    #import matplotlib.pyplot as plt
    do_lay(None)
    fig2.savefig('Example.png')
