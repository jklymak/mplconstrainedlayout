# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import kiwisolver as kiwi
import matplotlib
#matplotlib.use('Qt5Agg')
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


def get_text_size(mpl_txt, renderer):
    bbox = mpl_txt.get_window_extent(renderer)
    return bbox.width, bbox.height

class TickContainer(Box):
    def __init__(self, parent, name):
        # parent is the axis
        # wth does this do?
        super(TickContainer, self).__init__(parent, name)

    def set_mpl_text(self, txt):
        self.mpl_text = txt
        txt.set_figure(self.parent.figure)
        self.parent.figure.texts.append(txt)
        text_ex = get_text_size(txt, self.parent.renderer)
        print('text_ex',text_ex)
        self.solver.suggestValue(self.min_width, text_ex[0]*2.)
        self.solver.suggestValue(self.min_height, text_ex[1]*2.)

    def place(self):
        txt = self.mpl_text
        if txt is not None:
            txt.set_position((self.left.value(),
                              self.bottom.value()))


class TextContainer(Box):
    def __init__(self, parent, name):
        # parent is the axis usually
        super(TextContainer, self).__init__(parent, name)
        self.mpl_text = None

    def set_mpl_text(self, txt):
        print("TXT:",txt)
        self.mpl_text = txt
        txt.set_figure(self.parent.figure)
        self.parent.figure.texts.append(txt)
        text_ex = get_text_size(txt, self.parent.renderer)
        print("Text_ex",text_ex)
        self.solver.suggestValue(self.min_width, text_ex[0]*2.)
        self.solver.suggestValue(self.min_height, text_ex[1]*2.)

    def place(self):
        txt = self.mpl_text
        print("Text:",self.name,self.left.value(),self.bottom.value(),self.right.value(),self.top.value())
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

        self.axes = parent.figure.add_axes([0, 0, 0.1, 0.1], label=str(id(self)))

    def place(self):
        print("Place raw axis!")
        figure = self.parent.figure
        renderer = self.parent.renderer
        invTransFig = figure.transFigure.inverted().transform_bbox

        # get the rectangle that solver wants us to have (from Box)
        # unfortunately, we already know the rectangle size at this point,
        # so getting the tickbox size is quite difficult.

        print(self)
        print(*self.get_mpl_rect())
        box = matplotlib.transforms.Bbox.from_bounds(*self.get_mpl_rect())

        bbox = invTransFig(box)
        print('bbox',bbox)
        # This is the rectangle given by out box.left, box.bottom etc.

        self.axes.set_position(bbox)

        if 0:
            tight_bbox = self.axes.get_tightbbox(renderer)
            tight_bbox = invTransFig(tight_bbox)
            dx = bbox.xmin-tight_bbox.xmin
            dx2 = bbox.xmax-tight_bbox.xmax
            dy = bbox.ymin-tight_bbox.ymin
            dy2 = bbox.ymax-tight_bbox.ymax
            new_size = (bbox.x0 + dx, bbox.y0 + dy,
                        bbox.width - dx + dx2, bbox.height - dy + dy2)
            print(new_size, self.axes.get_position())

            self.axes.set_position(new_size)
            #self.adjusted_axes_box.set_geometry(*new_size)
        #sol.suggestValue(self.adjusted_axes_box.min_width, new_size[2])

class AxesTickContainer(Box):
    def __init__(self, parent, name):
        super(RawAxesTickContainer, self).__init__(parent, name)

        self.axes = parent.figure.add_axes([0, 0, 0.1, 0.1], label=str(id(self)))

    def place(self):
        print("Place raw axis!")
        figure = self.parent.figure
        renderer = self.parent.renderer
        invTransFig = figure.transFigure.inverted().transform_bbox

        # get the rectangle that solver wants us to have (from Box)
        # unfortunately, we already know the rectangle size at this point,
        # so getting the tickbox size is quite difficult.

        print(self)
        print(*self.get_mpl_rect())
        box = matplotlib.transforms.Bbox.from_bounds(*self.get_mpl_rect())

        bbox = invTransFig(box)
        print('bbox',bbox)
        # This is the rectangle given by out box.left, box.bottom etc.

        self.axes.set_position(bbox)

        if 0:
            tight_bbox = self.axes.get_tightbbox(renderer)
            tight_bbox = invTransFig(tight_bbox)
            dx = bbox.xmin-tight_bbox.xmin
            dx2 = bbox.xmax-tight_bbox.xmax
            dy = bbox.ymin-tight_bbox.ymin
            dy2 = bbox.ymax-tight_bbox.ymax
            new_size = (bbox.x0 + dx, bbox.y0 + dy,
                        bbox.width - dx + dx2, bbox.height - dy + dy2)
            print(new_size, self.axes.get_position())

            self.axes.set_position(new_size)
            #self.adjusted_axes_box.set_geometry(*new_size)
        #sol.suggestValue(self.adjusted_axes_box.min_width, new_size[2])

class AxesContainer(Box):
    def __init__(self, parent, name='ac'):
        super(AxesContainer, self).__init__(parent, name)
        self.children = []

        self.parent = parent
        self.figure = parent.figure
        self.renderer = parent.renderer
        self.solver = parent.solver

        self.raw_axes = RawAxesContainer(self, 'ax')
        self.top_title = TextContainer(self, 'tt')
        self.left_label = TextContainer(self, 'll')
        self.right_label = TextContainer(self, 'rl')
        self.top_label = TextContainer(self, 'tl')
        self.bottom_label = TextContainer(self,'bl')
        self.left_ticks = TickContainer(self,'lk')
        self.bottom_ticks = TickContainer(self,'bk')
        # set up dummy ticklables
        txt = plt.Text(0, 0, text='0000', transform=None,
                       va='bottom', ha='left', fontsize=12)
        self.left_ticks.set_mpl_text(txt)
        self.bottom_ticks.set_mpl_text(txt)

        self.children = [self.top_title, self.top_label, self.bottom_label,
                         self.left_label, self.right_label, self.raw_axes,
                         self.left_ticks, self.bottom_ticks]
        self.padding = Variable(name + '_padding')

        self.solver.addEditVariable(self.padding, 'weak')
        self.solver.suggestValue(self.padding, 10)

        constraints = vstack([self.top_title, self.top_label,
                              self.raw_axes, self.bottom_ticks, self.bottom_label])
        constraints += hstack([self.left_label, self.left_ticks, self.raw_axes,  self.right_label])

        #constraints += [self.left_label.right  <= self.raw_axes.left]

        pad = self.padding
        constraints += [self.left + pad   <= self.left_label.left,
                        self.right - pad >= self.right_label.right,
                        self.top - pad >= self.top_title.top,
                        self.bottom + pad <= self.bottom_label.bottom,
                        self.left >= 0,
                        self.bottom >= 0]

        if 1:
            constraints += align([self.top_title, self.top_label,
                                  self.raw_axes, self.bottom_label], 'h_center')
            constraints += align([self.left_label, self.raw_axes,
                                  self.right_label], 'v_center')

        for c in constraints:
            self.solver.addConstraint(c)
        # these end up setting the maximum size of the axis.  Not sure why we
        # want this to be that way, so set to big numbers.
        self.solver.suggestValue(self.raw_axes.pref_width, 100000)
        self.solver.suggestValue(self.raw_axes.pref_height, 100000)
        self.solver.updateVariables()

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
            txt = plt.Text(0, 0, text=text, transform=None, rotation=rotation,
                           va='bottom', ha='left', fontsize=fs)
        else:
            rotation = 'horizontal'
            txt = plt.Text(0, 0, text=text, transform=None, rotation=rotation,
                           va='bottom', ha='left', fontsize=fs)

        r.set_mpl_text(txt)


    def do_layout(self):
        self.solver.updateVariables()
        print("Do Layout")
        print(self)
        print(self.left_label)
        print(self.raw_axes)
        print(self.right_label)
        print("Do Layout")

        for c in self.children:
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
        self.set_geometry(0, 0, mpl_figure.bbox.width, mpl_figure.bbox.height)
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
    ac1 = AxesContainer(fl)
    ac2 = AxesContainer(fl)
    ac3 = AxesContainer(fl)
    gl = GridLayout(2, 2, fig2.bbox.width, fig2.bbox.height)

    #fl.solver.addConstraint(ac1.top_label.v_center == ac2.top_label.v_center)
    #fl.solver.addConstraint(ac1.raw_axes.right == ac3.raw_axes.right)
    #fl.solver.addConstraint(ac1.raw_axes.left == ac3.raw_axes.left)
    #fl.solver.addConstraint(ac1.left_label.right == ac3.left_label.right)

    fl.solver.addConstraint(ac1.raw_axes.height == ac3.raw_axes.height)
    fl.solver.addConstraint(ac2.raw_axes.width == ac3.raw_axes.width)
    fl.solver.addConstraint(ac2.raw_axes.bottom == ac3.raw_axes.bottom)
    fl.solver.addConstraint(ac2.raw_axes.top == ac1.raw_axes.top)
    # ac.add_label('title', 'title')
    ac1.add_label('hallo', 'top')
    ac1.add_label('Abo', 'left')
    ac1.add_label('Abo', 'bottom')
    ac1.add_label('title1', 'title')

    ac2.add_label('sda', 'top')
    ac2.add_label('title2', 'title')
    ac2.add_label('left', 'left')

    ac3.add_label('title3', 'title')
    ac3.add_label('Wavenumbers / [cm]', 'left')
    ac3.add_label('Ac3 bottom label', 'bottom')

    ac1.raw_axes.axes.xaxis.set_ticks_position('top')
    ac1.raw_axes.axes.plot([1000, 50000, 100000, 1000000])

    ac2.raw_axes.axes.plot([1000, 50000, 100000, 1000000],[1000, 50000, 100000, 1000000])
    def do_lay(ev):
        print('Resize geom',fig2.canvas.get_width_height())
        gl.calc_borders(fig2.bbox.width, fig2.bbox.height)
        gl.place_rect(ac1, (0, 0))
        gl.place_rect(ac2, (0, 1), rowspan=2)
        gl.place_rect(ac3, (1, 0))
        for a in [ac1, ac2, ac3]:
            a.do_layout()

    do_lay(None)
    cid = fig2.canvas.mpl_connect('resize_event', do_lay)

    # print(ac)
    plt.show()
