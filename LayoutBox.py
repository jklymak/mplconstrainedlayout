# -*- coding: utf-8 -*-
"""

"""
from __future__ import division, print_function
import kiwisolver as kiwi
import numpy as np
import matplotlib.gridspec as gridspec



# plt.close('all')


class LayoutBox(object):
    """
    Basic rectangle representation using kiwi solver variables
    """



    def __init__(self, parent=None, name='', tight=False,
                 lower_left=(0, 0), upper_right=(1, 1)):
        Variable = kiwi.Variable
        self.parent = parent
        self.name = name
        sn = self.name + '_'
        if parent is None:
            self.solver = kiwi.Solver()
        else:
            self.solver = parent.solver
            parent.add_child(self)
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
        self.tight = tight
        self.add_constraints()
        self.children = []

    def add_child(self, child):
        self.children += [child]

    def remove_child(self, child):
        try:
            self.children.remove(child)
        except ValueError:
            print("Tried to remove child that doesn't belong to parent")

    def add_constraints(self):
        sol = self.solver
        # never let width and height go negative.
        for i in [self.min_width, self.min_height]:
            sol.addEditVariable(i, 1e9)
            sol.suggestValue(i, 0)
        # define relation ships between things thing width and right and left
        self.hard_constraints()
        self.soft_constraints()
        if self.parent:
            self.parent_constrain()
        sol.updateVariables()

    def parent_constrain(self):
        parent = self.parent
        hc = [self.left >= parent.left,
              self.bottom >= parent.bottom,
              self.top <= parent.top,
              self.right <= parent.right]
        for c in hc:
            self.solver.addConstraint(c)



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
        if self.tight:
            suggest = 0
        else:
            suggest = 1e10
        for i in [self.pref_width, self.pref_height]:
            sol.addEditVariable(i, 'strong')
            sol.suggestValue(i, suggest)
        c = [(self.pref_width == self.width),
             (self.pref_height == self.height)]
        for i in c:
            sol.addConstraint(i|0.000001)

    def set_geometry_soft(self, left, bottom, right, top, strength=1e9):
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

    def set_geometry(self, left, bottom, right, top, strength=1e9):
        hc = [self.left == left,
            self.right == right,
            self.bottom == bottom,
            self.top == top]
        for c in hc:
            self.solver.addConstraint(c)
        self.solver.updateVariables()

    def set_left_margin(self, margin):
        c = (self.left == self.parent.left + margin )
        self.solver.addConstraint(c)

    def set_left_margin_min(self, margin):
        c = (self.left >= self.parent.left + margin )
        self.solver.addConstraint(c)

    def set_right_margin(self, margin):
        c = (self.right == self.parent.right - margin )
        self.solver.addConstraint(c)

    def set_right_margin_min(self, margin):
        c = (self.right <= self.parent.right - margin )
        self.solver.addConstraint(c)

    def set_bottom_margin(self, margin):
        c = (self.bottom == self.parent.bottom + margin )
        self.solver.addConstraint(c)

    def set_bottom_margin_min(self, margin):
        c = (self.bottom >= self.parent.bottom + margin )
        self.solver.addConstraint(c)

    def set_top_margin(self, margin):
        c = (self.top == self.parent.top - margin )
        self.solver.addConstraint(c)

    def set_top_margin_min(self, margin):
        c = (self.top <= self.parent.top - margin )
        self.solver.addConstraint(c)

    def set_width_margins(self,margin):
        self.set_left_margin(margin)
        self.set_right_margin(margin)

    def set_height_margins(self,margin):
        self.set_top_margin(margin)
        self.set_bottom_margin(margin)

    def set_margins(self,margin):
        self.set_height_margin(margin)
        self.set_width_margin(margin)


    def get_rect(self):
        return (self.left.value(), self.bottom.value(),
                self.width.value(), self.height.value())

    def update_variables(self):
        '''
        Update *all* the variables that are part of the solver this LayoutBox
        is created with
        '''
        self.solver.updateVariables()

    def set_height(self,height):
        c = (self.height == height)
        self.solver.addConstraint(c)

    def set_width(self,width):
        c = (self.width == width)
        self.solver.addConstraint(c)

    def set_left(self,left):
        c = (self.left == left)
        self.solver.addConstraint(c)

    def set_bottom(self,bottom):
        c = (self.bottom == bottom)
        self.solver.addConstraint(c)

    def layout_from_gridspec(self, subspec, name=''):
        '''  Make a layout box from a subplotspec
        '''
        lb = LayoutBox(parent=self, name=name)
        gs = subspec.get_gridspec()
        nrows, ncols = gs.get_geometry()

        # from gridspec.  prob should be new method in gridspec
        left = 0.0
        right = 1.0
        bottom = 0.0
        top = 1.0
        totWidth = right-left
        totHeight = top-bottom
        hspace = 0.
        wspace = 0.

        # calculate accumulated heights of columns
        cellH = totHeight/(nrows + hspace*(nrows-1))
        sepH = hspace*cellH

        if gs._row_height_ratios is not None:
            netHeight = cellH * nrows
            tr = float(sum(gs._row_height_ratios))
            cellHeights = [netHeight*r/tr for r in gs._row_height_ratios]
        else:
            cellHeights = [cellH] * nrows

        sepHeights = [0] + ([sepH] * (nrows-1))
        cellHs = np.add.accumulate(np.ravel(list(zip(sepHeights, cellHeights))))

        # calculate accumulated widths of rows
        cellW = totWidth/(ncols + wspace*(ncols-1))
        sepW = wspace*cellW

        if gs._col_width_ratios is not None:
            netWidth = cellW * ncols
            tr = float(sum(gs._col_width_ratios))
            cellWidths = [netWidth*r/tr for r in gs._col_width_ratios]
        else:
            cellWidths = [cellW] * ncols

        sepWidths = [0] + ([sepW] * (ncols-1))
        cellWs = np.add.accumulate(np.ravel(list(zip(sepWidths, cellWidths))))

        figTops = [top - cellHs[2*rowNum] for rowNum in range(nrows)]
        figBottoms = [top - cellHs[2*rowNum+1] for rowNum in range(nrows)]
        figLefts = [left + cellWs[2*colNum] for colNum in range(ncols)]
        figRights = [left + cellWs[2*colNum+1] for colNum in range(ncols)]

        rowNum, colNum =  divmod(subspec.num1, ncols)
        figBottom = figBottoms[rowNum]
        figTop = figTops[rowNum]
        figLeft = figLefts[colNum]
        figRight = figRights[colNum]

        if subspec.num2 is not None:

            rowNum2, colNum2 =  divmod(subspec.num2, ncols)
            figBottom2 = figBottoms[rowNum2]
            figTop2 = figTops[rowNum2]
            figLeft2 = figLefts[colNum2]
            figRight2 = figRights[colNum2]

            figBottom = min(figBottom, figBottom2)
            figLeft = min(figLeft, figLeft2)
            figTop = max(figTop, figTop2)
            figRight = max(figRight, figRight2)

        lb.set_geometry(figLeft, figBottom, figRight, figTop)
        return lb

    def __repr__(self):
        args = (self.name, self.left.value(), self.bottom.value(),
                self.right.value(), self.top.value())
        return 'LayoutBox: %s, (left: %1.2f) (bot: %1.2f) (right: %1.2f) (top: %1.2f)'%args

def hstack(boxes, padding=0):
    '''
    Stack LayoutBox instances from left to right
    '''

    for i in range(1,len(boxes)):
        c = (boxes[i-1].right + padding <= boxes[i].left)
        boxes[i].solver.addConstraint(c)

def vstack(boxes, padding=0):
    '''
    Stack LayoutBox instances from top to bottom
    '''

    for i in range(1,len(boxes)):
        c = (boxes[i-1].bottom - padding >= boxes[i].top)
        boxes[i].solver.addConstraint(c)

def match_heights(boxes, height_ratios=None):
    '''
    Stack LayoutBox instances from top to bottom
    '''

    if height_ratios == None:
        height_ratios = np.ones(len(boxes))
    for i in range(1,len(boxes)):
        c = (boxes[i-1].height ==
                boxes[i].height*height_ratios[i-1]/height_ratios[i])
        boxes[i].solver.addConstraint(c)

def match_widths(boxes, width_ratios=None):
    '''
    Stack LayoutBox instances from top to bottom
    '''

    if width_ratios == None:
        width_ratios = np.ones(len(boxes))
    for i in range(1,len(boxes)):
        c = (boxes[i-1].width ==
                boxes[i].width*width_ratios[i-1]/width_ratios[i])
        boxes[i].solver.addConstraint(c)

def vstackeq(boxes, padding=0, height_ratios=None):
    vstack(boxes,padding=padding)
    match_heights(boxes, height_ratios=height_ratios)

def hstackeq(boxes, padding=0, width_ratios=None):
    hstack(boxes,padding=padding)
    match_widths(boxes, width_ratios=width_ratios)

def align(boxes, attr):
    cons = []
    for box in boxes[1:]:
        cons= (getattr(boxes[0], attr) == getattr(box, attr))
        boxes[0].solver.addConstraint(cons)

def match_left_margins(boxes):
    box0 = boxes[0]
    for box in boxes[1:]:
        c = (box0.left-box0.parent.left == box.left-box.parent.left)
        box0.solver.addConstraint(c)

def match_bottom_margins(boxes):
    box0 = boxes[0]
    for box in boxes[1:]:
        c = (box0.bottom-box0.parent.bottom == box.bottom-box.parent.bottom)
        box0.solver.addConstraint(c)

def match_right_margins(boxes):
    box0 = boxes[0]
    for box in boxes[1:]:
        c = (box0.right-box0.parent.right == box.right-box.parent.right)
        box0.solver.addConstraint(c)

def match_top_margins(boxes):
    box0 = boxes[0]
    for box in boxes[1:]:
        c = (box0.top-box0.parent.top == box.top-box.parent.top)
        box0.solver.addConstraint(c)

def match_width_margins(boxes):
    match_left_margins(boxes)
    match_right_margins(boxes)

def match_height_margins(boxes):
    match_top_margins(boxes)
    match_bottom_margins(boxes)

def match_margins(boxes):
    match_width_margins(boxes)
    match_height_margins(boxes)
