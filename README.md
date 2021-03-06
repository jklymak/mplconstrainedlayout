# Toy Matplotlib layout manager

To run you need matplotlib and kiwi solver.  `pip install git+https://github.com/nucleic/kiwi.git` is needed to use the latest `python3` version.  

This is just lightly modified from https://github.com/Tillsten/MplLayouter who did the real work here.

See https://github.com/matplotlib/matplotlib/issues/1109

To run `python layout.py`

Status: its pretty messy, but does what I want.  Todo: clean up methods, and come up with clean way to link axes.

  - [x] Figure out how to add subsidiary axes like colorbars.  This is now done via a `append_right` method for the `AxesContainer` class, where whatever is appended to the right is put to the right and centered on the container vertically.  
     - [ ] Other directions
     - [ ] Optional align arguments (i.e. center, top, bottom, offset)

  - [ ] See if this procedure is robust on other backends.  So far it doesn't work at all on `nbagg` because it doesn't seem to do `figure.transFigure` properly.  (??)

Long term, turn into a MEP

<img width="645" alt="example" src="https://github.com/jklymak/mplconstrainedlayout/blob/master/Example.png?raw=true">
