"""
A stripped-down main.py for the purpose of testing speed of DNMR plots
and improving animation.

Kernprof test conclusion:
99.9% of time is spent plotting
0.1% is spent doing math
Of plot time, about 71% is canvas.clear() and 29% is canvas.plot()
"""
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, \
    NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from tkinter import *



up_arrow = u"\u21e7"
down_arrow = u"\u21e9"


class VarButtonBox(Frame):
    """
    A deluxe VarBox that is closer to WINDNMR-style entry boxes.
    ent = entry that holds the value used for calculations
    increment = the amount added to or subtracted from ent by the buttons
    minus and plus buttons subtract/add once;
    up and down buttons repeat as long as button held down.
    Arguments:
    -text: appears above the entry box
    -default: default value in entry
    """

    # To do: use inheritance to avoid repeating code for different widgets
    def __init__(self, parent=None, name='', default=0.00, **options):
        Frame.__init__(self, parent, relief=RIDGE, borderwidth=1, **options)
        Label(self, text=name).pack(side=TOP)

        self.widgetName = name  # will be key in dictionary

        # Entries will be limited to numerical
        ent = Entry(self, width=7,
                    validate='key')  # check for number on keypress
        ent.pack(side=TOP, fill=X)
        self.value = StringVar()
        ent.config(textvariable=self.value)
        self.value.set(str(default))

        # Default behavior: both return and tab will shift focus to next
        # widget; only save data and ping model if a change is made
        # To-Do: consistent routines for VarBox, VarButtonBox, ArrayBox etc.
        # e.g. rename on_tab for general purpose on focus-out
        ent.bind('<Return>', lambda event: self.on_return(event))
        ent.bind('<Tab>', lambda event: self.on_tab())

        # check on each keypress if new result will be a number
        ent['validatecommand'] = (self.register(self.is_number), '%P')
        # sound 'bell' if bad keypress
        ent['invalidcommand'] = 'bell'

        # Create a grid for buttons and increment
        minus_plus_up = Frame(self)
        minus_plus_up.rowconfigure(0, minsize=30)  # make 2 rows ~same height
        minus_plus_up.columnconfigure(2, weight=1)  # lets arrow buttons fill
        minus_plus_up.pack(side=TOP, expand=Y, fill=X)

        minus = Button(minus_plus_up, text='-',
                       command=lambda: self.decrease())
        plus = Button(minus_plus_up, text='+',
                      command=lambda: self.increase())
        up = Button(minus_plus_up, text=up_arrow, command=lambda: None)
        up.bind('<Button-1>', lambda event: self.zoom_up())
        up.bind('<ButtonRelease-1>', lambda event: self.stop_action())

        self.mouse1 = False  # Flag used to check if left button held down

        minus.grid(row=0, column=0, sticky=NSEW)
        plus.grid(row=0, column=1, sticky=NSEW)
        up.grid(row=0, column=2, sticky=NSEW)

        # Increment is also limited to numerical entry
        increment = Entry(minus_plus_up, width=4, validate='key')
        increment.grid(row=1, column=0, columnspan=2, sticky=NSEW)
        self.inc = StringVar()
        increment.config(textvariable=self.inc)
        self.inc.set(str(1))  # 1 replaced by argument later?
        increment['validatecommand'] = (self.register(self.is_number), '%P')
        increment['invalidcommand'] = 'bell'

        down = Button(minus_plus_up, text=down_arrow, command=lambda: None)
        down.grid(row=1, column=2, sticky=NSEW)
        down.bind('<Button-1>', lambda event: self.zoom_down())
        down.bind('<ButtonRelease-1>', lambda event: self.stop_action())

    @staticmethod
    def is_number(entry):
        """
        tests to see if entry is acceptable (either empty, or able to be
        converted to a float.)
        """
        if not entry:
            return True  # Empty string: OK if entire entry deleted
        try:
            float(entry)
            return True
        except ValueError:
            return False

    def entry_is_changed(self):
        """True if current entry doesn't match stored entry"""
        return self.master.vars[self.widgetName] != float(self.value.get())

    def on_return(self, event):
        """Records change to entry, calls model, and focuses on next widget"""
        if self.entry_is_changed():
            self.to_dict()
            self.master.call_model()
        event.widget.tk_focusNext().focus()

    def on_tab(self):
        """Records change to entry, and calls model"""
        if self.entry_is_changed():
            self.to_dict()
            self.master.call_model()

    def to_dict(self):
        """
        Records widget's contents to the container's dictionary of
        values, filling the entry with 0.00 if it was empty.
        """
        if not self.value.get():  # if entry left blank,
            self.value.set(0.00)  # fill it with zero
        # Add the widget's status to the container's dictionary
        self.master.vars[self.widgetName] = float(self.value.get())

    def stop_action(self):
        """ButtonRelease esets self.mouse1 flag to False"""
        self.mouse1 = False

    def increase(self):
        """Increases ent by inc"""
        current = float(self.value.get())
        increment = float(self.inc.get())
        self.value.set(str(current + increment))
        self.on_tab()

    def decrease(self):
        """Decreases ent by inc"""
        current = float(self.value.get())
        decrement = float(self.inc.get())
        self.value.set(str(current - decrement))
        self.on_tab()

    def zoom_up(self):
        """Increases ent by int as long as button-1 held down"""
        increment = float(self.inc.get())
        self.mouse1 = True
        self.change_value(increment)

    def zoom_down(self):
        """Decreases ent by int as long as button-1 held down"""
        decrement = - float(self.inc.get())
        self.mouse1 = True
        self.change_value(decrement)

    def change_value(self, increment):
        """Adds increment to the value in ent"""
        if self.mouse1:
            self.value.set(str(float(self.value.get()) + increment))
            self.on_tab()  # store value, call model

            # Delay is set to 10, but really depends on model call time
            self.after(10, lambda: self.change_value(increment))


# def warw(bar): pass
    """
    Many of the models include Wa (width), Right-Hz, and WdthHz boxes.
    This function tacks these boxes onto a ToolBar.
    Input:
    -ToolBar that has been filled out
    Output:
    -frame with these three boxes and default values left-packed on end
    ***actually, this could be a function in the ToolBar class definition!
    """


class ToolBar(Frame):
    """
    A frame object that contains entry widgets, a dictionary for
    containing the values of children widgets, and a function to call the
    appropriate model.
    """

    def __init__(self, parent=None, **options):
        Frame.__init__(self, parent, **options)
        self.vars = {}

    def call_model(self):
        print('Sending to dummy_model: ', self.vars)


class DNMR_TwoSingletBar(ToolBar):
    """
    DNMR simulation for 2 uncoupled exchanging nuclei.
    -Va > Vb are the chemcial shifts (slow exchange limit)
    -ka is the a-->b rate constant (note: WINDNMR uses ka + kb here)
    -Wa, Wb are width at half height
    -pa is % of molecules in state a. Note for calculation need to /100 to
    convert to mol fraction.
    """
    def __init__(self, parent=None, **options):
        ToolBar.__init__(self, parent, **options)
        Va = VarButtonBox(self, name='Va', default=165.00)
        Vb = VarButtonBox(self, name='Vb', default=135.00)
        ka = VarButtonBox(self, name='ka', default=1.50)
        Wa = VarButtonBox(self, name='Wa', default=0.5)
        Wb = VarButtonBox(self, name='Wb', default=0.5)
        pa = VarButtonBox(self, name='%a', default=50)
        for widget in [Va, Vb, ka, Wa, Wb, pa]:
            widget.pack(side=LEFT)

        # initialize self.vars with toolbox defaults
        for child in self.winfo_children():
            child.to_dict()

        Button(self, text='Model 1', bg='blue',
               command=lambda: self.call_model1()).pack(side=LEFT)
        Button(self, text='Model 2', command=lambda: self.call_model2()).pack(
            side=LEFT)
        self.status = Label(root, text='Model 1')
        self.status.pack(side=TOP, expand=Y, fill=X)
        self.call_model = self.call_model1

    # @profile  # comment out when not running kernprof test
    def call_model1(self):
        self.status.config(text='Model 1')
        self.call_model = self.call_model1
        _Va = self.vars['Va']
        _Vb = self.vars['Vb']
        _ka = self.vars['ka']
        _Wa = self.vars['Wa']
        _Wb = self.vars['Wb']
        _pa = self.vars['%a'] / 100

        print('calling model 1')
        for i in range(10):
            x, y = plot1(_Va, _Vb, _ka, _Wa, _Wb, _pa)
            canvas.clear()
            canvas.plot(x, y)

    # @profile  # comment out when not running kernprof test
    def call_model2(self):
        self.status.config(text='Model 2')
        self.call_model = self.call_model2
        _Va = self.vars['Va']
        _Vb = self.vars['Vb']
        _ka = self.vars['ka']
        _Wa = self.vars['Wa']
        _Wb = self.vars['Wb']
        _pa = self.vars['%a'] / 100

        print('calling model 2')
        for i in range(10):
            x, y = plot2(_Va, _Vb, _ka, _Wa, _Wb, _pa)
            canvas.clear()
            canvas.plot(x, y)


class MPLgraph(FigureCanvasTkAgg):
    def __init__(self, f, master=None, **options):
        FigureCanvasTkAgg.__init__(self, f, master, **options)
        self.f = f
        self.a = f.add_subplot(111)
        self.a.invert_xaxis()
        self.show()
        self.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)
        self.toolbar = NavigationToolbar2TkAgg(self, master)
        self.toolbar.update()

    def plot(self, x, y):
        self.a.plot(x, y)
        self.f.canvas.draw()  # DRAW IS CRITICAL TO REFRESH

    def clear(self):
        self.a.clear()
        self.f.canvas.draw()


def plot1(va, vb, ka, Wa, Wb, pa):
    """
    plots the function nmrmath.dnmr_2spin
    Currently assumes va > vb
    """

    l_limit = vb - 50
    r_limit = va + 50
    x = np.linspace(l_limit, r_limit, 800)
    y = model1(x, va, vb, ka, Wa, Wb, pa)

    return x, y


def plot2(va, vb, ka, Wa, Wb, pa):
    """
    plots the function nmrmath.dnmr_2spin
    Currently assumes va > vb
    """

    l_limit = vb - 50
    r_limit = va + 50
    x = np.linspace(l_limit, r_limit, 800)

    model2 = model2maker(va, vb, ka, Wa, Wb, pa)
    y = model2(x)

    return x, y


def model1(v, va, vb, ka, Wa, Wb, pa):
    """
    A translation of the equation from Sandström's Dynamic NMR Spectroscopy,
    p. 14, for the uncoupled 2-site exchange simulation.
    v: frequency whose amplitude is to be calculated
    va, vb: frequencies of a and b singlets (slow exchange limit) (va > vb)
    ka: rate constant for state A--> state B
    pa: fraction of population in state Adv: frequency difference (va - vb)
    between a and b singlets (slow exchange)
    T2a, T2b: T2 (transverse relaxation time) for each nuclei
    returns: amplitude at frequency v
    """
    pi = np.pi
    pb = 1 - pa
    tau = pb / ka
    dv = va - vb
    Dv = (va + vb) / 2 - v
    T2a = 1 / (pi * Wa)
    T2b = 1 / (pi * Wb)

    P = tau * ((1 / (T2a * T2b)) - 4 * (pi ** 2) * (Dv ** 2) +
               (pi ** 2) * (dv ** 2))
    P += ((pa / T2a) + (pb / T2b))

    Q = tau * (2 * pi * Dv - pi * dv * (pa - pb))

    R = 2 * pi * Dv * (1 + tau * ((1 / T2a) + (1 / T2b)))
    R += pi * dv * tau * ((1 / T2b) - (1 / T2a)) + pi * dv * (pa - pb)

    I = (P * (1 + tau * ((pb / T2a) + (pa / T2b))) + Q * R) / (P ** 2 + R ** 2)
    return I


def model2maker(va, vb, ka, Wa, Wb, pa):
    """
    Attempt to create a function factory that creates tailored
    dnmr_2spin-like functions for greater speed.
    v: frequency whose amplitude is to be calculated
    va, vb: frequencies of a and b singlets (slow exchange limit) (va > vb)
    ka: rate constant for state A--> state B
    pa: fraction of population in state Adv: frequency difference (va - vb)
    between a and b singlets (slow exchange)
    T2a, T2b: T2 (transverse relaxation time) for each nuclei
    returns: amplitude at frequency v
    """
    pi = np.pi
    pi_squared = pi ** 2
    T2a = 1 / (pi * Wa)
    T2b = 1 / (pi * Wb)
    pb = 1 - pa
    tau = pb / ka
    dv = va - vb
    Dv = (va + vb) / 2
    P = tau * (1 / (T2a * T2b) + pi_squared * (dv ** 2)) + (pa / T2a + pb / T2b)
    p = 1 + tau * ((pb / T2a) + (pa / T2b))
    Q = tau * (- pi * dv * (pa - pb))
    R = pi * dv * tau * ((1 / T2b) - (1 / T2a)) + pi * dv * (pa - pb)
    r = 2 * pi * (1 + tau * ((1 / T2a) + (1 / T2b)))

    def model2(v):
        nonlocal Dv, P, Q, R
        Dv -= v
        P -= tau * 4 * pi_squared * (Dv ** 2)
        Q += tau * 2 * pi * Dv
        R += Dv * r
        return(P * p + Q * R) / (P ** 2 + R ** 2)
    return model2


root = Tk()
root.title('DNMR plot speed testing app')

twospinbar = DNMR_TwoSingletBar(root)
twospinbar.pack(side=TOP, expand=NO, fill=X)

f = Figure(figsize=(5, 4), dpi=100)
canvas = MPLgraph(f, root)
canvas._tkcanvas.pack(expand=YES, fill=BOTH)

root.mainloop()
