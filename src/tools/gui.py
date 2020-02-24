import PySimpleGUI as sg      
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def show_figure(fig):

    figure_x, figure_y, figure_w, figure_h = fig.bbox.bounds
    
    # define the window layout
    layout = [[sg.Text('Add to training data?', font='Any 18')],
            [sg.Canvas(size=(figure_w, figure_h), key='canvas')],
            [   sg.Yes(pad=((400, 0), 3), size=(4, 2)), 
                sg.No(size=(4, 2)),
                sg.Button('Center', size=(6, 2)),
                sg.Button('incLon', size=(6, 2)),
                sg.Button('incLat', size=(6, 2)),
                sg.Button('decLon', size=(6, 2)),
                sg.Button('decLat', size=(6, 2))]]

    # create the form and show it without the plot
    window = sg.Window('Training data selecter',
                    layout, finalize=True)

    # add the plot to the window
    fig_canvas_agg = draw_figure(window['canvas'].TKCanvas, fig)

    event, values = window.read()

    window.close()

    return event, values
