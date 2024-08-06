#)##Jonathan Bostock
###Plotting Module for Python: jbplot
###Requirements: matplotlib, numpy, pandas

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.gridspec as grid_spec
import matplotlib.collections as mcollections
import matplotlib
from adjustText import adjust_text
import numpy as np
import pandas as pd
from icecream import ic

#Define our styles, these can be changed rapidly which is the point
okabe_ito = ["#56B4E9", #Light Blue
             "#D55E00", #Dark Reddish-orange
             "#009E73", #Teal
             "#F0E442", #Pale Yellow
             "#CC79A7", #Pink
             "#0072B2", #Dark Blue
             "#e69f00", #Orange
             "black","white"]

imperial = ["#0000CC", #Blue
            "#FE0000", #Red
            "#FFFF00", #Yellow
            "#00FF7F", #Mint
            "#ee82ef", #Pink
            "#ff8b00", #Orange
            "#7cee68", #Purple
            "black", "white"]

# Always do it
two_pi = 2*np.pi

#The colors variable (in american spelling) can be altered by users of the module if they want
colors = okabe_ito

lines = ["solid","dotted", "dashed", "dashdot",
        (5, (10,3)), (0, (3, 3, 1, 3, 1, 3)),(5, (10, 1, 3, 1)),(0, (1,3)),"none"]
markf = ["s", "D", ">", "<", "^", "v","d","*","none"]
markl = ["+", "x", "2", "1", "4", "3","|","_","none"]
markc = ["o"] * 8
marks = {"f":   markf,
         "l":    markl,
         "c":   markc}
#Table of marker sizes to make them look the same size
marksizes = {"s":50, "D":40, "^":50, "v": 50, ">":50, "<":50, "d":50, "*":50,
             "+":80, "x":50, "2":80, "1": 80, "4":80, "3":80, "|":80, "_":80,
             "o":50, "none": 0}

# Define our colormaps
    # Bilinear colormap for positive/negative values
bilin_list = ["#004C77","white","#D45E00"]
    # Linear colormap
lin_list = [(0,"#140A00"),(0.25,"#6A2F00"),(0.75,"#6A9E50"),(1,"#96D8FF")]
    # Second, backup colormap
lin_list_2 = [(0,"#000000"),(0.34,"#602a00"),(0.65,"#00ba87"),(1,"#ffc1e4")]
    # Cyclic colormap
cyclic_list = [(0,"#100a10"),(0.25,"#1c719f"),(0.5,"#cbc584"),
               (0.7,"#b34f00"),(1,"#100a10")]
    # AFM colormap
afm_list = [(0, "#000000"), (0.23,"#420E67"), (0.50, "#5252C4"),
            (0.75, "#7ECCCF"), (1, "#EEEEEE")]
    # Black/white colormap
bw_list = [(0, "#000000"), (1, "#FFFFFF")]
# Blue-orange colormap for complexj
bgw_list = [(0, "#000000"), (0.5, "#878787"), (1, "#ffffff")]
bgo_list = [(0, "#0080ff"), (0.5, "#878787"), (1, "#ff8000")]
wobb_list = [(0, "#FFFFFF"), (0.25, "#ff8000"), (0.5, "#000000"), (0.75, "#0080ff"), (1, "#FFFFFF")]
    # Cmy colormap
cmy_list = [(0, "#ff708f"), (1/7, "#ffde21"), (2/7, "#b3ff4c"),
            (3/7, "#46ffb9"), (4/7, "#24dbff"), (5/7, "#926dff"),
            (6/7, "#ff00ff"), (1, "#ff708f")]
    # Grey/grey colormap
grey_list = [(0, "#404040"), (1, "#BFBFBF")]
    # Okabe-Ito Fire colormap

fire_list = [(0,"#D55E00"), (1, "#F0E422")]
water_list = [(0, "#011e41"), (0.6, "#1eb9bc"), (1, "#b0ebdb")]

ice_list = [(0, "#c3478b"), (0.5, "#a072b6"), (1, "#7dc8f2")]

    # Auto colormaps
auto_colormap_lists = [[[0, "#000000"], [0.5, c], [1, "#FFFFFF"]] for c in colors]


# Convert colormap lists to colormaps
bilinear_colormap = mcolors.LinearSegmentedColormap.from_list("bilin",
                                                              bilin_list)
linear_colormap = mcolors.LinearSegmentedColormap.from_list("lin",
                                                            lin_list)
linear_colormap_2 = mcolors.LinearSegmentedColormap.from_list("lin2",
                                                            lin_list_2)
cyclic_colormap = mcolors.LinearSegmentedColormap.from_list("cyc",
                                                            cyclic_list)
afm_colormap = mcolors.LinearSegmentedColormap.from_list("afm",
                                                         afm_list)
bw_colormap = mcolors.LinearSegmentedColormap.from_list("bw",
                                                        bw_list)
cmy_colormap = mcolors.LinearSegmentedColormap.from_list("cmy",
                                                         cmy_list)
grey_colormap = mcolors.LinearSegmentedColormap.from_list("grey",
                                                          grey_list)
bgo_colormap = mcolors.LinearSegmentedColormap.from_list("bgo",
                                                         bgo_list)
bgw_colormap = mcolors.LinearSegmentedColormap.from_list("bgw",
                                                         bgw_list)
wobb_colormap = mcolors.LinearSegmentedColormap.from_list("wobb",
                                                          wobb_list)
line_grad_colormap = lambda x: matplotlib.colormaps["viridis"](x)
line_grad_colormap_1 = lambda x: matplotlib.colormaps["plasma"](0.9*x)
line_grad_colormap_2 = lambda x: matplotlib.colormaps["magma"](0.7*x + 0.2)
fire_colormap = mcolors.LinearSegmentedColormap.from_list("fire",
                                                          fire_list)
water_colormap = mcolors.LinearSegmentedColormap.from_list("water",
                                                           water_list)
ice_colormap = mcolors.LinearSegmentedColormap.from_list("ice",
                                                         ice_list)


# Make a dictionary of colormaps
colormap_dict = {"bilin":       bilinear_colormap,
                 "lin":         linear_colormap,
                 "lin2":        linear_colormap_2,
                 "line_grad":   [line_grad_colormap, line_grad_colormap_1, line_grad_colormap_2,
                                 fire_colormap, water_colormap, ice_colormap],
                 "cyc":         cyclic_colormap,
                 "afm":         afm_colormap,
                 "bw":          bw_colormap,
                 "bgw":         bgw_colormap,
                 "bgo":         bgo_colormap,
                 "wobb":        wobb_colormap,
                 "cmy":         cmy_colormap,
                 "grey":        grey_colormap,
                 "fire":        fire_colormap,
                 "water":       water_colormap,
                 "ice":         ice_colormap,
                 "auto":        lambda x, i: mcolors.LinearSegmentedColormap.from_list(
                     f"Auto",
                     auto_colormap_lists[i])(0.15 + x * 0.7)}



"""
Code calling order is:
        scatterset      --> scatter
        plotlineset     --> plotline
        scatter         --> plotline (sometimes)
        plotfunset      --> plotfun
        plot2dfun       --> plotfun
                ^haha currying go brrrrr
        plotfun         --> plotline
        plotdf          --> scatterset and/or plotlineset
        violinplot      --> ridgelinedf
        rdigelinedf     --> plotline

It all comes back to plotline in the end.

    "Do you see those times when there was only one function putting anything
    onto the axes? It was then that I carried the whole module"
        - jbplot.plotline
"""

# This is our other workhorse (although like any good function in this package
# it still calls plotline)
# Might add x_sig_vectors in the future (but these are kinda extra tbh)
def scatter(axis, x_vect, y_vect, y_sig_vect=None,
            colorcode=0,linecode=-1,marktype = "f",
            markcode=0,annotations=None,
            ann_y_offset=0,
            ann_x_offset=0,
            color_override = None,
            **kwargs):

    if color_override == None:
        color_to_plot = colors[colorcode]
    else:
        color_to_plot = color_override

    if type(y_sig_vect) != type(None):

        axis.errorbar(x_vect, y_vect, y_sig_vect,
                      color=color_to_plot,
                      capsize=4,zorder=1,
                      marker="none",linestyle="none")

    mark = marks[marktype][markcode]
    marksize = marksizes[mark]

    match marktype:
        case "f":
            axis.scatter(x_vect, y_vect,
                         color=color_to_plot,
                         edgecolors="black",
                         marker=mark,
                         s=marksize,
                         zorder=2,
                         **kwargs)
        case "l":
            axis.scatter(x_vect, y_vect,
                         color="black",
                         marker=mark,
                         s=marksize,
                         zorder=2,
                         **kwargs)
        case "c":
            axis.scatter(x_vect, y_vect,
                         edgecolors="black",
                         color=color_to_plot,
                         marker=mark,
                         s=marksize,
                         zorder=2,
                         **kwargs)

    #Note! this plotline does NOT take the kwargs. This might be a problem but
    #I don't really care yet since it doesn't come up very often
    #We feed it a color_override because scatter has already
    #done the work of calculating the color. This reduces dependencies
    if linecode != -1:
        plotline(axis, x_vect, y_vect,
                 color_override = color_to_plot,
                 linecode=linecode)

    if annotations is not None:

        for (x, y, label) in zip(x_vect, y_vect, annotations):
            axis.annotate(label, (x+ann_x_offset,y+ann_y_offset))

#The workhorse of the module
#The mighty plotline
def plotline(axis, x_vect, y_vect, y_sig_vect=None,
             colorcode=0,
             linecode=0, color_override = None,
             linecode_override = None,
             **kwargs):

    if color_override == None:
        color_to_plot = colors[colorcode]
    else:
        color_to_plot = color_override

    if linecode_override == None:
        linecode_to_plot = linecode
    else:
        linecode_to_plot = linecode_override

    if y_sig_vect != None:
        y_min_vect = [(y - ys) for y, ys in zip(y_vect, y_sig_vect)]
        y_max_vect = [(y + ys) for y, ys in zip(y_vect, y_sig_vect)]

        axis.fill_between(x_vect,
                          y_min_vect,
                          y_max_vect,
                          zorder=-1,
                          alpha=0.2,
                          linewidth=0,
                          color=color_to_plot)

    axis.plot(x_vect, y_vect,marker="none",
              zorder=0,
              linestyle=lines[linecode_to_plot],
              color=color_to_plot,**kwargs)

# A complex version of plotline
def complex_plotline(axis, x_vect, y_vect, fill=True):

    x_vect_doubled = [x for x in x_vect for i in range(2)]
    y_vect_doubled = [y for y in y_vect for i in range(2)]

    y_abs_vect = np.abs(y_vect_doubled)
    y_theta_norm = np.divide(np.add(np.angle(y_vect_doubled), np.pi), two_pi)

    x_coords = np.average([x_vect_doubled[:-1], x_vect_doubled[1:]], axis=0)
    y_coords = np.average([y_abs_vect[:-1], y_abs_vect[1:]], axis=0)

    points = np.array([x_coords, y_coords]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-1],points[1:]], axis=1)

    line_collection = mcollections.LineCollection(segments, cmap=wobb_colormap)
    line_collection.set_array(y_theta_norm[1:-1])
    axis.add_collection(line_collection)


def complex_scatter(axis, x_vect, y_vect, marktype = "f",
                    markcode = 0, marksize=10, line=False,
                    linewidth=1, phasebar = True,
                    **kwargs):

    y_abs_vect = np.abs(y_vect)
    y_theta_norm = np.divide(np.add(np.angle(y_vect), np.pi), two_pi)
    color_vect = [wobb_colormap(y_theta)[0:3] for y_theta in y_theta_norm]

    mark = marks[marktype][markcode]

    if line:
        plotline(axis, x_vect, y_abs_vect, colorcode=-2, linecode=0, linewidth=linewidth)

    if marktype == "f":
        axis.scatter(x_vect, y_abs_vect,
                     color=color_vect,
                     edgecolors="black",
                     marker=mark,
                     s=marksize,
                     zorder=2,
                     linewidth=linewidth,
                     **kwargs)
    else:
        axis.scatter(x_vect, y_abs_vect,
                     color=color_vect,
                     marker=mark,
                     s=marksize,
                     zorder=2,
                     **kwargs)

    if phasebar:

        heatmap_norm = mcolors.Normalize(vmin=0, vmax=two_pi)
        scalar_mappable = cm.ScalarMappable(norm=heatmap_norm,
                                            cmap = wobb_colormap)
        colorbar = axis.figure.colorbar(scalar_mappable, ax=axis,
                                        ticks=[0, np.pi, two_pi])
        colorbar.ax.set_yticklabels(["0", "$\pi$", "$2\pi$"])


def figax():

    fig, ax = plt.subplots(1,1,figsize=[4,4])
    return fig, ax

#A function to get the right color when handling gradients
def gradient_handler(gradient, gradient_vals, i, length,
                     color_override, gradient_code=0, auto_gradient=False):

    if gradient==True:

        if gradient_vals == None:
            s = 1- float(i)/(length-1)

        else:
            s = (gradient_vals[i]-min(gradient_vals))
            s /= (max(gradient_vals)-min(gradient_vals))

        if auto_gradient:
            return colormap_dict["auto"](s, gradient_code)
        else:
            return colormap_dict["line_grad"][gradient_code](s)

    else:
        return color_override

#This is the major timesaver function
#Takes all your data lists and plots them, you can run it with just three
#parameters but it also has plenty of customizability.
#I've tried to make the default options "nice"
def scatterset(axis, x_vect_set, y_vect_set,
               y_sig_vect_set=None,
               marktype="f",
               line=False,
               linecode_override=-1,
               type_start=0,
               name_list=None,
               gradient=False,
               gradient_vals=None,
               gradient_code=0,
               third_vars=None,
               color_override=None,
               markcode_max=None,
               **kwargs):

    if markcode_max is None:
        markcode_max = len(marks[marktype])

    if third_vars is not None:
        split_set = list(dict.fromkeys(gradient_vals))
        split_list = gradient_vals

    for i in range(len(y_vect_set)):

        j = i + type_start

        y_vect = y_vect_set[i]

        if len(x_vect_set) == 1: x_vect = x_vect_set[0]
        else: x_vect = x_vect_set[i]

        if line:
            if linecode_override != -1:
                linecode = linecode_override
            else:
                linecode = j
        else: linecode = -1

        #Handle y_sig vect sets
        if y_sig_vect_set == None: y_sig_vect = None
        else: y_sig_vect = y_sig_vect_set[i]

        #Decide whether to give it the scatter a name
        if name_list == None: name = None
        else: name = name_list[i]

        color_override = gradient_handler(gradient,
                                          gradient_vals,
                                          i,
                                          len(y_vect_set),
                                          color_override,
                                          gradient_code=gradient_code,
                                          auto_gradient=False)

        scatter(axis, x_vect, y_vect, y_sig_vect=y_sig_vect,
                colorcode=j,linecode=linecode,
                marktype=marktype,markcode=j%markcode_max,
                label=name,
                color_override=color_override,
                **kwargs)

#This plots a bunch of lines, otherwise behaves like scatterset.
#There's fewer options here, since scatterset can be made to plot lines if you want.
def plotlineset(axis, x_vect_set, y_vect_set,
                y_sig_vect_set=None,
                type_start=0,
                gradient=False,
                gradient_vals=None,
                gradient_code=0,
                third_vars=None,
                name_list=None,
                markcode_max=None,
                color_override=None,
                **kwargs):

    y_sig_vect=None
    name=None

    for i, y_vect in enumerate(y_vect_set):
        j = i + type_start

        if len(x_vect_set) == 1: x_vect = x_vect_set[0]
        else: x_vect = x_vect_set[i]

        #Calculate a gradient if necessary
        color_override = gradient_handler(gradient,
                                          gradient_vals,
                                          i,
                                          len(y_vect_set),
                                          color_override,
                                          gradient_code=gradient_code)

        if gradient != False:
            j = 0

        if name_list != None:
            name = name_list[i]

        if y_sig_vect_set != None:
            y_sig_vect = y_sig_vect_set[i]

        plotline(axis, x_vect, y_vect,
                 colorcode=j, linecode=j,
                 color_override=color_override,
                 y_sig_vect=y_sig_vect,
                 label=name,
                 **kwargs)

#This inputs a function and plots it across a given range
def plotfun(axis, function,
            sig_function=None,
            x_min=0, x_max=1,
            log=False, n=100.0,
            **kwargs):

    y_sig_vals=None
    #Generate our x values
    if log == False:
        step =(x_max-x_min)/n
        x_vals = np.arange(x_min, x_max, step)
    else:
        l_min = np.log10(x_min)
        l_max = np.log10(x_max)
        l_step = (l_max-l_min)/n
        x_vals = [
            np.power(10,z) for z in np.arange(l_min,
                                              l_max,
                                              l_step)]

    #Apply our function (this is slow but it's python, who cares?)
    y_vals = [function(x) for x in x_vals]
    if sig_function != None:
        y_sig_vals = [sig_function(x) for x in x_vals]

    #Pass everything to plotline
    #The kwargs flow smoothly through plotfun without
    #needing to be processed here.
    plotline(axis, x_vals, y_vals,
             y_sig_vect = y_sig_vals,
             **kwargs)

#Behaves just like you'd expect if you know what plotline, plotlineset, and plotfun do
def plotfunset(axis, function_set,
               sig_function_set=None,
               **kwargs):

    sig_function = None

    for i, function in enumerate(function_set):

        if sig_function_set != None:
            sig_function = sig_function_set[i]

        plotfun(axis, function,
                colorcode=i,
                linecode=i,
                **kwargs)

#Plots a two-argument function against a list for the second var
#This can be useful when your functions are kinda weird
def plot2dfun(axis, function,
              second_var,
              sig_function = None,
              gradient=False,
              name_list=None,
              gradient_code=0,
              **kwargs):

    color_override = None
    sig_function_curried = None
    name = None


    for i, var in enumerate(second_var):

        if gradient==True:
            color_override = gradient_handler(gradient,
                                              second_var,
                                              i,
                                              len(second_var),
                                              color_override,
                                              gradient_code=gradient_code)

        if sig_function != None:
            sig_function_curried = lambda x: sig_function(x, var)

        if name_list != None:
            name = name_list[i]

        plotfun(axis, lambda x: function(x, var),
                sig_function = sig_function_curried,
                colorcode=i,
                linecode=0,
                color_override = color_override,
                label=name,
                **kwargs)


#Extremely handy function for plotting from dataframes
#You can really abuse this thing with what you pass it and it handles most things
#Long form, short form, multiple X, if you really want you don't even need to give it anything but an axis and a dataframe
def plotdf(axis, df,
           x="x", y="y",
           y_sig=None, split=None,
           third_var=None,
           gradient=False,
           sort_x=False,
           name=None,
           name_map=None,
           gradient_code=0,
           name_list=None,
           plot_type="scatter",
           gradient_map = lambda x: x,
           assemble_legend = False,
           third_var_name_map=None,
           gradient_list = None,
           marktype="f",
           annotations=None,
           ann_y_offset=0,
           ann_x_offset=0,
           **kwargs):

    markcode_max=None
    double_split = False
    y_sig_vect_set =None
    split_list=None
    third_var_list=None

    if sort_x:
        df =df.sort_values(x)

    #If we get passed a column for sample names, then make a list of them
    if name == "y":
        name_list = y
    elif name is not None:
        if name_map is None:
            name_map = lambda x: f"{name} = {x}"

        name_list = [name_map(n) for n in df[name].drop_duplicates()]

    #If we're not splitting by variable, check if x is passed as a list
    if split is None:
        if isinstance(x, list):
            x_vect_set = [df[x_inst] for x_inst in x]
        else:
            x_vect_set = [df[x].tolist()]
        #Repeat for y
        if isinstance(y, list):
            y_vect_set = [df[y_inst] for y_inst in y]
        else:
            y_vect_set = [df[y].tolist()]

        if y_sig is not None:
            #Must also check for lists here
            if isinstance(y_sig, list):
                y_sig_vect_set = [df[y_sig_inst].tolist() for y_sig_inst in y_sig]
            else:
                y_sig_vect_set = [df[y_sig].tolist()]

    # If split is just a string, do things normally
    if isinstance(split, str):
        split_list = df[split].drop_duplicates().tolist()

    if isinstance(third_var, str):
        third_var_list = df[third_var].drop_duplicates().tolist()
        markcode_max = len(third_var_list)

    # Awful hacky variable renaming
    if isinstance(split_list, list) and isinstance(third_var_list, list):

        second_var_list = split_list.copy()
        split_list = []
        for s in second_var_list:
            for t in third_var_list:
                split_list.append((s,t))

        double_split = True

    # Bad bad bad
    if double_split:
        x_vect_set = []
        y_vect_set = []
        third_var_list = []
        for s in split_list:
            df_split = df.loc[df[split]==s[0]]
            x_vect_set.append(df_split.loc[df_split[third_var]==s[1]][x].tolist())
            y_vect_set.append(df_split.loc[df_split[third_var]==s[1]][y].tolist())
            third_var_list.append(s[1])

    elif isinstance(split, str):
        x_vect_set = [df.loc[df[split]==s][x].tolist() for s in split_list]
        y_vect_set = [df.loc[df[split]==s][y].tolist() for s in split_list]

        if y_sig != None:
            y_sig_vect_set = [df.loc[df[split]==s][y_sig].tolist() for s in split_list]

    # Handle the gradient if a map is needed
    if gradient and gradient_list is None:
        if type(y) == list:
            gradient_list = [gradient_map(y_item) for y_item in y]
        elif double_split:
            gradient_list = [gradient_map(s[0]) for s in split_list]
        else:
            gradient_list = [gradient_map(s_item) for s_item in split_list]

    # Generate a cool legend
    if assemble_legend:
        if double_split:
            if third_var_name_map is None:
                third_var_name_map = lambda x: f"{third_var} = {x}"

            for i, t in enumerate(list(dict.fromkeys(third_var_list))):
                scatter(axis, [],[], label=third_var_name_map(t),color_override="#808080",
                        markcode=i)

            if name_map is None:
                name_map = lambda x: f"{split} = {x}"

            scatterset(axis, [[]] * len(second_var_list), [[]] * len(second_var_list),
                       gradient=gradient, gradient_vals = second_var_list,
                       gradient_code=gradient_code,
                       marktype="c",
                       name_list=[name_map(s) for s in second_var_list])

        nice_legend(axis)
        axis.set_xlabel(x)
        axis.set_ylabel(y)

    if annotations is not None:
        if type(annotations) != list:
            annotations = list(df[annotations])

    # Call scatterset or plotlineset
    if "scatter" in plot_type:
        scatterset(axis, x_vect_set, y_vect_set,
                   y_sig_vect_set=y_sig_vect_set,
                   gradient=gradient,
                   gradient_vals=gradient_list,
                   gradient_code=gradient_code,
                   name_list=name_list,
                   third_vars=third_var_list,
                   markcode_max=markcode_max,
                   marktype=marktype,
                   annotations=annotations,
                   ann_y_offset=ann_y_offset,
                   ann_x_offset=ann_x_offset,
                   **kwargs)
    if "line" in plot_type:
        plotlineset(axis, x_vect_set, y_vect_set,
                    y_sig_vect_set = y_sig_vect_set,
                    gradient=gradient,
                    gradient_vals=gradient_list,
                    gradient_code=gradient_code,
                    name_list=name_list,
                    **kwargs)

# Function which plots a heatmap
def heatmap(axis, matrix, colormap="viridis",
            gradient_code = 0,
            legend=True,val_range = None):

    if val_range == None:
        val_min = np.min(matrix)
        val_max = np.max(matrix)
    else:
        val_min = val_range[0]
        val_max = val_range[1]

    if legend == True:

        heatmap_norm = mcolors.Normalize(vmin=val_min, vmax=val_max)
        scalar_mappable = cm.ScalarMappable(norm=heatmap_norm,
                                            cmap = colormap)

        axis.figure.colorbar(scalar_mappable, ax=axis)

    axis.imshow(matrix, cmap=colormap, vmin=val_min, vmax = val_max)

# Complex heatmap coz i'm smart and cool
# Doesnt work for the colorblind sadly
# No way I can work out to transform a 2d plane where coordinates are X = blue intensity and Y = yellow intensity
# into sensible polar coordinates, so the colorblind unfortunatley get griddied on by topology
# Sorry Lorenzo

# Edit it now works! It's kinda janky though
# white = +ve real, black = -ve real, blue = -ve imaginary, orange = +ve imaginary

def get_complex_color(complex_value, scale=1):

    theta = np.angle(complex_value)
    real_value = (np.tanh(np.real(complex_value / scale)) + 1)/2
    imag_value = (np.tanh(np.imag(complex_value / scale)) + 1)/2

    real_color = np.array(bgw_colormap(real_value))[0:3] 
    imag_color = np.array(bgo_colormap(imag_value))[0:3]

    final_color = list(real_color * np.cos(theta)**2 + imag_color * np.sin(theta)**2)

    return final_color

def complex_heatmap(axis, complex_number_array, intensity_transform = lambda x: x,
                    phasebar=True):
    """
    theta_array = np.divide(np.add(np.angle(complex_number_array), np.pi), two_pi)
    magnitude_array = intensity_transform(np.abs(complex_number_array))
    normalized_magnitude_array = np.divide(magnitude_array, np.max(magnitude_array))
    """

    """
    color_array = [[np.array(cmy_colormap(theta)[0:3]) * normalized_magnitude
                    for normalized_magnitude, theta in zip(normalized_magnitude_list, theta_list)]
                   for normalized_magnitude_list, theta_list in zip(normalized_magnitude_array, theta_array)]
    """

    square_mag_list = np.power(np.abs(complex_number_array.flatten()), 2)
    scale = np.sqrt(np.mean(square_mag_list)) * 1.5

    color_array = [[get_complex_color(c, scale=scale)
                    for c in complex_number_list]
                   for complex_number_list in complex_number_array]

    axis.imshow(color_array)

    if phasebar:

        heatmap_norm = mcolors.Normalize(vmin=0, vmax=two_pi)
        scalar_mappable = cm.ScalarMappable(norm=heatmap_norm,
                                            cmap = wobb_colormap)
        colorbar = axis.figure.colorbar(scalar_mappable, ax=axis,
                                        ticks=[0, np.pi, two_pi])
        colorbar.ax.set_yticklabels(["0", "$\pi$", "$2\pi$"])


# Bar chart time
def barchart(axis, values, name_list=[], sigma_list=[],
             groups=1, label_rotation=0):


    barchart_colors = []
    for color in colors:
        for i in range(groups):
            brightness_adjust = 0.35*(i-(groups-1)/2)/(groups+1)

            old_color_hsv = matplotlib.colors.rgb_to_hsv(
                matplotlib.colors.to_rgb(color))

            new_brightness = old_color_hsv[2] + brightness_adjust
            new_brightness = max(new_brightness, 0)
            new_brightness = min(new_brightness, 1)

            new_color_hsv = [old_color_hsv[0],
                             old_color_hsv[1],
                             new_brightness]
            new_color = matplotlib.colors.hsv_to_rgb(new_color_hsv)

            barchart_colors.append(new_color)

    values_list = values

    if type(values) == dict:
        name_list = [k for k in values.keys()]
        values_list = [v for v in values.values()]
        if type(values_list[0]) == tuple:
            values_list = [v[0] for v in values.values()]
            sigma_list = [v[1] for v in values.values()]

    if sigma_list is []:
        axis.bar(name_list, values_list,
                 color=barchart_colors)
    else:
        axis.bar(name_list, values_list,
                 color=barchart_colors,
                 yerr=sigma_list, capsize=5)


    axis.spines["top"].set_position(("data", 0))
    remove_box(axis, lines=["bottom", "right"])

    axis.tick_params(axis="x", length=0)


    #This line of code throws a "warning" but who cares it works
    axis.set_xticks([x + 0.5 for x in range(len(name_list))])
    axis.set_xticklabels(name_list, rotation=label_rotation, ha="right")

# Do I really need to comment what this function does?
def barchart_label(axis, text, start, end, height, text_raise=0):

    props = {"connectionstyle": "bar",
             "arrowstyle":      "-",
             "shrinkA":         20,
             "shrinkB":         20,
             "linewidth":       2}

    axis.annotate(text, xy=((start+end)/2, height+text_raise), zorder=10,
                  horizontalalignment="center",
                  verticalalignment="bottom")
    axis.annotate("", xy=(start,height),xytext=(end,height),
                  arrowprops=props)


# Useful and quick save function
# dpi is kinda high but it be like that sometimes
# Memory is like 30 £/TB these days who cares
def save(figure, name, file_types=["svg","png"], dpi=600, keep_box=False):

    if keep_box == False:
        for axis in figure.axes:
            remove_box(axis)

    for file_type in file_types:
        file_name = name + "." + file_type

        figure.savefig(file_name,
                       dpi=dpi,
                       format=file_type,
                       bbox_inches="tight")

#Gets rid of that box so your colleagues don't make fun of you for having a box
def remove_box(axis, lines = ["top","right"]):

    for line in lines:
        axis.spines[line].set_visible(False)

# Makes a nice legend
def nice_legend(axis):

    axis.legend(loc="center left", bbox_to_anchor = (1, 0.5),
                frameon=False)

#Recursively flattens a list of lists into a list
def flatten(possible_list):
    if hasattr(possible_list, "__itera__"):
        return [flatten(i) for i in possible_list]
    else:
        return possible_list

# Does a ridgeline plot
def ridgelinedf(figure,
                data_frame,
                x = ["x"],
                y = ["y"],
                gradient = False,
                colorcodes = None,
                linecodes = None,
                x_scale = None,
                plots_per_axis = 1,
                overlap = 0.5,
                axis_names = None,
                legend_names = None,
                x_axis_name = None,
                gradient_code=0):

    x_list = [data_frame.loc[:,x_item] for x_item in x]
    y_list = [data_frame.loc[:,y_item] for y_item in y]

    plots = len(y_list)

    if len(x_list) == 1:
        x_list = [x_list[0]] * plots

    gs = grid_spec.GridSpec(int(plots/ plots_per_axis), 1)
    gs.update(hspace = -overlap)

    ax_objs = []

    for i in range(plots):

        # Genius little hack
        if i % plots_per_axis == 0:
            j = int(i / plots_per_axis)
            ax_objs.append(figure.add_subplot(gs[j:j+1, 0:]))

        if x_scale != None:
            ax_objs[-1].set_xscale(x_scale)

        # Calculate data for color and line
        if colorcodes == None:
            colorcode = 0
        else:
            colorcode = colorcodes[i]

        if linecodes == None:
            linecode = 0
        else:
            linecode = linecodes[i]

        #Calculate a gradient if necessary
        color_override = gradient_handler(gradient,
                                          None,
                                          j,
                                          len(y),
                                          None,
                                          gradient_code=gradient_code)

        # Plot the line
        plotline(ax_objs[-1],
                 x_list[i],
                 y_list[i],
                 colorcode = colorcode,
                 linecode = linecode,
                 color_override = color_override)

        # Make background transparent 
        rect = ax_objs[-1].patch
        rect.set_alpha(0)

        # Remove y ticks
        ax_objs[-1].set_yticklabels([])
        ax_objs[-1].tick_params(left=False)

        # Remove the x ticks
        ax_objs[-1].set_xticklabels([], minor =True)
        ax_objs[-1].set_xticklabels([])
        ax_objs[-1].set_xticks([], minor = True)
        ax_objs[-1].set_xticks([])

    # Format all but the last axis
    for ax_obj in ax_objs[:-1]:
        # Remove the bottom line
        remove_box(ax_obj, lines = ["bottom"])

        # Share x axis allegedly
        ax_obj.get_shared_x_axes().join(ax_obj, ax_objs[-1])

    #Handle legend names and axis names
    if legend_names != None:
        ax_objs[0].legend(legend_names, frameon=False)

    if axis_names != None:

        for ax_obj, axis_name in zip(ax_objs, axis_names):
            ax_obj.set_ylabel(axis_name)

    if x_axis_name != None:
        ax_objs[-1].set_xlabel(x_axis_name)


#Don't use a violin plot
def violinplot(*args, **kwargs):

    import webbrowser
    webbrowser.open_new_tab("youtube.com/watch?v=_0QMKFzW9fw")

    try:
        ridgelinedf(*args, **kwargs)
        print("Don't use violin plots, here's a better plot")
    except:
        raise Exception("Don't use violin plots")
