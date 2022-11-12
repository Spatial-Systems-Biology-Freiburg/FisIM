import matplotlib.pyplot as plt
import numpy as np

from FisInMa import optimization as opt
plt.rcParams["font.family"] = "serif"
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{amsmath}"



def plot_discretization_product(outdir):
    x_discr = np.array([0.0, 2.0, 3.0, 6.0, 8.0, 9.0])
    x_values = np.linspace(0, 9, 500)

    y_prod, y_arr = opt.discrete_penalty_calculator_default(x_values, x_discr)
    fig, ax = plt.subplots()
    l1, = ax.plot(x_values, y_arr, linewidth=2)
    for i in range (len(x_discr)):
        l2, = ax.plot([x_discr[i] for _ in range (100)], np.linspace(-0.1, 1.1, 100), 'k', linestyle='dotted')
    ax.set_ylim(0.5, 1.01)
    ax.set_xlabel(r'$v$', fontsize=17)
    ax.set_ylabel(r'$U_1 (v)$', fontsize=17)
    #plt.xlim(4, 12)
    plt.legend([l1, l2], [r"Penalty", r"$v^\text{discr}$"], fontsize=15, handlelength=1.3, framealpha=0)
    plt.savefig(outdir / "discretization_product.png", transparent=True, bbox_inches='tight')
    plt.close(fig) 

def plot_discrete_penalty_individual_template(outdir):
    x_discr = np.array([0.0, 2.0, 3.0, 6.0, 8.0, 9.0])
    x_values = np.linspace(0, 9, 500)
    titles = ["penalty_structure_zigzag", "penalty_structure_cos", "penalty_structure_gauss"]

    y_prod, y_arr_zigzag = opt.discrete_penalty_individual_template(x_values, x_discr, opt.penalty_structure_zigzag)
    y_prod, y_arr_cos = opt.discrete_penalty_individual_template(x_values, x_discr, opt.penalty_structure_cos)
    y_prod, y_arr_gauss = opt.discrete_penalty_individual_template(x_values, x_discr, opt.penalty_structure_gauss)
    
    fig, ax = plt.subplots(3, 1, figsize=(7, 7.7), sharex=True)
    fig.subplots_adjust(hspace=0.4)
    l1, = ax[0].plot(x_values, y_arr_zigzag, linewidth=2)
    l2, = ax[1].plot(x_values, y_arr_cos, linewidth=2)
    l3, = ax[2].plot(x_values, y_arr_gauss, linewidth=2)
    for j in range (3):
        for i in range (len(x_discr)):
            l4, = ax[j].plot([x_discr[i] for _ in range (100)], np.linspace(-0.2, 1.2, 100), 'k', linestyle='dotted')
        ax[j].set_title(titles[j], fontsize=17)
        ax[j].set_xlabel(r'$v$', fontsize=14)
        ax[j].set_ylabel(r'$U_1 (v)$', fontsize=14)
        ax[j].set_ylim(-0.1, 1.1)
    
    #plt.xlim(4, 12)
    ax[0].legend([l1, l4], ["Penalty", r"$v^\text{discr}$"], fontsize=13, handlelength=1.3, framealpha=0)
    plt.savefig(outdir / "discretization_template.png", bbox_inches='tight', transparent=True)
    plt.close(fig) 

