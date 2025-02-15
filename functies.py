from csv import Error
from os import error
import numpy as np
#from sphinx import ret
import sympy as sp
import sympy.stats as stats
from matplotlib import pyplot as plt
from scipy.optimize import minimize, fsolve, root_scalar
from scipy.stats import chi2
import classes
from IPython.display import display



########### Algemene data analyse ############
def round_to_n(x, n): #rond uw data af op n beduidende cijfers
    return round(x, -int(np.floor(np.log10(abs(x))))-1+n) 

def foutpropagatie(expr, parameters):
    """
    Geeft formule voor foutenpropagatie in expr
    -------------------------
    @param:
     - expr: De vergelijking waar waardes ingevuld worden
     - parameters: Een lijst van datapunt objecten die ingevuld moeten worden in de vergelijking
    -------------------------
    @return:
     - sigmakwadr: Een formule om de nieuwe variantie te bepalen
    """
    sigmakwadr = 0
    for indx in range(len(parameters)):
        param = sp.diff(expr, parameters[indx].get_naam())
        variance = parameters[indx].get_variance()
        sigmakwadr += (param**2)*variance
    #display(sigmakwadr)
    return sigmakwadr
    
def data_analyse(equation, param_values, eval_name: sp.symbols, detailed_logs = False):
    """
    Voert foutenpropagatie uit op een vergelijking met waardes param_values
    ----------------------------
    @param:
     - equation: De vergelijking waarop foutenpropagatie moet gebeuren 
     - param_values: De waardes die ingevuld moeten worden in de vergelijking, als lijst van datapunt objecten, of als meting object (nog niet geïmplementeerd)
     - eval_name: De naam (een sympy symbool) van het resultaat
    ----------------------------
    @return:
     - Een datapunt object met waarde en fout bepaald via de vergelijking, normale verdeling en naam bepaald door eval_name
    """
    sigmas = []
    parameters = []
    substitutie = []
    vgl = equation.copy()
    #print(param_values)
    for param_value in param_values:
        sigmas.append(param_value.get_variance()**0.5)
        parameters.append(param_value.get_naam())
        substitutie.append((param_value.get_naam(), param_value.get_val()))
    sigmakwadr = foutpropagatie(vgl.formule, param_values)
    for subs in substitutie:
        sigmakwadr = sigmakwadr.subs(subs[0],subs[1])
    waarde = vgl.calculate(substitutie)
    sigmakwadr = sigmakwadr.evalf()
    datapt = classes.datapunt(waarde, sigmakwadr**0.5, eval_name, verdeling = "Normaal")
    if detailed_logs:
        print(sigmakwadr)
        print(datapt)
    return datapt

def multiple_analysis(equation, params_list, eval_name, detailed_logs = False):
    """
    Voert foutenpropagatie uit op een lijst van metingen (of dataset)
    ------------------------
    @param:
     - equation: De vergelijking die ingevuld moet worden
     - params_list: De data die ingevuld moet worden, als matrix van datapunt objecten, lijst van meting objecten (niet geïmplementeerd)
                    of als dataset object (niet geïmplementeerd)
     - eval_name: De naam van het resultaat
    @return: 
     - een np.array van datapunten    
    """
    data = []
    for params in params_list:
        data.append(data_analyse(equation, params, eval_name, detailed_logs))
    dat = np.array(data)
    return dat

def gemiddelde(waarden: list, naam = None):
    """
    Berekent het gemiddelde en de fout er op van een lijst (of matrix) meetwaarden of datapunt objecten
    ------------------------
    @param:
     - waarden: De lijst (of matrix) van waarden waarvan het gemiddelde moet bepaald worden. Gemiddeldes worden rij per rij bepaald
     - naam: Een naam (of een lijst van namen) voor de resultaten van de berekening. Als waarden bestaat uit datapunten wordt de naam
                datapunt.naam _ gem genomen
    ------------------------
    @return:
     - (Lijst van) datapunt object(en) met naam 
    """
    if type(waarden) != np.array:
        waarden = np.array(waarden)
    dimensies = waarden.ndim
    datapunten = False
    if dimensies == 1:
        if type(waarden[0]) == classes.datapunt:
            datapunten = True
            for i in range(len(waarden)):
                waarden[i] = waarden[i].get_val()
    elif dimensies == 2:
        if type(waarden[0][0]) == classes.datapunt:
            datapunten = True
            for i in range(len(waarden)):
                for j in range(len(waarden[i])):
                    waarden[i][j] = waarden[i][j].get_val()
    else:
        raise Error("Fix dimensies pls, enkel 1 of 2 werken!")
    if naam == None and not datapunten:
        raise Error("Geef een naam in als waarden geen datapunten zijn!")
    if dimensies == 1:
        som = np.sum(waarden)
        N = len(waarden)
        avg = som/N
    elif dimensies == 2:
        som = np.sum(waarden, axis=0)
        N = len(waarden[0])
        avg = som.T/N
    sigmasqsom = 0
    for element in waarden:
        sigmasqsom += (avg - element)**2
    sigmasqsom /= N*(N-1)
    sigma = np.sqrt(sigmasqsom)
    if dimensies == 1:
        if naam == None:
            naam = sp.symbols(str(waarden[0].naam) +"_gem")
        else:
            if type(naam) != sp.symbols:
                naam = sp.symbols(naam)
        terug = classes.datapunt(avg, sigma, naam)
    elif dimensies == 2:
        if naam == None:
            naam = [sp.symbols(str(waarden[0][i].naam) +"_gem") for i in range(len(waarden[0]))]
        else:
            if type(naam) != sp.symbols:
                naam = sp.symbols(naam)
        terug = [classes.datapunt([avg[i],sigma[i],naam[i]]) for i in range(len(waarden[0]))]
    return terug

def mu_sigma(waarden: list, naam = None):
    """
    Berekent het gewogen gemiddelde van een set data
    ----------------------
    @param:
     - waarden: een lijst (of matrix) van datapunt objecten waarvan het gewogen gemiddelde (over de kolommen) berekend wordt.
     - naam: waarden.naam _gem als naam == None, anders de geeft het de naam van de resulterende datapunt objecten
    ----------------------
    @return:
     - Een lijst (of matrix) van datapunt objecten met gegeven namen en waardes/fouten bepaald door gewogen gemiddelde (en normale verdeling)
    """
    if type(waarden) != np.array:
        waarden = np.array(waarden)
    if type(waarden[0]) != classes.datapunt and naam == None:
        raise Error("Geef datapunt objecten, ik heb er zo veel werk in gestoken")
    dimensies = waarden.ndim
    if dimensies == 2:
        if type(waarden[0][0]) != classes.datapunt:
            raise Error("Geef datapunt objecten, ik heb er zo veel werk in gestoken")
    vals = []
    g_vals = []
    if dimensies == 1:
        for waarde in waarden:
            vals.append(waarde.get_val())
            g_vals.append(1/waarde.get_variance())
    elif dimensies == 2:
        for waarde in waarden:
            vals.append([waarde[i].get_val() for i in range(len(waarde))])
            g_vals.append([waarde[i].get_variance() for i in range(len(waarde))])
    else:
        raise Error("Kijk bro, drie (of 0) dimensies is te veel (of te weinig)")
    vals = np.array(vals)
    g_vals = np.array(g_vals)
    
    if dimensies == 1:
        if naam == None:
            naam = sp.symbols(str(waarden[0].get_naam()) +"_gem")
        else:
            if type(naam) == str:
                naam = sp.symbols(naam)
            elif type(naam) == sp.Symbol:
                pass
            else:
                tiepuh = str(type(naam))
                raise Error("bro wtf uw naam is geen string en geen sympy.symbol maar een "+str(tiepuh))
        teller = np.sum(vals * g_vals)
        noemer = np.sum(g_vals)
        eind_waarde = teller/noemer
        eind_fout = 1/np.sqrt(noemer)
        outp = classes.datapunt(eind_waarde, eind_fout, naam)
    elif dimensies == 2:
        if naam == None:
            naam = [sp.symbols(str(waarden[0][i].naam) +"_gem") for i in range(len(waarden[0]))]
        else:
            if type(naam) != sp.symbols:
                naam = sp.symbols(naam)
        teller = np.sum(vals * g_vals, axis= 0)
        noemer = np.sum(g_vals, axis = 0)
        eind_waarde = teller/noemer
        eind_fout = 1/np.sqrt(noemer)
        outp = [classes.datapunt([eind_waarde[i],eind_fout[i],naam[i]]) for i in range(len(waarden))]
    return outp

        
    
########### Fit code - 1D ############
def chi2_bereken(param, x_val, y_val, y_err, soort_fout, model):
    """Geeft chi^2 waarde in functie van de parameters
    
    Args: 
        param: Waardes voor de parameters van het model
        model: Het gebruikte model dat gefit wordt
        x_val: Een vector van invoerwaardes voor het model
        y_val: Een vector met meetwaardes die gefit moeten worden
        y_err: Een vector met de fouten op y
        
    Return:
        chi_2_val: De chi^2 waarde van het model gegeven de waardes voor de parameters uit param.
        
    """
    if soort_fout == "Unif":
        fouten = y_err**2 / 12
    else:
        print(y_err)
        fouten = y_err**2
    chi_2_val = np.sum((y_val - model(x_val, param))**2 / fouten)
    return chi_2_val

def minimize_chi2(model, initial_vals, x_val, y_val, y_err, soort_fout = 'Stat'):
    """Minimaliseert de chi^2 waarde voor een gegeven model en een aantal datapunten
    
    Args:
        model: Het gebruikte model dat gefit wordt.
        x_val: Een vector van invoerwaardes voor het model
        y_val: Een vector met meetwaardes die gefit moeten worden
        y_err: Een vector met de fouten op y
        soort_fout: Laat toe om het type fout mee te geven, dit is enkel van belang als de fout op de meetpunten uniform is.
                    
    Return:
        oplossing: Een array dat de minimale waardes voor de parameters geeft
    """
    chi2_func = lambda *args: chi2_bereken(*args)
    gok = initial_vals(x_val, y_val)
    mini = minimize(chi2_func, gok, args = (x_val, y_val, y_err, soort_fout, model), method="Nelder-Mead")
    return mini

def chi2_in_1_var(var, ind_var, x_val, y_val, y_err, param_values, chi_min, model, soort_fout = "Stat"):
    outp = np.array([])
    aant_param = len(param_values)
    for val in var:
        kopie = param_values.copy()
        np.put(kopie, ind_var, val)
        outp = np.append(outp, chi2_bereken(kopie, x_val, y_val, y_err, soort_fout, model) - chi2.ppf(0.68, df=aant_param) - chi_min)
    return outp

def find_sigma_values(x_val, y_val, y_err, param_values, te_checken_param_ind, chi_min, soort_fout, model):
    functie = lambda *args: chi2_in_1_var(*args)
    gok = param_values[te_checken_param_ind]
    oplossing_max = fsolve(functie, args = (te_checken_param_ind, x_val, y_val, y_err, param_values, chi_min, model, soort_fout), x0 = gok + abs(gok)/2)
    oplossing_min = fsolve(functie, args = (te_checken_param_ind, x_val, y_val, y_err, param_values, chi_min, model, soort_fout), x0 = gok - abs(gok)/2)
    return [oplossing_min[0], oplossing_max[0]]
    
def uncertainty_intervals(min_values, x_val, y_val, y_err,  chi_min, model, soort_fout = "Stat"):
    aant_param = len(min_values)
    intervallen = []
    for i in range(0, aant_param):
        intervallen.append(find_sigma_values(x_val, y_val, y_err, min_values, i, chi_min, soort_fout, model))
    return intervallen

def fit(parameters, model, initial_vals, x_val, y_val, y_err, soort_fout = "Stat", 
        x_as_titels = "Generic", y_as_titels = "Generic", titel = "Generic", detailed_logs = False): #Veel van deze inputs doen niets, kmoet nog pretty
    #print code schrijven
    #TODO: cas_matrix support maken
    #TODO: ML code schrijven
    print("Raw output")
    mini = minimize_chi2(model, initial_vals, x_val, y_val, y_err, soort_fout)
    chi_min = mini["fun"]
    min_param = mini["x"]
    print(mini)
    
    betrouwb_int = uncertainty_intervals(min_param, x_val, y_val, y_err, chi_min, model, soort_fout)
    print(betrouwb_int)
    foutjes = []
    for i in range(0, len(parameters)):
        top = betrouwb_int[i][1] - min_param[i]
        bot = min_param[i] - betrouwb_int[i][0]
        foutjes.append((bot, top))
        outp = parameters[i] + " heeft als waarde: %.5g + %.5g - %.5g met 68%% betrouwbaarheidsinterval: [%.5g, %.5g] "%(min_param[i], top, bot, betrouwb_int[i][0], betrouwb_int[i][1])
        print(outp)

    nu = len(x_val) - len(parameters)
    p_waarde = chi2.sf(chi_min, df=nu)
    chi_red = chi_min/nu
    if detailed_logs:
        for i in range(len(min_param)):
            plot_chi2((betrouwb_int[i], i), min_param, x_val, y_val, y_err, soort_fout, model, len(parameters), chi_min)
    
    print("De p-waarde voor de hypothese test dat het model zinvol is, wordt gegeven door: %.5g"%p_waarde)
    print("De gereduceerde chi^2 waarde is: %.5g"%chi_red)
    fouten = []
    for fout in foutjes:
        if abs(fout[0]/fout[1]) < 1.25 and abs(fout[0]/fout[1]) > 0.8:
            fouten.append(round_to_n(max(fout[0], fout[1]), 2))
        else:
            fouten.append((round_to_n(fout[0],2),round_to_n(fout[1],2)))
    outp = []
    for i in range(0, len(parameters)):
        outp.append([min_param[i], fouten[i], 'S'])
    return outp


def plot_chi2(plotwaarde, min_param, x_val, y_val, y_err, soort_fout, model, n_param, chi_min):
    """
    plotwaarde = (range, indx)
    range is de linspace waarover geplot wordt bij de parameter met index indx
    """
    [bot, top], indx = plotwaarde
    rangge = np.linspace(bot, top, 10000)
    fig, ax = plt.subplots(1,1)
    y_as = []
    for i in rangge:
        parami = min_param.copy()
        parami[indx] = i
        y_as.append(chi2_bereken(parami, x_val, y_val, y_err, soort_fout, model))
    y_as = np.array(y_as)
    ax.plot(rangge, y_as, label = "$\\chi^2$ in functie van param")
    ax.errorbar([min_param[indx]], [chi_min], fmt = "o", label = "Optimale punt", color = "k")
    ax.plot(rangge, np.full(len(rangge), chi_min + chi2.ppf(0.68, df=n_param)), label = "Minimale $\\chi^2$ plus 1$\\sigma$")
    ax.set_xlabel('Parameter op index'+str(indx))
    ax.set_ylabel('$\\chi^2$')
    ax.legend()
    plt.tight_layout();plt.show()

########## Fit code - 2D ###########
def  chi2_bereken_2D(hybrid, x_val, y_val, x_variance, y_variance, model, n_param):
    """Geeft chi^2 waarde in functie van de parameters
    
    Args: 
        hybrid: combinatie van param en x_guesses
            param: Waardes voor de parameters van het model
            x_guesses: Nog "parameter" waardes, zie demming regression op wikipedia
        model: Het gebruikte model dat gefit wordt
        x_val: Een vector van invoerwaardes voor het model
        y_val: Een vector met meetwaardes die gefit moeten worden
        y_err: Een vector met de fouten op y
        x_err: Een vector met de fouten op x
        
    Return:
        chi_2_val: De chi^2 waarde van het model gegeven de waardes voor de parameters uit param.
        
    """
    param = hybrid[:n_param]
    x_guesses = hybrid[n_param: ]
    x_diffs = ((x_val - x_guesses)**2)/x_variance
    y_diffs = ((y_val - model(x_guesses, param))**2)/y_variance

    chi_2_val = np.sum(x_diffs + y_diffs)
    return chi_2_val

def plot_chi2_2D(plotwaarde, min_param, x_val, y_val, x_variance, y_variance, model, n_param, chi_min):
    """
    plotwaarde = (range, indx)
    range is de linspace waarover geplot wordt bij de parameter met index indx
    """
    [bot, top], indx = plotwaarde
    rangge = np.linspace(bot, top, 10000)
    fig, ax = plt.subplots(1,1)
    y_as = []
    for i in rangge:
        parami = min_param.copy()
        parami[indx] = i
        y_as.append(chi2_bereken_2D(parami, x_val, y_val, x_variance, y_variance, model, n_param))
    y_as = np.array(y_as)
    ax.plot(rangge, y_as, label = "$\\chi^2$ in functie van param")
    ax.errorbar([min_param[indx]], [chi_min], fmt = "o", label = "Optimale punt", color = "k")
    ax.plot(rangge, np.full(len(rangge), chi_min + chi2.ppf(0.68, df=n_param)), label = "Minimale $\\chi^2$ plus 1$\\sigma$")
    ax.set_xlabel('Parameter op index'+str(indx))
    ax.set_ylabel('$\\chi^2$')
    ax.legend()
    plt.tight_layout();plt.show()

def initial_vals_2D(x_val, y_val, initial_vals):
    param_initials = initial_vals(x_val, y_val)
    outp = np.concatenate((param_initials, x_val))
    print(param_initials)
    print(outp)
    return outp

def minimize_chi2_2D(model, initial_vals, x_val, y_val, y_variance, x_variance, n_param):
    """Minimaliseert de chi^2 waarde voor een gegeven model en een aantal datapunten
    
    Args:
        model: Het gebruikte model dat gefit wordt.
        x_val: Een vector van invoerwaardes voor het model
        y_val: Een vector met meetwaardes die gefit moeten worden
        y_err: Een vector met de fouten op y
        soort_fout: Laat toe om het type fout mee te geven, dit is enkel van belang als de fout op de meetpunten uniform is.
                    
    Return:
        oplossing: Een array dat de minimale waardes voor de parameters geeft
    """
    chi2_func = lambda *args: chi2_bereken_2D(*args)
    gok = initial_vals_2D(x_val, y_val, initial_vals)
    mini = minimize(chi2_func, gok, args = (x_val, y_val,x_variance, y_variance, model, n_param))
    return mini

def chi2_in_1_var_2D(var, ind_var, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param):
    try:
        some_object_iterator = iter(var)
        outp = np.array([])
        display(var)
        for val in var: #Laat deze functie een vector gebruiken voor var i.p.v. slechts 1 waarde.
            #Deze regel dient om de referentie semantiek van lijsten te omzeilen. Zonder dit wordt de vector param_values globaal aangepast
            kopie = np.copy(hybrid)
             #De waarde van var wordt op de juiste index ingevuld in de parameter vector.
            np.put(kopie, ind_var, val)
            outp = np.append(outp, chi2_bereken_2D(kopie, x_val, y_val, x_variance, y_variance, model, n_param) - chi2.ppf(0.68, df=n_param) - chi_min)
        return outp
    except TypeError as te:
        kopie = np.copy(hybrid)
        #De waarde van var wordt op de juiste index ingevuld in de parameter vector.
        np.put(kopie, ind_var, var)
        outp = chi2_bereken_2D(kopie, x_val, y_val, x_variance, y_variance, model, n_param) - chi2.ppf(0.68, df=n_param) - chi_min
        return outp

def find_sigma_values_2D(x_val, y_val, x_variance,  y_variance, hybrid, te_checken_param_ind, chi_min, model, n_param, grootteorde, detailed_logs = False):
    functie = lambda *args: chi2_in_1_var_2D(*args)
    gok = hybrid[te_checken_param_ind]
    if detailed_logs:
        print("------------------------")
        print("gok: %s" %gok)
    #De snijpunten met de 1\sigma hypercontour van de chi^2_mu verdeling zullen rond de best fittende waardes liggen
    i = 0.1
    terminate = False
    while i<=3 and not terminate:
        #scipy.optimize.fsolve vindt de nulpunten van de gegeven functie, chi2_in_1_var is gedefinieerd zodat de gezochte boven
        #en ondergrenzen van het BI precies de nulpunten zijn.
        #Om de bovengrens te vinden wordt een initiële waarde boven de best fittende waarde genomen, omgekeerd voor de ondergrens.
        try:
            print(functie)
            sol_left = root_scalar(functie, args = (te_checken_param_ind, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param), method = "brentq", bracket = [gok*(1-i), gok], x0 = gok, x1 = (1-i)*gok)
            sol_right = root_scalar(functie, args = (te_checken_param_ind, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param), method = "brentq", bracket = [gok, gok*(1+i)], x0 = gok, x1 = (1+i)*gok)
            if detailed_logs:
                print("Root scalar worked!")
            return [sol_left.root, sol_right.root]
        except:
            try:
                print(functie)
                sol_left = root_scalar(functie, args = (te_checken_param_ind, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param), method = "brentq", bracket = [gok*(1-i), gok], x0 = gok, x1 = (1-i)*gok)
                sol_right = root_scalar(functie, args = (te_checken_param_ind, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param), method = "brentq", bracket = [gok, gok*(1+i)], x0 = gok, x1 = (1+i)*gok)
                if detailed_logs:
                    print("Root scalar worked by askin' Papa Newton for help!")
                return [sol_left.root, sol_right.root]
            except:
                if i <= 2:
                    if detailed_logs:
                        print("i increased!, now %s" %i)
                        print("Nieuwe lower bound is %s, met waarde %s" %((1-i)*gok, chi2_in_1_var_2D((1-i)*gok, te_checken_param_ind, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param)))
                        print("---------------------")
                    i+=0.1
                else:
                    if detailed_logs:
                        print("tried to terminate!")
                    terminate = True
    if terminate:
        if detailed_logs:
            print("Succesfully terminated !")
            print("Geen fout gevonden in 300 percent foutenmarge")
        return [-5*grootteorde, 5*grootteorde]
    else:
        print('WARNING: UNEXPECTED OCCURENCE HAS HAPPENED, PLEASE PROCEED DEBUGGING functies.find_sigma_values_2D')
        return [None, None]

def uncertainty_intervals_2D(min_hybrid, x_val, y_val, x_variance, y_variance,  chi_min, model, n_param, grootteorde, detailed_logs = False):
    intervallen = []
    for i in range(0, n_param):
        intervallen.append(find_sigma_values_2D(x_val, y_val, x_variance, y_variance, min_hybrid, i, chi_min, model, n_param, grootteorde, detailed_logs))
    return intervallen

def jackknife_parameters_i(index, n_param, model, initial_vals, x_val, y_val, x_variance, y_variance):
    x_val_i = np.delete(x_val, index)
    x_variance_i = np.delete(x_variance, index)
    y_val_i = np.delete(y_val, index)
    y_variance_i = np.delete(y_variance, index)
    mini = minimize_chi2_2D(model, initial_vals, x_val_i, y_val_i, y_variance_i, x_variance_i, n_param)
    return np.array(mini["x"][:n_param])

def jackknife_parameterschattingen(model, initial_vals, n_param, x_val, y_val, x_variance, y_variance, min_params):
    num_points = len(x_val)
    i_pseudovariat = np.zeros((num_points,n_param))
    for i in range(num_points):
        i_de_schatting = jackknife_parameters_i(i, n_param, model, initial_vals, x_val, y_val, x_variance, y_variance)
        corrected = num_points*min_params - (num_points - 1)*i_de_schatting
        i_pseudovariat[i] = corrected
    jackknife_estimation = np.sum(i_pseudovariat, axis=0)/num_points
    jackknife_variance = np.sum((i_pseudovariat - jackknife_estimation)**2, axis=0)/(num_points-1)
    jackknife_standard_error = np.sqrt(jackknife_variance/num_points)
    return (jackknife_estimation, jackknife_standard_error)

def plot_fit(x_val, y_val, x_variance, y_variance, x_as_titel, y_as_titel, titel, model, parameter_vals, chi_2, p, save_name = None, size = None, savefig = False,fontsize = 5, titlesize = None, axsize = None):
    if size is None:
        fig, ax = plt.subplots(1,1, figsize = (10,10))
    else:
        fig, ax = plt.subplots(1,1, figsize = size)
    ax.errorbar(x_val, y_val, xerr = np.sqrt(x_variance), yerr = np.sqrt(y_variance), 
                fmt="o", label = "Datapunten", color = "k", ecolor= "k", elinewidth=0.8, capsize=1)
    length = np.max(x_val) - np.min(x_val)
    t = np.linspace(np.min(x_val) - length/20, np.max(x_val) + length/20, 100000)
    model_label = "Model waardes, \n $\chi^2_{red}$ = %.2f, p = %.2f %%" %(chi_2, p*100)
    ax.plot(t, model(t, parameter_vals), 'r--', label = model_label)
    if axsize is None:
        axsize = fontsize
    ax.set_xlabel(x_as_titel, fontsize = axsize)
    ax.set_ylabel(y_as_titel, fontsize = axsize)
    if titlesize is None:
        titlesize = fontsize
    ax.set_title(titel,fontsize = titlesize)
    ax.legend(fontsize = fontsize)
    if save_name is not None:
        plt.savefig(save_name)
    if savefig:
        plt.savefig(titel+'.png')
    plt.show()


def fit_2D(parameters, model, initial_vals, x_val, y_val, x_variance, y_variance, grootteorde = 1,
        x_as_titel = "X-as", y_as_titel = "Y-as", titel = "Fit", figure_name = None, size = None,
        error_method = "Old", savefig = False, detailed_logs = False, fontsize = 18, titlesize = 20, axsize = 16): 
    """
    OUTDATED CODE, GEBRUIK NIEUWERE FUNCTIES INDIEN DEZE AL GEÏMPLEMENTEERD ZIJN
    #################################
    @param:
     - parameters: De parameters van het model, in een vector gegeven
     - model: Het model dat gefit wordt. Dit dient een functie model(x, param) te zijn die in x een vector datapunten kan accepteren
     - initial_vals: Initiële waardes voor de fitparameters
     - x_val: Een vector met x_waardes
     - y_val: Een vector met y_waardes van dezelfde grootte als x_val
     - x_variance: De varianties van de x_waardes, in dezelfde volgorde en van dezelfde lengte als x_val
     - y_variance: De varianties van de y_waardes, in dezelfde volgorde en van dezelfde lengte als y_val
    
    @kwargs:
     - grootteorde = 1: Geeft een schatting van de grootteorde van de fitparameters
     - x_as_titel = "X-as": De titel van de x-as van de grafiek van de datapunten en de fit
     - y_as_titel = "Y-as": De titel van de y-as van de grafiek van de datapunten en de fit
     - titel = "Fit": De titel van de grafiek van de datapunten en de fit
     - figure_name = None: If not None, geeft de naam waaronder de grafiek opgeslagen moet worden
     - size = None: If not None, geeft de grootte van de grafiek
     - error_method = "Old": Geeft de methode waarmee gefit moet worden, is "Old" of "Jacknife" (Jacknife werkt momenteel nogal slecht)
     - savefig = False: If True, slaag de figuur met de fit op onder de naam "titel.png"
     - detailed_logs = False: If True, geeft gedetaileerd logs van het fitprocess
     - fontsize, titlesize, axsize: Geven fontgroottes van respectievelijk de legende, titel en assen.
    #############################################################
    
    @return:
     - outp: Een matrix van de vorm [[min_param[i], param_fout[i], "S"]]
    """
    #Veel van deze inputs doen niets, kmoet nog pretty
    #print code schrijven
    #TODO: cas_matrix support maken
    #TODO: ML code schrijven
    n_param = len(parameters)
    mini = minimize_chi2_2D(model, initial_vals, x_val, y_val, y_variance, x_variance, n_param)
    chi_min = mini["fun"]
    min_hybrid = mini["x"]
    min_param = min_hybrid[:n_param]
    if detailed_logs: 
        print("Ditctionary van minimize:")
        print(mini)
        print("---------------------------")
        print("Minimale parameter waardes:")
        print(min_param)
        print("---------------------------")
        print("Minimale hybrid waardes:")
        print(min_hybrid)
        print("---------------------------")
    if error_method == "Old":
        betrouwb_int = uncertainty_intervals_2D(min_hybrid, x_val, y_val, x_variance, y_variance, chi_min, model, n_param, grootteorde, detailed_logs)
        if detailed_logs:
            print("Betrouwbaarheids intervallen voor de parameters: ")
            print(betrouwb_int)
            print("---------------------------")
        print(betrouwb_int)
        foutjes = []
        for i in range(0, n_param):
            top = betrouwb_int[i][1] - min_param[i]
            bot = min_param[i] - betrouwb_int[i][0]
            foutjes.append((bot, top))
            outp = str(parameters[i]) + " heeft als waarde: %.5g + %.5g - %.5g met 68%% betrouwbaarheidsinterval: [%.5g, %.5g] "%(min_param[i], top, bot, betrouwb_int[i][0], betrouwb_int[i][1])
            print(outp)
        nu = len(x_val) - n_param
        p_waarde = chi2.sf(chi_min, df=nu)
        chi_red = chi_min/nu
        if detailed_logs:
            for i in range(n_param):
                top = betrouwb_int[i][1] - min_param[i]
                bot = min_param[i] - betrouwb_int[i][0]
                plot_chi2_2D(([betrouwb_int[i][0] -bot*0.1, betrouwb_int[i][1] + top*0.1], i), min_hybrid, x_val, y_val, x_variance, y_variance, model, n_param, chi_min)
        
        print("De p-waarde voor de hypothese test dat het model zinvol is, wordt gegeven door: %.5g"%p_waarde)
        print("De gereduceerde chi^2 waarde is: %.5g"%chi_red)
        fouten = []
        for fout in foutjes:
            if fout[0]/fout[1] < 1.25 and fout[0]/fout[1] > 0.8:
                fouten.append(max(fout[0], fout[1]))
            else:
                fouten.append(fout)
        outp = []
        for i in range(0, len(parameters)):
            outp.append([min_param[i], fouten[i], 'S'])
        if not fontsize is None:
            plot_fit(x_val, y_val, x_variance, y_variance, x_as_titel, y_as_titel, titel, model, min_param, chi_red, p_waarde, figure_name, size,savefig=savefig, fontsize = fontsize, titlesize=titlesize,axsize=axsize)
        else:
            plot_fit(x_val, y_val, x_variance, y_variance, x_as_titel, y_as_titel, titel, model, min_param, chi_red, p_waarde, figure_name, size, savefig=savefig ,titlesize=titlesize,axsize=axsize)
        return outp
    elif error_method == "Jackknife":
        Jackknife_result = jackknife_parameterschattingen(model, initial_vals, n_param, x_val, y_val, x_variance, y_variance, min_param)
        parameter_vals = Jackknife_result[0]
        parameter_fouten = Jackknife_result[1]
        if detailed_logs:
            print(Jackknife_result)
        for i in range(len(parameter_vals)):
            outp = str(parameters[i]) + " heeft als waarde: %.5g \pm %.5g"%(parameter_vals[i],parameter_fouten[i])
            print(outp)
        
        fin = np.append(parameter_vals, min_hybrid[n_param:])
        nu = len(x_val) - n_param
        chi_val = chi2_bereken_2D(fin, x_val, y_val, x_variance, y_variance, model, n_param)
        p_waarde = chi2.sf(chi_val, df=nu)
        chi_red = chi_val/nu
        print("De p-waarde voor de hypothese test dat het model zinvol is, wordt gegeven door: %.5g"%p_waarde)
        print("De gereduceerde chi^2 waarde is: %.5g"%chi_red)
        plot_fit(x_val, y_val, x_variance, y_variance, x_as_titel, y_as_titel, titel, model, parameter_vals, "idk", "Geen flauw idee", figure_name, size, savefig)
        return parameter_vals #TODO: Dit met de nieuwe klasses doen

    else:
        print("Given error method not yet implemented, try ""Jackknife"" instead.")
        return None
    #Werkt nog niet, kmoet de code nog algemeen schrijven :(
    #if printen:
    #    print("##################### Pretty print #####################")
    #    pretty_print_results(x_val, y_val, y_err, chi_min, min_param, betrouwb_int, parameters)

########## stat ############

def normaaltest(hypothese, gemeten): #gemeten is [mu, sigma] merk op dat sigma de steekproefstand.afwijking is!
    Zscore = (hypothese-gemeten[0])/gemeten[1]
    Z = sp.stats.Normal('Z', 0,1)
    if Zscore >0:
        return 2*sp.stats.P(Z > Zscore)
    else:
        return 2*sp.stats.P(Z < Zscore)

def Ttest(hypothese, gemeten, vrijheidsgraden):
    Zscore = (hypothese-gemeten[0])/gemeten[1]
    Z = sp.stats.StudentT('Z', vrijheidsgraden)
    if Zscore >0:
        return 2*sp.stats.P(Z > Zscore)
    else:
        return 2*sp.stats.P(Z < Zscore)    

def test_sigma1issigma2(sigma1,sigma2, n1, n2, p_waarde = 0.05, return_p = False):
    sigmamax = max(sigma1, sigma2)
    sigmamin = min(sigma1, sigma2)
    Fvalue = sigmamax**2/sigmamin**2 #altijd >= 1
    F = sp.stats.FDistribution('F',n1-1,n2-1)
    kans = 2*sp.stats.P(F > Fvalue)
    kans = kans.evalf()
    if return_p:
        return kans
    else:
        if kans >= p_waarde:
            return True #de twee sigma's zijn gelijk
        else:
            return False
    
def test_mu1ismu2(meting1, meting2, n1, n2, p_waarde = 0.05, return_p = False, cutoff_normaal = 30):
    mu1, sigma1, _ = meting1
    mu2, sigma2, _ = meting2
    sigma1_is_sigma2 = test_sigma1issigma2(sigma1, sigma2, n1, n2, p_waarde)
    if sigma1_is_sigma2:
        print('standaardafwijkingen zijn gelijk')
        SIGMA = np.sqrt(((n1-1)*sigma1**2 +(n2-1)*sigma2**2)/ (n1+n2-2))
        noemer = SIGMA*np.sqrt(1/n1 + 1/n2)
        Twaarde = abs((mu1-mu2)/noemer)
        print('T = ', end = ' ')
        print(Twaarde)
        if n1 + n2 -2 < cutoff_normaal:
            T = sp.stats.StudentT('T',n1+n2-2)
        else: #dat ding kan niet rekenen met te veel vrijheidsgraden in de T-test
            T = sp.stats.Normal('T',0, 1)
        kans = 2*sp.stats.P(T > Twaarde)
        print('p =',end = ' ')
        kans = kans.evalf()
        print(kans)

    else:
        print('standaardafwijkingen zijn ongelijk')
        r = (sigma1**2/n1 + sigma2**2/n2)**2 / ( (sigma1**2/n1)**2/(n1-1) + (sigma2**2/n2)**2/(n2-1))
        noemer = np.sqrt(sigma1**2/n1+sigma2**2/n2)
        Twaarde = abs((mu1-mu2)/noemer)
        print('r = ', end = ' ')
        print(r)
        if r < cutoff_normaal:
            r = sp.floor(r)
            T = sp.stats.StudentT('T',r)
        else: #dat ding kan niet rekenen met te veel vrijheidsgraden in de T-test
            T = sp.stats.Normal('T',0, 1)
        print('T = ',end = ' ');print(Twaarde)
        kans = 2*sp.stats.P(T > Twaarde)
        print('p = ',end = ' ')
        kans = kans.evalf()
        print(kans)
        
    
    if return_p:
        return kans
    else:
        if kans < p_waarde:
            return False
        else:
            return True

######## latex prints #########

def latex_print_tabel(meetwaarden, namen, tabletype = 'row'):
    """
    namen = lijst [param1naam, param2naam, param3naam, ...], de hoofdingen van de tabel
    meetwaarden = matrix [ [param1meting1, param2meting1, param3meting1,...],
                           [param1meting2, param2meting2, param3meting3,...],
                        ]
                met elke paramimetingj = [waarde, fout] (het standaard formaat)

    tabletype = 'row' / 'col'
    output:
        een latex tabel
        elke rij komt overeen met een rij in de tabel bij 'row'
                                  een kolom in de tabel bij 'col'
    """
    
    if tabletype == 'col':
        printmat = []
        for kolom in range(len(meetwaarden[0])):
            kol = []
            for rij in range(len(meetwaarden)):
                kol.append(meetwaarden[rij][kolom])
            printmat.append(kol.copy())
        meetwaarden = printmat.copy()
    elif tabletype != 'row':
        raise ValueError('EEJ FOEMP LEES DE DOCUMENTATIE')
    print('\\begin{table}[h!]')
    tabular = '\\begin{tabular}{' + '|c'*len(namen) + '|}'
    print(tabular)

    #de hoofdingen
    print('\\hline')
    for naam in namen[:-1]:
        print(naam, end = '&')
    print(namen[-1],end = '\\\\')

    #de meetwaarden
    for reeks in meetwaarden:
        print('\\hline')
        for meting in reeks[:-1]:
            output = latex_print_meting(meting, printing = False)
            print(output, end = '&')
        print(latex_print_meting(reeks[-1], printing = False), end = '\\\\')
    print('\\hline')
    print('\\end{tabular}\\end{table}')

def latex_print_meting(meetwaarde, naam = None, printing = True):
    """
    meetwaarden is de standaard vector [waarde, fout] met eventueel 'U'/'N' erachter
    het kan ook een datapunt object zijn
    """
    if type(meetwaarde) == classes.datapunt:
        naam = str(meetwaarde.get_naam())
        meetwaarde = datapunt_to_vector(meetwaarde)
    waarde, fout = meetwaarde[0],meetwaarde[1]
    exponent = 0
    TUPPEL = False

    if type(fout) == tuple:
        print(meetwaarde) #dit moet ik eens fixen als ik goesting heb

        if 0.9 <= fout[1]/fout[0] <= 1.1:
            fout = max(fout)
        else:
            TUPPEL = True
            TUPPELVAL = fout
            fout = max(fout)

    while not 1 <= fout*(10**(-1*exponent)) < 100:
        if 1 < fout:
            exponent += 1
        else:
            exponent -= 1
    waarde /= 10**exponent
    fout /= 10**exponent
    if TUPPEL:
        foutlinks = TUPPELVAL[0]/10**exponent
        foutrechts = TUPPELVAL[1]/10**exponent
    waarde = round(waarde, 1)
    fout = round(fout,1)
    if not TUPPEL:
        latex_output = f'(%s \\pm %s)\\cdot 10^{{%s}}$'%(waarde, fout,exponent)
    else:
        latex_output = f'%s^{{-%s}}_{{+%s}}\\cdot 10^{{%s}}$'%(waarde, foutlinks, foutrechts, exponent)
    if naam is None:
        latex_output = '$' + latex_output
    else:
        latex_output = '$' + naam +' = ' + latex_output
    if printing:
        print(latex_output)
    return latex_output    



######## Hulpfuncties voor classes ########


####### datatype transformations ##########
def datapunt_to_vector(datapunt):
    waarde = datapunt.get_val()
    fout = datapunt.get_fout()
    verdeling = datapunt.get_verdeling()
    return [waarde, fout, verdeling]

def datapuntmatrix_to_matrix(datalijst):
    """
    Input: een matrix [[param1datapunt1, param1datapunt2, ...],[paramZdatapunt1, param2datapunt2,...]]
    Output: dezelfde matrix maar elk datapunt is een vector
    """
    matrix = [[datapunt_to_vector(punt) for punt in kolom]
              for kolom in datalijst]
    return matrix

def vector_to_datapunt(vector, variabele):
    """
    Input: een vector in dataformaat, en de variabele die het representeert
    Output: het datapunt van deze vector
    """
    alfa = sp.symbols('alpha')
    if type(variabele) != type(alfa):
            raise TypeError("variabele moet een sp.symbol zijn")
    waarde, fout, soort_fout = vector
    return classes.datapunt(waarde, fout, variabele, soort_fout)

def matrix_to_datapunten(matrix, variabele):
    """
    Input: een matrix in dataformaat, en de variabele die zij representeren
    Output: een lijst (in de juiste volgorde) van datapunten van deze datamatrix
    """
    alfa = sp.symbols('alpha')
    if type(variabele) != type(alfa):
            raise TypeError("variabele moet een sp.symbol zijn")
    datapunten = []
    for vector in matrix:
        datapunten.append(vector_to_datapunt(vector, variabele))
    return datapunten


def same_order(to_order, guide):
    if len(to_order) != len(guide):
        raise Exception("Lengtes moeten gelijk zijn.")
    order_names = set()
    guide_names = set()
    if type(guide[0]) == tuple:
        for i in range(len(guide)):
            order_names.add(to_order[i][1])
            guide_names.add(guide[i][1])
        if order_names != guide_names:
            raise Exception("Namen niet allemaal gelijk.")
        help_array = np.zeros(len(guide))
        for i in range(len(guide)):
            ind = [k for k in range(len(guide)) if guide[k][1] == to_order[i][1]]
            if len(ind) !=1:
                raise Exception("Elke naam hoort slechts eenmaal voor te komen.")
            help_array[ind] = to_order[i]
    elif type(guide[0]) == str or type(guide[0]) == sp.symbols:
        for i in range(len(guide)):
            order_names.add(to_order[i][1])
            guide_names.add(guide[i])
        if order_names != guide_names:
            raise Exception("Namen niet allemaal gelijk.")
        help_array = np.zeros(len(guide))
        for i in range(len(guide)):
            ind = [k for k in range(len(guide)) if guide[k] == to_order[i][1]]
            if len(ind) !=1:
                raise Exception("Elke naam hoort slechts eenmaal voor te komen.")
            help_array[ind] = to_order[i]
    else:
        raise Exception("Guide hoort effectief namen te bevatten foemp.")
    return help_array

    