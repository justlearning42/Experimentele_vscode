from os import error
import numpy as np
import sympy as sp
import sympy.stats as stats
from matplotlib import pyplot as plt
from scipy.optimize import minimize, fsolve, root_scalar
from scipy.stats import chi2
import classes
from IPython.display import display



########### Algemene data analyse ############

def foutpropagatie(expr, parameters, sigmas):
    # onveranderd
    sigmakwadr = 0
    for indx in range(len(parameters)):
        param = sp.diff(expr, parameters[indx])
        sigm = sigmas[indx]
        sigmakwadr += (param*sigm)**2
    return sigmakwadr

def data_analyse(equation, param_values, eval_name):
    # param_values is een lijst van datapunt objecten
    fouten = []
    parameters = []
    substitutie = []
    for param_value in param_values:
        fouten.append(param_value.variance)
        parameters.append(param_value.naam)
        substitutie.append((param_value.naam, param_value.waarde))
    sigmakwadr = foutpropagatie(equation.formule.copy(), parameters, fouten)
    waarde = equation.subs(substitutie)
    return classes.datapunt(waarde, sigmakwadr, eval_name, verdeling = "Normaal")

def multiple_analysis(equation, params_list, eval_name):
    """
    params_list is een lijst van lijsten van datapunt objecten
    
    return: een np.array van datapunten
    """
    data = []
    for params in params_list:
        data.append(data_analyse(equation, params, eval_name))
    dat = np.array(data)
    return dat

def gemiddelde(waarden): #bepaalt het gemiddelde van N getallen en de fout op het gemiddelde
                         #kan ook werken met een array van arrays van getallen (meerdere berekeningen tegelijk)
    if type(waarden) != np.array:
        waarden = np.array(waarden)
    dimensies = waarden.ndim
    if dimensies == 1:
        som = np.sum(waarden)
        N = len(waarden)
        avg = som/N
    elif dimensies == 2:
        som = np.array([np.sum(waarden[i]) for i in range(len(waarden))])
        N = len(waarden[0])
        avg = som/N
    sigmasqsom = 0
    waardentranspose = waarden.T
    for elem in waardentranspose:
        sigmasqsom += (elem-avg)**2
    sigmasqsom /= N*(N-1)
    sigma = np.sqrt(sigmasqsom)
    terug = 'eej foemp fix uw dimensies'
    if dimensies == 1:
        terug = [avg, sigma, 'S']
    elif dimensies == 2:
        terug = [[avg[i],sigma[i],'S'].copy() for i in range(len(waarden))]
    return terug

def mu_sigma(waarden): #waarden met hun fout; bepaalt het gemiddelde en de meetfout (statistisch en/of meetfout)
    fout = waarden[0][1]
    gewogengemiddelde = False
    for waarde in waarden:
        if waarde[1] != fout:
            gewogengemiddelde = True
    if not gewogengemiddelde:
        #gemiddelde
        value = 0
        n = len(waarden)
        for waarde in waarden:
            value += waarde[0]
        mu = value/n
        #standaardafwijking
        value = 0
        for waarde in waarden:
            value += (mu-waarde[0])**2
        value /= (n-1)
        sigma_stat = np.sqrt(value)
    else:
        value = 0
        gewicht = 0
        n = len(waarden)
        for waarde in waarden:
            gew = 1/(waarde[1]**2)
            value += gew*waarde[0]
            gewicht += gew
        mu = value/gewicht
        sigma = 1/sp.sqrt(gewicht)
        return (mu, sigma, 'fout bij niet-constante meetfout')
    #welke fout teruggeven?
    if sigma_stat/fout > 10:
        print('statistische fout')
        return (mu, sigma_stat, 'N')
    elif sigma_stat == 0:
        print('fout van het meettoestel')
        return (mu, fout, 'U')
    elif fout/sigma_stat > 10:
        print('fout van het meettoestel')
        return (mu, fout, 'U')
    else:
        return (mu, sigma_stat, fout)
    


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
    mini = minimize(chi2_func, gok, args = (x_val, y_val, y_err, soort_fout, model))
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
        x_as_titels = "Generic", y_as_titels = "Generic", titel = "Generic", printen = "False"): #Veel van deze inputs doen niets, kmoet nog pretty
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
    return outp
    #Werkt nog niet, kmoet de code nog algemeen schrijven :(
    #if printen:
    #    print("##################### Pretty print #####################")
    #    pretty_print_results(x_val, y_val, y_err, chi_min, min_param, betrouwb_int, parameters)


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

def initial_vals_2D(x_val, y_val, initial_vals):
    param_initials = initial_vals(x_val, y_val)
    guess_initials = x_val
    outp = np.concatenate(param_initials, x_val)
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
    gok = initial_vals_2D(x_val, y_val, n_param, initial_vals)
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
            outp = np.append(outp, chi2_bereken_2D(var, x_val, y_val, x_variance, y_variance, model, n_param) - chi2.ppf(0.68, df=n_param) - chi_min)
        return outp
    except TypeError as te:
        kopie = np.copy(hybrid)
        #De waarde van var wordt op de juiste index ingevuld in de parameter vector.
        np.put(kopie, ind_var, var)
        outp = chi2_bereken_2D(var, x_val, y_val, x_variance, y_variance, model, n_param) - chi2.ppf(0.68, df=n_param) - chi_min
        return outp

def find_sigma_values_2D(x_val, y_val, x_variance,  y_variance, hybrid, te_checken_param_ind, chi_min, model, n_param):
    functie = lambda *args: chi2_in_1_var(*args)
    gok = hybrid[te_checken_param_ind]
    #De snijpunten met de 1\sigma hypercontour van de chi^2_mu verdeling zullen rond de best fittende waardes liggen
    i = 0.1
    terminate = False
    while i<=2 and not terminate:
        #scipy.optimize.fsolve vindt de nulpunten van de gegeven functie, chi2_in_1_var is gedefinieerd zodat de gezochte boven
        #en ondergrenzen van het BI precies de nulpunten zijn.
        #Om de bovengrens te vinden wordt een initiële waarde boven de best fittende waarde genomen, omgekeerd voor de ondergrens.
        try:
            sol_left = root_scalar(functie, args = (te_checken_param_ind, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param), method = "brentq", bracket = [gok*(1-i), gok], x0 = gok, x1 = (1-i)*gok)
            sol_right = root_scalar(functie, args = (te_checken_param_ind, x_val, y_val, x_variance, y_variance, hybrid, chi_min, model, n_param), method = "brentq", bracket = [gok, gok*(1+i)], x0 = gok, x1 = (1+i)*gok)
            return [sol_left.root, sol_right.root]
        except:
            if i != 2:
                i+=0.1
            else:
                terminate = True
                print("Geen fout gevonden in 200% foutenmarge")
                return [0, 0]

def uncertainty_intervals_2D(min_hybrid, x_val, y_val, x_variance, y_variance,  chi_min, model, n_param):
    intervallen = []
    for i in range(0, n_param):
        intervallen.append(find_sigma_values_2D(x_val, y_val, x_variance, y_variance, min_hybrid, i, chi_min, model, n_param))
    return intervallen

def jackknife_errors():
    pass
def fit_2D(parameters, model, initial_vals, x_val, y_val, x_variance, y_variance, 
        x_as_titel = "X-as", y_as_titel = "Y-as", titel = "Fit", error_method = "Old", detailed_logs = False): #Veel van deze inputs doen niets, kmoet nog pretty
    #print code schrijven
    #TODO: cas_matrix support maken
    #TODO: ML code schrijven
    n_param = len(parameters)
    print("Raw output")
    mini = minimize_chi2_2D(model, initial_vals, x_val, y_val, y_variance, x_variance, n_param)
    chi_min = mini["fun"]
    min_hybrid = mini["x"]
    min_param = min_hybrid[:n_param]
    if detailed_logs: 
        print(mini)
    
    if error_method == "Old":
        betrouwb_int = uncertainty_intervals_2D(min_hybrid, x_val, y_val, x_variance, y_variance, chi_min, model, n_param)
        print(betrouwb_int)
        foutjes = []
        for i in range(0, n_param):
            top = betrouwb_int[i][1] - min_param[i]
            bot = min_param[i] - betrouwb_int[i][0]
            foutjes.append((bot, top))
            outp = parameters[i] + " heeft als waarde: %.5g + %.5g - %.5g met 68%% betrouwbaarheidsinterval: [%.5g, %.5g] "%(min_param[i], top, bot, betrouwb_int[i][0], betrouwb_int[i][1])
            print(outp)

        nu = len(x_val) - n_param
        p_waarde = chi2.sf(chi_min, df=nu)
        chi_red = chi_min/nu
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
        return outp
    elif error_method == "Jackknife":
        pass
    else:
        print("Given error method not yet implemented, try ""Jackknife"" instead.")
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

def latex_print_tabel(meetwaarden, namen):
    """
    namen = lijst [param1naam, param2naam, param3naam, ...], de hoofdingen van de tabel
    meetwaarden = matrix [ [param1meting1, param2meting1, param3meting1,...],
                           [param1meting2, param2meting2, param3meting3,...],
                        ]
                met elke paramimetingj = [waarde, fout] (het standaard formaat)
    """


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
    #meetwaarden is de standaard vector [waarde, fout] met eventueel 'U'/'N' erachter
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
        print(fout, exponent)
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
def vector_to_datapunt(vector, variabele):
    """
    Input: een vector in dataformaat, en de variabele die het representeert
    Output: het datapunt van deze vector
    """
    if type(variabele) != sp.core.symbol.Symbol:
            raise TypeError
    waarde, fout, soort_fout = vector
    return classes.datapunt(waarde, fout, variabele, soort_fout)


def matrix_to_datapunten(matrix, variabele):
    """
    Input: een matrix in dataformaat, en de variabele die zij representeren
    Output: een lijst (in de juiste volgorde) van datapunten van deze datamatrix
    """
    if type(variabele) != sp.core.symbol.Symbol:
            raise TypeError("variabele moet een sp.symbol zijn")
    datapunten = []
    for vector in matrix:
        datapunten.append(vector_to_datapunt(vector, variabele).copy())
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

    