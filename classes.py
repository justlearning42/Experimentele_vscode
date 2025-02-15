from math import sqrt
import numpy as np
import sympy as sp
import sympy.stats as stats
from matplotlib import pyplot as plt
from scipy.optimize import minimize, fsolve
from scipy.stats import chi2
import functies
import numbers
from IPython.display import display



class vergelijking:
    def __init__(self, sympy_formule, parameters = None, constants = None):
        self.formule = sympy_formule
        if parameters == None and constants == None:
            self.param = self.formule.free_symbols
            self.constants = set()
        elif parameters == None:
            self.constants = set(constants)
            form_parameters = self.formule.free_symbols
            self.param = form_parameters.difference(self.constants)
        elif constants == None:
            form_parameters = self.formule.free_symbols
            self.param = set(parameters).intersection(form_parameters)
            self.constants = form_parameters.difference(self.param)
        else:
            self.param = set(parameters)
            self.constants = set(constants)
    ################################
    #### Bewerkingen definiëren ####
    ################################
    def __add__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule + other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule + other, parameters=self.param)
        return NotImplemented
    def __sub__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule - other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule - other, parameters=self.param)
        return NotImplemented
    def __mul__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule * other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule * other, parameters=self.param)
        return NotImplemented
    def __div__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule / other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule / other, parameters=self.param)
        return NotImplemented
    def __pow__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule ** other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule ** other, parameters=self.param)
        return NotImplemented
    def __radd__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule + other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule + other, parameters=self.param)
        return NotImplemented
    def __rsub__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule - other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule - other, parameters=self.param)
        return NotImplemented
    def __rmul__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule * other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule * other, parameters=self.param)
        return NotImplemented
    def __rdiv__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule / other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule / other, parameters=self.param)
        return NotImplemented
    def __rpow__(self, other):
        if isinstance(other, vergelijking):
            return vergelijking(self.formule ** other.formule, parameters=self.param.union(other.param))
        elif isinstance(other, numbers.Number) or isinstance(other, sp.Symbol):
            return vergelijking(self.formule ** other, parameters=self.param)
        return NotImplemented

    #################################
    #### Samenwerking met andere ####
    ####        functies         ####
    #################################
    def __repr__(self):
        return str(self.formule)
    def _sympy_(self):
        return self.formule
    def copy(self):
        return vergelijking(self.formule, self.param, self.constants)
    #################################
    #### Eigenschappen aanpassen ####
    #################################
    def set_constant(self, parameters):
        try:
            iterator = iter(parameters)
        except TypeError:
            if isinstance(sp.Symbol, parameters) and parameters in self.formule.free_symbols:
                self.constants += set(parameters)
                self.param -= set(parameters)
            else:
                raise TypeError
        else:
            for val in parameters:
                if val in self.formule.free_symbols:
                    self.constants += set(parameters)
                    self.param -= set(parameters)
                else:
                    raise "Parameters moet bestaan uit parameters van de vergelijking."

    def set_to_param(self, constants):
        try:
            iterator = iter(constants)
        except TypeError:
            if isinstance(sp.Symbol, constants) and constants in self.formule.free_symbols:
                self.param += set(constants)
                self.constants -= set(constants)
            else:
                raise TypeError
        else:
            for val in constants:
                if val in self.formule.free_symbols:
                    self.param += set(constants)
                    self.constants -= set(constants)
                else:
                    raise "constants moet bestaan uit constanten van de vergelijking."

    #########################
    #### Dingen invullen ####
    #########################
    def subs(self, substituties):
        """
        substituties van hetzelfde formaat als sp.subs()
        veranderd de interne representatie
        """
        variabelen = set()
        if substituties[0] == ():
            return self
        for substitutie in substituties:
            if substitutie[0] not in self.formule.free_symbols:
                #raise "Substitueer enkel waardes in de vgl"
                pass #fuck de errors
            else:
                variabelen.add(substitutie[0])
            self.formule = self.formule.subs(substitutie[0], substitutie[1])
        self.param -= variabelen
        self.constants -= variabelen
        return self
    
    def calculate(self, substituties):
        """
        substituties: een lijst [(Symbol, value),...] die de symbolen geeft met de waarde die ze moeten bevatten
        @return: de waarde die de vergelijking zou opleveren met de gegeven substituties
        """
        variabelen = set()
        formule = self.formule
        for substitutie in substituties:
            formule = formule.subs(substitutie[0], substitutie[1])
        self.param -= variabelen
        return formule

    def evaluate(self, parameter_vals, constant_vals, eval_name):
        """
        parameter_vals = np array van datapunt objecten met zelfde lengte als self.param. Als dit een lijst van lijsten is wordt return ook een lijst teruggegeven
        constant_vals = np array van getallen met de symbolen waarvoor ze de waarde zijn in koppels, zelfde formaat als sp.subs()

        return: een (np array van) datapunt objecten verkregen door self.formule te evalueren met foutenpropagatie
        """
        if type(parameter_vals[0]) == datapunt:
            te_evalueren = self.subs(constant_vals)
            outp = functies.data_analyse(te_evalueren, parameter_vals, eval_name)
            return outp
        else:
            print("Nog niet geimplementeerd")
            return None

    ################
    #### Fitten ####
    ################
    def fit_dataset(self, dataset, model, initial_vals):
        # initial vals is v.d. vorm [(val, name)]
        return functies.fit_model(model, dataset, initial_vals)
        

class datapunt:
    def __init__(self, waarde, fout, variabele, verdeling = "Normaal"):
        #Fout is wat we geven als \pm, voor uniform kan ook (resolutie, nauwkeurigheid)
        self.waarde = waarde
        if type(variabele) != sp.Symbol:
            raise TypeError
        self.naam = variabele
        self.verdeling = verdeling
        if verdeling == "Normaal" or verdeling == "N" or verdeling == 'S':
            if type(fout) == tuple: #eigenlijk mag dit niet
                fout = np.sqrt(abs(fout[0])**2+fout([1])**2) #kwadratisch gemiddelde nemen
            self.pmfout = fout
            self.variance = fout**2
        elif verdeling == "Uniform" or verdeling == "U":
            if type(fout) == tuple:
                self.pmfout = (fout[0] + fout[1]*waarde)/2
                self.variance = (self.pmfout**2)/3
            else:
                self.pmfout = fout
                self.variance = (fout**2)/3
        else:
            print("Andere verdelingen nog niet geïmplementeerd, fout als wortel van variantie genomen")
            self.variance = fout**2
    
    def __str__(self):
        '''
        output: [waarde, sigma, type_fout]
        
        '''
        return 'datapunt: ' + str(self.naam) + " = " +str([self.waarde, self.pmfout, self.verdeling])
    def __repr__(self):
        return self.__str__()

    def get_naam(self):
        return self.naam
    def get_val(self):
        return self.waarde
    def get_fout(self):
        return self.pmfout
    def get_latex(self):
        return functies.latex_print_meting(functies.datapunt_to_vector(datapunt))

    def get_verdeling(self):
        return self.verdeling
    def get_variance(self):
        return self.variance
    
class meting:
    def __init__(self):
        pass

class dataset:
    def __init__(self, punten, fouten, namen, verdelingen):
       """
       punten = Matrix (np array) van datapunten
       fouten = Matrix (np array) van fouten in dezelfde vorm als punten
                alternatief: fouten een np array met len(punten[0]) (=hoeveelheid param), dan krijgt alles dezelfde fout
       namen = np array van namen, heeft lengte van het aantal verschillende param
       verdelingen = idem als bij fouten

       maakt: een matrix van datapunt objecten: [[datapunt(punten[i][j], fouten[i][j], namen[j], verdelingen[i][j])]]
       of [[datapunt(punten[i][j], fouten[j], namen[j], verdelingen[j])]]
       """ 
       self.punten = np.copy(punten)
       self.fouten = np.copy(fouten)
       self.namen = np.copy(namen)
       self.verdelingen = np.copy(verdelingen)
       matrix = []
       for i in range(len(self.punten)):
           rij = []
           for j in range(len(self.punten[i])):
                nieuw_punt = [self.punten[i][j]]
                if fouten.ndim == 1:
                    nieuw_punt.append(self.fouten[j])
                else:
                    nieuw_punt.append(self.fouten[i][j])
                if namen.ndim == 1:
                    nieuw_punt.append(self.namen[j])
                else:
                    nieuw_punt.append(self.namen[i][j])
                if verdelingen.ndim == 1:
                    nieuw_punt.append(self.verdelingen[j])
                else:
                    nieuw_punt.append(self.verdelingen[i][j])
                nieuw_datapunt = datapunt(*nieuw_punt)
                rij.append(nieuw_datapunt)
           matrix.append(rij)
       self.representatie = np.array(matrix)
       self.metingen = [meting(self.representatie[i]) for i in range(len(self.representatie))]
    
       

       
