#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:10:06 2021

@author: leonardo
"""

#  import a few packages:
import math as m
import json
import numpy as np
import pandas as pd
import networkx as nx
import cartopy.crs as ccrs
from cartopy.feature import LAND
import matplotlib.pyplot as plt
from IPython.display import Image
from lmfit import minimize, Parameters, Parameter, report_fit, Model, Minimizer
from scipy.integrate import odeint
from scipy import interpolate
from scipy.interpolate import interp1d
import pickle
from datetime import datetime, timedelta
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def GetDataSets():
    """
    This function don't take any argument as entry, just download from repositories online
    the data abput cases, deaths, recovred and vaccinated popoulation
    Return a dictionary with each counttry time series with the next information:
        Name of the country
       Edpidemic Data : 'Time','Active','Confirmed','Deaths','Recovered'
       Vaccination data 'Dates vaccination','Vaccines per millon'
    """
    DataCases=pd.read_csv(
        'https://github.com/CSSEGISandData/COVID-19/blob/master/'
        'csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv?raw=true',
        header=0)
    DataDeaths = pd.read_csv(
            'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/'
            'time_series_covid19_deaths_global.csv?raw=true',
            header=0)
    DataRecovered = pd.read_csv(
            'https://github.com/CSSEGISandData/COVID-19/blob/master/csse_covid_19_data/csse_covid_19_time_series/'
            'time_series_covid19_recovered_global.csv?raw=true',
            header=0)
    DataVaccination = pd.read_csv(
            'https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv?raw=true',
            header=0)    
    # Correct data for United States
    DataCases['Country/Region'][DataCases['Country/Region']=='US']='United States'
    DataDeaths['Country/Region'][DataDeaths['Country/Region']=='US']='United States'
    DataRecovered['Country/Region'][DataRecovered['Country/Region']=='US']='United States'
    
     # Correct data for Taiwan
    DataCases['Country/Region'][DataCases['Country/Region']=='Taiwan*']='Taiwan'
    DataDeaths['Country/Region'][DataDeaths['Country/Region']=='Taiwan*']='Taiwan'
    DataRecovered['Country/Region'][DataRecovered['Country/Region']=='Taiwan*']='Taiwan'
    
    
    DataCases.to_csv ('EpiData/Cases_By_Country.csv', index = False, header=True, sep=',')
    DataDeaths.to_csv ('EpiData/Deaths_By_Country.csv', index = False, header=True, sep=',')
    DataRecovered.to_csv ('EpiData/Recovered_By_Country.csv', index = False, header=True, sep=',')
    DataVaccination.to_csv ('VacData/Vaccination_By_Country.csv', index = False, header=True, sep=',')
    
    Columnas=DataCases.columns
    Countries=pd.unique(DataCases['Country/Region'])
    FullData={'Country': [],
              'Data':[],
              'Data Vac':[]}
    for i in Countries:
        
        Country=i
        
        datei=pd.to_datetime(Columnas[4])
        datef=pd.to_datetime(Columnas[len(Columnas)-1])
        time=pd.to_datetime(np.arange(datei, datef, dtype='datetime64[D]'))
        
        Confirmed=DataCases[ (DataCases['Country/Region']==Country)].fillna(0)#.to_numpy()
        Confirmed['Province/State']=''
        Confirmedsum=Confirmed.groupby(['Country/Region'],as_index = False).sum()
        Confirmedsum['Lat']=Confirmed['Lat'].mean()
        Confirmedsum['Long']=Confirmed['Long'].mean()
        Confirmed=Confirmedsum.to_numpy()
        Confirmed=Confirmed[len(Confirmed)-1][3:len(Confirmed[0])-1]
        Confirmed=Confirmed.astype(float).reshape(Confirmed.size,1)
        
       
        
        Deaths=DataDeaths[DataDeaths['Country/Region']==Country]#.to_numpy()
        Deaths['Province/State']=''
        DeathsSum=Deaths.groupby(['Country/Region'],as_index = False).sum()
        DeathsSum['Lat']=Deaths['Lat'].mean()
        DeathsSum['Long']=Deaths['Long'].mean()
        Deaths=DeathsSum.to_numpy()
        Deaths=Deaths[len(Deaths)-1][3:len(Deaths[0])-1]
        Deaths=Deaths.astype(float).reshape(Deaths.size,1)
        locations = np.where(np.diff(Deaths.reshape(len(Deaths))) != 0)[0] + 1
        result = np.split(Deaths.reshape(len(Deaths)), locations)
        for i in np.arange(len(result)):
            # print(len(result[i]))
            if len(result[i])>=15:
                if i-5>=0:
                    relation=np.mean(Confirmed[locations[i-5]:locations[i-1]]/Deaths[locations[i-5]:locations[i-1]])
                    Deaths[locations[i-1]:len(Deaths)]=Confirmed[locations[i-1]:len(Confirmed)]/relation
                    Recovered=Confirmed/relation
        
        Recovered=DataRecovered[DataRecovered['Country/Region']==Country]#.to_numpy()
        Recovered['Province/State']=''
        Recoveredsum=Recovered.groupby(['Country/Region'],as_index = False).sum()
        Recoveredsum['Lat']=Recovered['Lat'].mean()
        Recoveredsum['Long']=Recovered['Long'].mean()
        Recovered=Recoveredsum.to_numpy()
        Recovered=Recovered[len(Recovered)-1][3:len(Recovered[0])-1]
        Recovered=Recovered.astype(float).reshape(Recovered.size,1)
        locations = np.where(np.diff(Recovered.reshape(len(Recovered))) != 0)[0] + 1
        result = np.split(Recovered.reshape(len(Recovered)), locations)
        for i in np.arange(len(result)):
            # print(len(result[i]))
            if len(result[i])>=15:
                # print(len(result[i]))
                if i-5>=0:
                    relation=np.mean(Confirmed[locations[i-5]:locations[i-2]]/Recovered[locations[i-5]:locations[i-2]])
                    Recovered[locations[i-1]:len(Recovered)]=Confirmed[locations[i-1]:len(Confirmed)]/relation
        
        if sum(Confirmed)/sum(Recovered)>=20:
               if sum(Recovered)==0:
                   Recovered=Confirmed*.4
               else:
                   prop=(sum(Confirmed)/sum(Recovered))*.4
                   Recovered=Recovered*prop
               
        Vaccinated=DataVaccination['daily_vaccinations_per_million'][DataVaccination['location']==Country].fillna(0).to_numpy()
        VaccinatedDates=DataVaccination['date'][DataVaccination['location']==Country].fillna(0).to_numpy()
        VaccinatedFull=DataVaccination['people_fully_vaccinated'][DataVaccination['location']==Country].fillna(0).to_numpy()
        VaccinatedFullOneDose=DataVaccination['people_vaccinated'][DataVaccination['location']==Country].fillna(0).to_numpy()
        Active=abs(Confirmed-Recovered-Deaths)
        
        Data = {'Time': list(time),
                'Active': Active.reshape((len(Active))), 
                'Confirmed': Confirmed.reshape((len(Confirmed))), 
                'Deaths': Deaths.reshape((len(Deaths))), 
                'Recovered' : Recovered.reshape((len(Recovered)))
                }
        
        DataVac ={'Dates vaccination': VaccinatedDates.reshape(len(VaccinatedDates)),
                 'Vaccines per millon': Vaccinated.reshape(len(Vaccinated)),
                 'Total vaccintaed': VaccinatedFullOneDose.reshape(len(VaccinatedFullOneDose)),
                 'Total full vaccinated': VaccinatedFull.reshape(len(VaccinatedFull))}
        
        Data= pd.DataFrame(Data, columns= ['Time','Active','Confirmed','Deaths',
                                           'Recovered'])
        DataVac= pd.DataFrame(DataVac, columns= ['Dates vaccination','Vaccines per millon','Total vaccintaed','Total full vaccinated'])
        
        FullData['Country'].append(Country)
        FullData['Data'].append(Data)
        FullData['Data Vac'].append(DataVac)
        Name='EpiData/'+Country + '.csv'
        Data.to_csv (Name, index = False, header=True, sep=',')
        Name='VacData/'+Country + '.csv'
        DataVac.to_csv (Name, index = False, header=True, sep=',')
    print('Data have been created in local directories')
    return FullData

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def Make_Network():
    """
    This function don't take any argument.
    As a result returns the final grapsh taking country conections from the rotutes
    and airports databases
    """
    # Load general population info for each country
    Continents= pd.read_csv('Data/Continents.csv',
                        header=0)
    Data = pd.read_csv('Data/WorldPopulation.csv',
                        header=0)
    # Select data for 2020
    Data = Data[['Country Name','Country Code','2018']]
    
    # Load the first dataset containing many flight routes:
    names = ('airline,airline_id,'
             'source,source_id,'
             'dest,dest_id,'
             'codeshare,stops,equipment').split(',')
    
    routes = pd.read_csv(
        'https://github.com/ipython-books/'
        'cookbook-2nd-data/blob/master/'
        'routes.dat?raw=true',
        names=names,
        header=None)
    routes
    # Load the list of countries I want to see
    listapais=pd.read_csv(
        'Data/Countires.csv',
        # names=names,
        header=0)
    
    listapais=listapais[listapais['Lat']!=0]
    listapais=listapais[listapais['Long']!=0]
    listapais['Country'][listapais['Country']=='US']='United States'
    listapais['Country'][listapais['Country']=='Taiwan*']='Taiwan'
    listapais=listapais.drop_duplicates(subset=['Country'],keep='last')
    listapais['Continent']=''

    # Add continents info to the data
    for i in Continents.index:
        listapais['Continent'][listapais['Country']==Continents['Country'][i]]=Continents['Continent'][i]
        
 
    #-------------corregir las coordenadas que estan mal--------------------------
    for i in listapais.index:
        if abs(listapais['Lat'][i])>1000:
            listapais['Lat'][i]=listapais['Lat'][i]/1000
        if abs(listapais['Long'][i])>1000:
            listapais['Long'][i]=listapais['Long'][i]/1000
    # ----------------------------------------------------------------------------
    
    # Create new fields in the routes table in ordert to add country info
    routes['source_country']=''
    routes['dest_country']=''
    #load the second dataset with details about the airports, and we only keep the airports from any cuntry
    
    names = ('id,name,city,country,iata,icao,lat,lon,'
             'alt,timezone,dst,tz,type,source').split(',')
    airports = pd.read_csv(
        'https://github.com/ipython-books/'
        'cookbook-2nd-data/blob/master/'
        'airports.dat?raw=true',
        header=None,
        names=names,
        index_col=4,
        na_values='\\N')
    
    # country_list=pd.unique(airports['country'])
    country_list=listapais['Country']
    airports['lat_country']=0
    airports['lon_country']=0
    airports['alt_country']=0
    # Actualize Pop data from data table 
    Data = Data[Data['Country Name'].isin(country_list)]
    
    print("Loading countries coordinates")
    # -----------ESTABLECER COORDENADAS DE CADA PAÍS-------------------------------
    for i in country_list:
        # print(i)
        # LAT=airports['lat'][airports['country'] == i]
        # LON=airports['lon'][airports['country'] == i]
        # ALT=airports['alt'][airports['country'] == i]
        LAT=listapais['Lat'][listapais['Country']==i]
        LON=listapais['Long'][listapais['Country']==i]
        ALT=airports['alt'][airports['country'] == i]
        # print('Latitud: ', LAT.mean(),'. Longitud: ',LON.mean(),' Altitud: ',ALT.mean())
        airports['lat_country'][airports['country'] == i]=LAT.mean()
        airports['lon_country'][airports['country'] == i]=LON.mean()
        airports['alt_country'][airports['country'] == i]=ALT.mean()
        # for j in airports:
    # -----------------------------------------------------------------------------
    
    print('Adding countries coordinates on plain routes')
    for i in airports.index.dropna():
    # for i in country_list:
        # print(i)
        routes['source_country'][routes['source']==i]=airports.country[i]
        routes['dest_country'][routes['dest']==i]=airports.country[i]
    
    airports = airports[airports['country'].isin(country_list)]
    
    reduced_rutes=pd.DataFrame(columns=routes.columns)
    reduced_rutes=routes
    reduced_rutes.dropna(subset=['source_country'], inplace=True)
    reduced_rutes.dropna(subset=['dest_country'], inplace=True)
    reduced_rutes=reduced_rutes[reduced_rutes['source_country'] != '']
    reduced_rutes=reduced_rutes[reduced_rutes['dest_country'] != '']
    reduced_rutes=reduced_rutes[reduced_rutes['source_country'] != reduced_rutes['dest_country']]
    
    print('Generating routes country keys')
    reduced_rutes['origing/dest']=''
    for i in reduced_rutes.index:
          reduced_rutes['origing/dest'][i]=reduced_rutes['source_country'][i]+"/"+ reduced_rutes['dest_country'][i]
    
    reduced_rutes = reduced_rutes[reduced_rutes['source_country'].isin(country_list)]
    reduced_rutes = reduced_rutes[reduced_rutes['dest_country'].isin(country_list)]
    
    # count the number of routes from country x To y
    print('Creating network edges weigths')
    reduced_rutes['Routes']=0
    indicerutas=pd.unique(reduced_rutes['origing/dest'])
    for i in indicerutas:
        # print(i)
        subset_routes = reduced_rutes[reduced_rutes['origing/dest'] == i]
        Number = subset_routes.count()['origing/dest']
        reduced_rutes['Routes'][reduced_rutes['origing/dest'] == i]=Number
    
        
    airports_us=airports
    
    routes_us=reduced_rutes
    routes=reduced_rutes
    # Vuelos internacionales que salen del país X
    routes_us = routes[
        routes['source'].isin(airports_us.index) &
        routes['dest'].isin(airports.index)]
    routes_us
    
    
    edges = routes_us[['source_country', 'dest_country']].values
    edges
    
    # g = nx.DiGraph()
    print('Creating Network')
    g = nx.Graph()
    
    g = nx.from_pandas_edgelist(routes_us, 'source_country', 'dest_country',edge_attr='Routes')
    
    Nodos = airports.reset_index().drop_duplicates(subset='country', keep='last').set_index('country').sort_index()
    Nodos = Nodos[['id', 'lat_country','lon_country','alt_country']]
    Nodos['Population']=0
    Nodos['Country_Code']='XXX'
    Nodos['Name']=Nodos.index
    Nodos['Continent']=''
    # Fill the fields od population and country code
    for i in Nodos.index:
        NodePop=Data[Data['Country Name']==i]['2018'].values
        
        CC=Data[Data['Country Name']==i]['Country Code'].values
        if len(NodePop) == 0:
            NodePop=0;
        if len(CC) == 0:
            CC='XXX'
        Nodos['Population'][i]=NodePop
        CC=str(CC)
        CC=CC.replace('[','')
        CC=CC.replace(']','')
        Nodos['Country_Code'][i]=CC
        Continente=str(listapais['Continent'][listapais['Country']==i].values)
        Continente=Continente.replace('[','')
        Continente=Continente.replace(']','')
        Nodos['Continent'][i]=Continente    
    print('Seting network attributes')
    nx.set_node_attributes(g, Nodos.to_dict('index'))
    
    return g , airports
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PlotNetwork(g):
    """
    This function takes as input argument a network elemet g and print the networ nodes
    and connections
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    nx.draw_networkx(g, ax=ax, node_size=20,
                 font_size=10, alpha=.5,
                 width=1)
    ax.set_axis_off()
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def ReduceNetwork(g):
    """
    This function takes a network as input and return a reduced netork sg
    """
    sg = next(g.subgraph(c) for c in nx.connected_components(g))
    return sg
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def GetListOfCountries(g):
    """
    This function retunr the list of nodes of the network g
    """
    paisesfinales=list()
    for n in sorted(g._node):
        paisesfinales.append(n)
    return pd.unique(paisesfinales)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PlotReducedNetwork(g):
    """
    This function takes as input argument a network elemet g and print the networ nodes
    and connections
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    nx.draw_networkx(g, ax=ax, with_labels=False,
                 node_size=10, width=1)
    ax.set_axis_off()
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PlotMap(airports,g):
    """
    This function takes as input argument a network elemet g, a list of airports and print the networ nodes
    and connections in a map
    """
    airports_us2 = airports.reset_index().drop_duplicates(subset='country', keep='last').set_index('country').sort_index()

    pos = {airport: (v['lon_country'], v['lat_country'])
           for airport, v in
          airports_us2.to_dict('index').items()}
    
    deg = nx.degree(g)
    sizes = [5 * deg[iata] for iata in g.nodes]
    
    altitude = airports_us2['alt_country']
    altitude = [altitude[iata] for iata in g.nodes]
    
    
    
    labels = {iata: iata if deg[iata] >= 20 else ''
              for iata in g.nodes}
    
    # Map projection
    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(
        1, 1, figsize=(12, 8),
        subplot_kw=dict(projection=crs))
    ax.coastlines()
    ax.stock_img()
    # Extent of continental US.
    # ax.set_extent([-128, -62, 20, 50])
    # ax.set_extent([-180, -90, 180, 90])
    nx.draw_networkx(g, ax=ax,
                     font_size=16,
                     alpha=.5,
                     width=0.5,
                     node_size=sizes,
                     labels=labels,
                     pos=pos,
                     node_color=altitude,
                     cmap=plt.cm.autumn)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def ShowCountryConections(g):
    """
    This function show countries connections showing the connection and the number of routes 
    """
    for n in sorted(g._node):
        print('País: ',n, '. Conexiones: ')
        u=n
        for m in g._adj[n]:
            v=m
            print('\t\t Vecino: ',g._node[m]['Name'],'. Rutas: ', g.get_edge_data(u, v, default=None)['Routes'])   
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def ShowNodeInfo(g):
    """
    This function show node info such as population size, country name and continent
    Parameters
    ----------
    g : NetworkX graph
        Network of countries.

    Returns
    -------
    None.

    """
    for n in sorted(g._node):
        print('Country: ',n)
        print('\t\t(*) Population: ',g._node[n]['Population'])
        print('\t\t(*) Code: ',g._node[n]['Country_Code'])
        print('\t\t(*) Continent: ',g._node[n]['Continent'])
        
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def GetNodeInformation(g):
    """
    This function return the node info such as population size, country name and continent
    input:
        g: the full graph
    output:
        Data: Country data frame contaning: 
            * population size
            * country code
            * continent
            * Lat
            * Long
            * Alt
    """
    ListCountry=list()
    LisPop=list()
    ListCC=list()
    ListContinent=list()
    ListLat=list()
    ListLon=list()
    ListAlt=list()
    for n in sorted(g._node):
        ListCountry.append(n)
        LisPop.append(g._node[n]['Population'])
        ListCC.append(g._node[n]['Country_Code'])
        ListContinent.append(g._node[n]['Continent'])
        ListLat.append(g._node[n]['lat_country'])
        ListLon.append(g._node[n]['lon_country'])
        ListAlt.append(g._node[n]['alt_country'])
    Data ={'Country': ListCountry,
        'Population': LisPop,
        'Code': ListCC,
        'Continent':ListContinent, 
        'Lat': ListLat,
        'Lon': ListLon,
        'Alt': ListAlt}    
    Data= pd.DataFrame(Data, columns= ['Country','Population','Code','Continent','Lat','Lon','Alt'])
    return Data   
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def ShowVaccination(g):
    """
    This function show node info such abput vaccinated countries. 
    As entry gets the graph

    Parameters
    ----------
    g : graph
        Network Graph.

    Returns
    -------
    VaccinData : Pandas Data Frame
        Shows the vaccination data for the countrys in the network. The data frame 
        hold the relevant information: 
            'Country' Name of the country
            'Rate' Vaccination rate per millon
            'Total Full vaccinated' Total number of fully vaccinated population
            'Days Vaccination' Days since the vaccination begun      

    """

    dfNet=pd.DataFrame.from_dict(dict(g.nodes(data=True)), orient='index')
    ListVac=list()
    LisRate=list()
    LisTotal=list()
    DiasVac=list()
    ListFirsDose=list()
    Population=list()
    for i in dfNet.index:
        name='VacData/'+i+'.csv'
        VacData=pd.read_csv(name, header=0)
        if VacData.empty == False:
            # rate=VacData['Vaccines per millon'].mode().mean()/(1e6)
            rate= VacData['Vaccines per millon'][len(VacData['Vaccines per millon'])-
                                                 21:len(VacData['Vaccines per millon'])-1].mean()/(1e6)
            total=VacData['Total full vaccinated'][len(VacData['Total full vaccinated'])-1]
            firstdose=VacData['Total vaccintaed'][len(VacData['Total vaccintaed'])-1]
            LisRate.append(rate)
            LisTotal.append(total)
            ListVac.append(i)
            ListFirsDose.append(firstdose)
            DiasVac.append(len(VacData['Vaccines per millon']))
            Population.append(dfNet['Population'][i])
   
    VaccinData ={'Country': ListVac,
                 'Rate': LisRate,
                 'Total Full vaccinated': LisTotal,
                 'Total vaccintaed':ListFirsDose,
                 'Days Vaccination':DiasVac,
                 'Population': Population}
     # DataVac ={'Dates vaccination': VaccinatedDates.reshape(len(VaccinatedDates)),
     #             'Vaccines per millon': Vaccinated.reshape(len(Vaccinated)),
     #             'Total vaccintaed': VaccinatedFullOneDose.reshape(len(VaccinatedFullOneDose)),
     #             'Total full vaccinated': VaccinatedFull.reshape(len(VaccinatedFull))}    
    VaccinData= pd.DataFrame(VaccinData, columns= ['Country','Rate','Total vaccintaed',
                                                   'Total Full vaccinated','Days Vaccination',
                                                   'Population'])
    return VaccinData   
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def GetCountriinfo(n,days):
    """
    This function return the epidemiological information about an specific country

    Parameters
    ----------
    n : name of the country
        

    Returns
    -------
    EpiData : A data frame contaning the tim vector, active cases, recovered, deaths 
    and total reported cases from data

    """
    name='EpiData/'+n+'.csv'
    nameT='Temp Data/Cuntries/'+n+'.csv'
    EpiData=pd.read_csv(name, header=0)
    EpiData=EpiData[len(EpiData)-(days+1):len(EpiData)-1]
    # datei=pd.to_datetime(EpiData['Time'].iloc[0])
    # datef=pd.to_datetime(EpiData['Time'].iloc[-1])
    TempData=pd.read_csv(nameT, header=0)
    TempData['Date']=pd.to_datetime(TempData['Date'])
    TempData['Date']+= timedelta(days=366)
    Temperature=TempData
    # Temperature=TempData[TempData['Date'].isin(pd.to_datetime(EpiData['Time']))]['Temperature']
    # Temperature=TempData[TempData['Date']>=datei]
    
    # EpiData['Temperature']=Temperature.values
    return EpiData,Temperature
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PopulationSize(n,CounttriesData):
    """
    This funcion returns the poulation size of the country n
    
    Parameters
    ----------
    n : Str
        Name of the country.
    CounttriesData : Pandas Data Frame
        Data frame with the full countries information .

    Returns
    -------
    Pop : int
        Population size.

    """
    Pop=CounttriesData['Population'][CounttriesData['Country']==n]
    return Pop
    
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def T_inv(T,timeT):
    """
    

    Parameters
    ----------
    T : TYPE
        DESCRIPTION.
    timeT : TYPE
        DESCRIPTION.

    Returns
    -------
    list
        DESCRIPTION.

    """

    T=pd.concat([T,T])
    That = savgol_filter(T, 51, 3) # window size 51, polynomial order 3

    # pos=timeT.iloc[0].day+30*timeT.iloc[0].month-30
    pos=timeT[ini].day+30*timeT[ini].month-30
    Tinv=1-T/np.mean(T)+1.5
    Tinv=Tinv[pos:len(Tinv)-1]
    
    Tinvhat=1-That/np.mean(That)+1.5
    Tinvhat=Tinvhat[pos:len(Tinvhat)-1]
    
    t=np.arange(len(Tinv))
  
    Tinv=Tinvhat

    return [t,Tinv]

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def SEIQRDP(t, y, dt, ps):   
    """
    This is the discrete random time function  for fitting propopuse

    Parameters
    ----------
    t : array
        time vector.
    y : array
        inititial conditions of the system.
    ps : parameters
        Parameters for the model.

    Returns
    -------
    a Matrix with the population dynamics

    """
    delta   = ps['delta'].value
    nuV     = ps['nuV'].value
    alpha  = ps['alpha0'].value*m.exp(-ps['alpha1'].value*t)
    # betQ    = ps['betaQ'].value*((1+np.sin(2*np.pi*t*1/360)))*1/2
    # bet     = ps['beta'].value*((1+np.sin(2*np.pi*t*1/360)))*1/2
    betQ    = ps['betaQ'].value*float(f(t)) 
    bet     = ps['beta'].value*float(f(t)) 
    # betQ    = ps['betaQ'].value
    # bet     = ps['beta'].value
    gamma   = ps['gamma'].value
    # Lambda  = ps['lambda0'].value
    # kappa   = ps['kappa0'].value
    Lambda = ps['lambda0'].value*(1.-m.exp(-ps['lambda1'].value*t))
    kappa = ps['kappa0'].value*m.exp(-ps['kappa1'].value*t)    
    tau    = ps['tau0'].value*(1.-m.exp(-ps['tau1'].value*t))
    rho1= ps['rho1'].value
    rho2= ps['rho1'].value
    deltaV1 = ps['deltaV1'].value
    deltaV2 = ps['deltaV2'].value
    omega = ps['omega'].value
    S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=y
   # ----------------------check non coherences-------------------------------
    if S<0:
        S=0
    if Ew<0:
        Ew=0
    if EwV<0:
        EwV=0
    if Ewm<0:
        Ewm=0
    if Iw<0:
        Iw=0
    if Iwm<0:
        Iwm=0
    if IwV<0:
        IwV=0
    if Qw<0:
        Qw=0
    if Qwm<0:
        Qwm=0
    if QwV<0:
        QwV=0
    if Rw<0:
        Rw=0
    if Em<0:
        Em=0
    if EmV<0:
        EmV=0
    if Emw<0:
        Emw=0    
    if Im<0:
        Im=0
    if IwV<0:
        ImV=0
    if Imw<0:
        Imw=0
    if Qm<0:
        Qm=0
    if QmV<0:
        QmV=0
    if Qmw<0:
        Qmw=0
    if Rm<0:
        Rm=0
    if D<0:
        D=0
    if Rb<0:
        Rb=0
    if P<0:
        P=0
    if V1<0:
        V1=0
    if V2<0:
        V2=0
#    beta=bet * I/N
    N=sum(y)
    IW=Iw+IwV+Iwm
    QW=Qw+QwV+Qwm
    IM=Im+ImV+Imw
    QM=Qm+QmV+Qmw
    betaw    = (1/N)*(bet*IW + betQ*QW)                     #|infection rate wild
    betam    = omega*(1/N)*(bet*IM + betQ*QM)               #|infection rate mutant
    # ---------------------Probabilities---------------------------------------
    efracw   = abs(1.0 - m.exp(-betaw*dt))                     #|exposed prob wild
    efracm   = abs(1.0 - m.exp(-betam*dt))                     #|exposed prob mutant
    efracVw  = abs(1.0 - m.exp(-nuV*betaw*dt))                 #|exposed prob vaccinated wild
    efracVm  = abs(1.0 - m.exp(-nuV*betam*dt))                 #|exposed prob vaccinated mutant
    ifrac    = abs(1.0 - m.exp(-gamma*dt))                     #|infection prob
    rfrac    = abs(1.0 - m.exp(-Lambda*dt))                    #|recov prob
    pfrac    = abs(1.0 - m.exp(-alpha*dt))                     #|protec prob
    dfrac    = abs(1.0 - m.exp(-kappa*dt))                     #|death prob
    relfrac  = abs(1.0 - m.exp(-tau*dt))                       #|release prob
    repfrac  = abs(1.0 - m.exp(-delta*dt))                     #|detected prob
    vacfrac1 = abs(1.0 - m.exp(-rho1*deltaV1*dt))              #|vac 1 prob
    vacfrac2 = abs(1.0 - m.exp(-rho2*(deltaV1+deltaV2)*dt))    #|vac 2 prob
    
    # ---------------------Transitions----------------------------------------
    exposedw =  np.random.binomial(S,efracw)
    exposedm =  np.random.binomial(S,efracm)
    
    exposedmw =  np.random.binomial(Rw,efracm)
    exposedwm =  np.random.binomial(Rm,efracw)
    
    protected = np.random.binomial(S,pfrac)
    released  = np.random.binomial(P,relfrac)
    
    infectionw = np.random.binomial(Ew,ifrac) 
    infectionm = np.random.binomial(Em,ifrac) 
    
    infectionwm = np.random.binomial(Ewm,ifrac) 
    infectionmw = np.random.binomial(Emw,ifrac)
    
    detectedw = np.random.binomial(Iw,repfrac)
    detectedm = np.random.binomial(Im,repfrac)
    
    detectedwm = np.random.binomial(Iwm,repfrac)
    detectedmw = np.random.binomial(Imw,repfrac)
    
    recoveryw = np.random.binomial(Qw,rfrac)
    recoverym = np.random.binomial(Qm,rfrac)
    
    recoverywm = np.random.binomial(Qwm,rfrac)
    recoverymw = np.random.binomial(Qmw,rfrac)
    
    deathsw = np.random.binomial(Qw,dfrac)
    deathsm = np.random.binomial(Qm,dfrac)
    
    deathswm = np.random.binomial(Qwm,dfrac)
    deathsmw = np.random.binomial(Qmw,dfrac)

    vaccinated1  = np.random.binomial(S,vacfrac1)
    vaccinated2  = np.random.binomial(V1,vacfrac2)
    VaccinatedP  = np.random.binomial(P,vacfrac1)
    
    exposedVw = np.random.binomial(V1,efracVw)
    exposedVm = np.random.binomial(V1,efracVm)
        
    infectionVw = np.random.binomial(EwV,ifrac) 
    infectionVm = np.random.binomial(EmV,ifrac) 
    
    detectedVw  = np.random.binomial(IwV,repfrac)
    detectedVm  = np.random.binomial(ImV,repfrac)
    
    recoveryVw  = np.random.binomial(QwV,rfrac)
    recoveryVm  = np.random.binomial(QmV,rfrac)
    # ---------------------Model-----------------------------------------------
    S   = S + released - exposedw - exposedm - protected - vaccinated1      #| Susceptible Wild
    # ---------------------Wild virus------------------------------------------
    Ew  = Ew  + exposedw    - infectionw                                    #| Exposed Wild
    EwV = EwV + exposedVw   - infectionVw                                   #| Exposed Wild Vac
    Ewm = Ewm + exposedwm   - infectionwm
    Iw  = Iw  + infectionw  - detectedw                                     #| Infected Wild
    Iwm = Iwm + infectionwm - detectedwm
    Qw  = Qw  + detectedw   - recoveryw  - deathsw                          #| Detected Wild
    Qwm = Qwm + detectedwm  - recoverywm - deathswm
    IwV = IwV + infectionVw - detectedVw                                    #| Infected Wild Vac
    QwV = QwV + detectedVw  - recoveryVw                                    #| Detected Wild Vac
    Rw  = Rw  + recoveryw   + recoveryVw - exposedmw                        #| Recovered Wild
    # ---------------------Mutant virus----------------------------------------
    Em  = Em  + exposedm    - infectionm                                    #| Exposed Mutant
    EmV = EmV + exposedVm   - infectionVm                                   #| Exposed Mutant Vac
    Emw = Emw + exposedmw   - infectionmw
    Im  = Im  + infectionm  - detectedm                                     #| Infected Mutant 
    Imw = Imw + infectionmw - detectedmw
    Qm  = Qm  + detectedm   - recoverym - deathsm                           #| Detected Mutant
    Qmw = Qmw + detectedmw  - recoverymw - deathsmw
    ImV = ImV + infectionVm - detectedVm                                    #| Infected Mutant Vac
    QmV = QmV + detectedVm  - recoveryVm                                    #| Detected Mutant Vac
    Rm  = Rm  + recoverym   + recoveryVm - exposedwm                                     #| Recovered Mutant
    # ---------------------Common states---------------------------------------
    D   = D   + deathsw + deathsm + deathswm + deathsmw                     #| Deaths
    Rb  = Rb  + recoverywm + recoverymw
    P   = P   + protected - released - VaccinatedP                                          #| Protected
    V1  = V1  + vaccinated1 + VaccinatedP - vaccinated2 - exposedVw - exposedVm           #| Vacccintaed 1 dose
    V2  = V2  + vaccinated2                                                 #| Vacccintaed 2 dose
#    ------------------------------------------
    return [S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,
            Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,
            D,Rb,P,V1,V2]
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def MutantProb(tn,Inf,dt):
    """
    

    Parameters
    ----------
    tn : TYPE
        DESCRIPTION.
    Inf : TYPE
        DESCRIPTION.
    dt : TYPE
        DESCRIPTION.

    Returns
    -------
    NewStrainInf : TYPE
        DESCRIPTION.

    """
    x=np.arange(0,10,(10/(365*1)))
    Rate=1/(1+(365*100)*np.exp(-(x)))
    f = InterpolatedUnivariateSpline(x, Rate, k=1)
    Rate2=f(tn*10/(365*1))
    if Rate2>1:
        Rate2=1
    # f = interpolate.interp1d(x, Rate, fill_value='extrapolate',k=2)
    prob=m.exp(-Rate2*dt)
    # prob=f(tn*10/(365*1))
    
    newstrainprob=(1.0 - prob) 
    # # newstrainprob=f(tn*10/(365*2))*dt
    # newstrainprob=Rate2*dt
    NewStrainInf=np.random.binomial(int(Inf*1e6),newstrainprob)

    return NewStrainInf
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def simulate(t,dt,u,ps): 
    """
    This function simulates the complet system for a country 

    Parameters
    ----------
    t : vector
        Time period to simulate.
    u : vector
        initial conditions of the system.
    ps : parameters
        parameters of the model.

    Returns
    -------
    dict
        population dynamics.

    """
    S= np.zeros(len(t))
    Ew= np.zeros(len(t))
    EwV=np.zeros(len(t))
    Ewm=np.zeros(len(t))
    Iw= np.zeros(len(t))
    IwV= np.zeros(len(t))
    Iwm = np.zeros(len(t))
    Qw= np.zeros(len(t))
    QwV= np.zeros(len(t))
    Qwm = np.zeros(len(t))
    Rw= np.zeros(len(t))
    Em= np.zeros(len(t))
    EmV= np.zeros(len(t))
    Emw = np.zeros(len(t))
    Im= np.zeros(len(t))
    ImV= np.zeros(len(t))
    Imw=np.zeros(len(t))
    Qm= np.zeros(len(t))
    QmV= np.zeros(len(t))
    Qmw=np.zeros(len(t))
    Rm= np.zeros(len(t))
    Rb=np.zeros(len(t))
    D= np.zeros(len(t))
    P= np.zeros(len(t))
    V1= np.zeros(len(t))
    V2= np.zeros(len(t))
# S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=y
    for j in range(len(t)):
        if j>15:            
            # if u[14]==0 and sum(Im)<=100 and t[j]>300:
            if u[14]==0 and sum(Im)<=100:
            # if u[14]==0 and t[j]>150:
            # if u[14]==0:
                # Itot=(Iw+IwV+Qw+QwV)/Npop
                # Inf=(sum(u[4:5])+sum(u[7:8]))/sum(u)
                Inf=(sum(Iw[j-15:j])+sum(IwV[j-15:j]))/sum(u)
                # Inf=(sum(Iw[j-15:j])+sum(IwV[j-15:j]))/sum(u)
                u[14]=MutantProb(t[j],Inf,dt)
                u[4]=u[4]-u[14]
        u = SEIQRDP(t[j],u,dt,ps)
        S[j],Ew[j],EwV[j],Ewm[j],Iw[j],IwV[j],Iwm[j],Qw[j],QwV[j],Qwm[j],Rw[j],Em[j],EmV[j],Emw[j],Im[j],ImV[j],Imw[j],Qm[j],QmV[j],Qmw[j],Rm[j],D[j],Rb[j],P[j],V1[j],V2[j]=u
    return {'t':t,'S':S,'Ew':Ew,'EwV':EwV,'Ewm':Ewm,'Iw':Iw,'IwV':IwV,'Iwm':Iwm,'Qw':Qw,'QwV':QwV,'Qwm':Qwm,'Rw':Rw,
            'Em':Em,'EmV':EmV,'Emw':Emw,'Im':Im,'ImV':ImV,'Imw':Imw,'Qm':Qm,'QmV':QmV,'Qmw':Qmw,'Rm':Rm,
            'D':D,'Rb':Rb,'P':P,'V1':V1,'V2':V2}
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def  simulate_N_times(t,dt,u, ps):
    """
    This function simulates the discrete system 100 times 

    Parameters
    ----------
    t : vector
        Time period.
    u : vector
        initial conditions.
    ps : parameters
        Model parameters.

    Returns
    -------
    Y : Array
        Mean model output.

    """
    times=100
    S= np.zeros(len(t))
    Ew= np.zeros(len(t))
    EwV=np.zeros(len(t))
    Ewm=np.zeros(len(t))
    Iw= np.zeros(len(t))
    IwV= np.zeros(len(t))
    Iwm = np.zeros(len(t))
    Qw= np.zeros(len(t))
    QwV= np.zeros(len(t))
    Qwm = np.zeros(len(t))
    Rw= np.zeros(len(t))
    Em= np.zeros(len(t))
    EmV= np.zeros(len(t))
    Emw = np.zeros(len(t))
    Im= np.zeros(len(t))
    ImV= np.zeros(len(t))
    Imw=np.zeros(len(t))
    Qm= np.zeros(len(t))
    QmV= np.zeros(len(t))
    Qmw=np.zeros(len(t))
    Rm= np.zeros(len(t))
    Rb=np.zeros(len(t))
    D= np.zeros(len(t))
    P= np.zeros(len(t))
    V1= np.zeros(len(t))
    V2= np.zeros(len(t))
    vec=np.zeros(len(t))
    Y={'t':vec,'S':vec,'Ew':vec,'EwV':vec,'Ewm':vec,'Iw':vec,'IwV':vec,'Iwm':vec,'Qw':vec,'QwV':vec,'Qwm':vec,'Rw':vec,
            'Em':vec,'EmV':vec,'Emw':vec,'Im':vec,'ImV':vec,'Imw':vec,'Qm':vec,'QmV':vec,'Qmw':vec,'Rm':vec,
            'D':vec,'Rb':vec,'P':vec,'V1':vec,'V2':vec}
    for i in np.arange(times):
        y = simulate(t, u, ps)
        S[i,:]   = y['S']
        Ew[i,:]  = y['Ew']
        EwV[i,:]  = y['EwV']
        Ewm[i,:]  = y['Ewm']
        Iw[i,:]  = y['Iw']
        IwV[i,:] = y['IwV']
        Iwm[i,:] = y['Iwm']
        Qw[i,:]  = y['Qw']
        QwV[i,:] = y['QwV']
        Qwm[i,:] = y['Qwm']
        Rw[i,:]  = y['Rw']
        Em[i,:]  = y['Em']
        EmV[i,:] = y['EmV']
        Emw[i,:] = y['Emw']
        Im[i,:]  = y['Im']
        ImV[i,:] = y['ImV']
        Imw[i,:] = y['Imw']
        Qm[i,:]  = y['Qm']
        QmV[i,:] = y['QmV']
        Qmw[i,:] = y['Qmw']
        Rm[i,:]  = y['Rm']
        Rb[i,:]  = y['Rb']
        D[i,:]   = y['D']
        P[i,:]   = y['P']
        V1[i,:]  = y['V1']
        V2[i,:]  = y['V2']
    Y['S']= S.mean(0)
    Y['Ew']= Ew.mean(0)
    Y['EwV']= EwV.mean(0)
    Y['Ewm']= Ewm.mean(0)
    Y['Iw']= Iw.mean(0)
    Y['IwV']= IwV.mean(0)
    Y['Iwm']= Iwm.mean(0)
    Y['Qw']= Qw.mean(0)
    Y['QwV']= QwV.mean(0)
    Y['Qwm']= Qwm.mean(0)
    Y['Rw']= Rw.mean(0)
    Y['Em']= Em.mean(0)
    Y['EmV']= EmV.mean(0)
    Y['Emw']= Emw.mean(0)
    Y['Im']= Im.mean(0)
    Y['ImV']= ImV.mean(0)
    Y['Imw']= Imw.mean(0)
    Y['Qm']= Qm.mean(0)
    Y['QmV']= QmV.mean(0)
    Y['Qmw']= Qmw.mean(0)
    Y['Rm']= Rm.mean(0)
    Y['Rb']= Rb.mean(0)
    Y['D']= D.mean(0)
    Y['P']= P.mean(0)
    Y['V1']= V1.mean(0)
    Y['V2']= V2.mean(0)
    return Y
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def interpolation(y,t,ti):
    """
    This function interpolate the discrete function in order to give back the dynamics of the model 
    by day

    Parameters
    ----------
    y : vector
        population y_i.
    t : vector
        time vector.
    ti : vector
        interpolation time vector.

    Returns
    -------
    f2 : vector
        interpolated population y_i.

    """
    f= interpolate.interp1d(t,y, kind='nearest')
    f2 =f(ti)
    return f2
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def Sys_interpolation(Y,t,ti):
    """
    This function makes the interpolation of the system

    Parameters
    ----------
    Y : Array
        full system dynamics.
    t : vector
        time vector.
    ti : vector
        interpolation time vector.

    Returns
    -------
    Yinterp : Array
            interpolated full system dynamics.

    """
    col=Y.columns
    datcol=col[1:len(col)]
    Yinterp={}
    Yinterp['t']=ti
    for i in datcol:
#        print(Y[str(i)])
        yi=Y[str(i)].to_numpy()
        f2=interpolation(yi,t,ti)
        Yinterp[str(i)]=f2

    return Yinterp
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def COVID_SEIRC(y,t,ps):
    """
    Continous time population model for fitting propouse

    Parameters
    ----------
    y : vector
        initial conditions.
    t : vector
        time vector.
    ps : parameters
        model parameters.

    Returns
    -------
    Y: vector
        Model output.

    """
    alpha0=ps['alpha0'].value
    alpha1=ps['alpha1'].value
    bet=ps['beta'].value
    betQ=ps['betaQ'].value
    gam=ps['gamma'].value
    delt=ps['delta'].value
    lamda0=ps['lambda0'].value
    lamda1=ps['lambda1'].value
    kappa0=ps['kappa0'].value
    kappa1=ps['kappa1'].value
    tau0=ps['tau0'].value
    tau1=ps['tau1'].value
    rho1= ps['rho1'].value
    rho2= ps['rho1'].value
    nuV= ps['nuV'].value
    deltaV1 = ps['deltaV1'].value
    deltaV2 = ps['deltaV2'].value
    omega = ps['omega'].value
    # rho=0
    S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=y
    Npop=sum(y)
 #    ______________parameters____________________________
    alpha = lambda t: alpha0*m.exp(-alpha1*t)
    # beta=lambda t:bet*((1+np.sin(2*np.pi*t*1/360)))*1/2
    # betaQ=lambda t:betQ*((1+np.sin(2*np.pi*t*1/360)))*1/2
    betaQ=lambda t:betQ*float(f(t))
    beta=lambda t:bet*float(f(t))
    # beta=lambda t:bet
    # betaQ=lambda t:betQ
    gamma=gam
    delta=delt
    # Lambda = lambda t: lamda0
    # kappa = lambda t: kappa0
    Lambda = lambda t: lamda0*(1.-m.exp(-lamda1*t))
    kappa = lambda t: kappa0*m.exp(-kappa1*t)
    tau= lambda t: tau0*(1.-m.exp(-tau1*t))
    
    IW=Iw+IwV+Iwm
    QW=Qw+QwV+Qwm
    IM=Im+ImV+Imw
    QM=Qm+QmV+Qmw
    
    BETAw=(beta(t)*IW+betaQ(t)*QW)*1/Npop
    BETAm=omega*(beta(t)*IM+betaQ(t)*QM)*1/Npop
    #    ______________Terms____________________________
    exposedw    = BETAw*(S)
    exposedm    = BETAm*(S)
    exposedwm   = BETAw*(Rm)
    exposedmw   = BETAm*(Rw)
    protected   = alpha(t)*S
    released    = tau(t)*P

    infectionw  = gamma*Ew 
    infectionm  = gamma*Em
    infectionwm = gamma*Ewm
    infectionmw = gamma*Emw
    detectedw   = delta*Iw
    detectedm   = delta*Im
    detectedwm  = delta*Iwm
    detectedmw  = delta*Imw
    
    recoveryw   = Lambda(t)*Qw
    recoverym   = Lambda(t)*Qm
    deathsw     = kappa(t)*Qw
    deathsm     = kappa(t)*Qm
    deathswm    = kappa(t)*Qwm
    deathsmw    = kappa(t)*Qmw

    vaccinated1 = rho1*deltaV1*S
    vaccinated2 = rho2*(deltaV1+deltaV2)*V1
    vaccinatedP = rho1*deltaV1*P
    exposedVw   = nuV*BETAw*V1
    exposedVm   = nuV*BETAm*V1
        
    infectionVw = gamma*EwV
    infectionVm = gamma*EmV 
    detectedVw  = delta*IwV 
    detectedVm  = delta*ImV 
    recoveryVw  = Lambda(t)*QwV
    recoveryVm  = Lambda(t)*QmV 
    recoverywm  = Lambda(t)*Qwm
    recoverymw  = Lambda(t)*Qmw
#    ___________equations___________________________________
    # ---------------------Model-----------------------------------------------
    dS =  released - exposedw - exposedm - protected - vaccinated1     #| Susceptible Wild
    # ---------------------Wild virus------------------------------------------
    dEw =  exposedw - infectionw                                      #| Exposed Wild
    dEwV =  exposedVw - infectionVw                                   #| Exposed Wild Vac
    dEwm =  exposedwm - infectionwm
    dIw =  infectionw - detectedw                                     #| Infected Wild
    dIwm =  infectionwm - detectedwm
    dQw =  detectedw - recoveryw - deathsw                            #| Detected Wild
    dQwm =  detectedwm - recoverywm - deathswm
    dIwV =  infectionVw - detectedVw                                  #| Infected Wild Vac
    dQwV =  detectedVw - recoveryVw                                   #| Detected Wild Vac
    dRw =  recoveryw + recoveryVw - exposedmw                                    #| Recovered Wild
    # ---------------------Mutant virus----------------------------------------
    dEm =  exposedm - infectionm                                      #| Exposed Mutant
    dEmV =  exposedVm - infectionVm                                   #| Exposed Mutant Vac
    dEmw =  exposedmw - infectionmw
    dIm =  infectionm - detectedm                                     #| Infected Mutant 
    dImw =  infectionmw - detectedmw
    dQm =  detectedm - recoverym - deathsm                            #| Detected Mutant
    dQmw =  detectedmw - recoverymw - deathsmw
    dImV =  infectionVm - detectedVm                                  #| Infected Mutant Vac
    dQmV =  detectedVm - recoveryVm                                   #| Detected Mutant Vac
    dRm =  recoverym + recoveryVm - exposedwm                                    #| Recovered Mutant
    # ---------------------Common states---------------------------------------
    dD =  deathsw + deathsm + deathswm + deathsmw                      #| Deaths
    dRb =  recoverywm + recoverymw
    dP =  protected - released - vaccinatedP                                       #| Protected
    dV1 =  vaccinated1 + vaccinatedP - vaccinated2 - exposedVw - exposedVm           #| Vacccintaed 1 dose
    dV2 =  vaccinated2                                                 #| Vacccintaed 2 dose
   
    return [dS,dEw,dEwV,dEwm,dIw,dIwV,dIwm,dQw,dQwV,dQwm,dRw,
            dEm,dEmV,dEmw,dIm,dImV,dImw,dQm,dQmV,dQmw,dRm,
            dD,dRb,dP,dV1,dV2]
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------    
def g(t, y0, ps):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = odeint(COVID_SEIRC, y0, t, args=(ps,))
    return x
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def residual(ps,ts,y0,data):
    """
    Residual's function

    Parameters
    ----------
    ps : parameters
        Model parameters.
    ts : vector
        time vector.
    data : Array
        Real data.
    y0: vector
        Initial conditions
    Returns
    -------
    Resid: array.
        residuals of the system

    """
#    y0 = ps['S0'],ps['E0'].value,ps['I0'].value,ps['Q0'].value,ps['R0'].value,ps['D0'].value,ps['C0'].value
    model = g(ts, y0, ps)
    Npop=sum(y0)
    Q= model[:,7] + model[:,8] + model[:,9] + model[:,17] + model[:,18] + model[:,19]
    # Q= model[:,7] + model[:,8] 
    R= model[:,10] + model[:,20] + model[:,22]
    D= model[:,21]
    Inf=((Q - data[0,:])/Npop).ravel()
    Rec=((R - data[1,:])/Npop).ravel()
    Dea=((D - data[2,:])/Npop).ravel()
#    resid=Inf+Rec+Dea
    resid=np.array([Inf,Rec,Dea])
#    nmseI=mean_squared_error(data[0,:],Q) 
#    nmseR=mean_squared_error(data[1,:],R) 
#    nmseD=mean_squared_error(data[2,:],D) 
#    resid=(nmseI+nmseR+nmseD)
    return resid
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def FitMoelforCountry(n,Data):
    """
    This function fit the model parameters for a specific country. As input get 
    the country name and, find the data for that country and fit the model's
    paramerters

    Parameters
    ----------
    n : str
        Name of the country.
    Data : pandas Data Frame
        Epidemic data for the country n.

    Returns
    -------
    FittedParams : numpy array
        The fitted parameters for the model

    """
    # set initial conditions of the model
    time = pd.to_datetime(Data.Time)
    Recovered = np.asarray(Data.Recovered,dtype=int)
    Deaths = np.asarray(Data.Deaths,dtype=int)
    Confirmed = np.asarray(Data.Confirmed,dtype=int)
    Active = np.asarray(Data.Active,dtype=int)
    Vac1=(Vaccines['Total vaccintaed'][Vaccines['Country']==n])
    Vac2=(Vaccines['Total Full vaccinated'][Vaccines['Country']==n])
        
    if  len(Vac2)== 0 or m.isnan(int(Vac2)):
        Vac2=0
    if  len(Vac1)== 0 or m.isnan(int(Vac1)):
        Vac1=0
        
    DataF=np.array([Active,Recovered,Deaths])
    Npop=int(PopulationSize(n,CountriesInfo))
     
    Ew0 = int(Active[0])
    EwV0 = 0
    Ewm0 = 0
    Iw0 = int(Active[0])
    IwV0=0
    Iwm0 = 0
    Qw0 = Iw0
    QwV0=0
    Qwm0 = 0
    Rw0 = int(Recovered[0])
    
    Em0 = 0
    EmV0 = 0
    Emw0 = 0
    Im0 = 0
    ImV0=0
    Imw0 = 0
    Qm0 = 0
    QmV0=0
    Qmw0 =0
    Rm0 = 0
    
    D0 = int(Deaths[0])
    Rb0 = 0
    P0 = 0
    V10 = int(Vac1)-int(Vac2)
    if V10 <0 :
        V10=0
    V20 = int(Vac2)
    S0 = Npop-Ew0-EwV0-Ewm0-Iw0-IwV0-Iwm0-Qw0-QwV0-Qwm0-Rw0-Em0-EmV0-Emw0-Im0-ImV0-Imw0-Qm0-QmV0-Qmw0-Rm0-D0-Rb0-P0-V10-V20

    rate=(Vaccines['Rate'][Vaccines['Country']==n])
    if len(rate)==0 or m.isnan(float(rate)) == True:
        rate=1e-10
    VacRate=float(rate)
    Iter=10
    Outputs=list()
    FullSetOptParam=list()
    Report=pd.DataFrame(columns=['AIC','BIC','Chi Square','Reduced Chi Square'
                             ,'N vars'])
    for i in np.arange(Iter):
        print('Optimziation: ',i+1,' of ',Iter)
        params = Parameters()
        # --------------------------------------------------------------------
        params.add('alpha0', value=np.random.uniform(0,0.03), min=0, max=1)
        params.add('alpha1', value=np.random.uniform(0,0.3), min=0, max=1)
        params.add('beta', value= np.random.uniform(1,2), min=0, max=3)
        params.add('betaQ', value= params['beta'].value*.2, min=0, max=params['beta'].value*.5)
        params.add('gamma', value= np.random.uniform(0,0.5), min=0, max=1)
        # params.add('delta', value= np.random.uniform(0,1/4), min=1/10, max=1/5)
        params.add('delta', value= np.random.uniform(0,0.3), min=0, max=1)
        params.add('lambda0', value= np.random.uniform(0,0.03), min=0, max=1)
        params.add('lambda1', value= np.random.uniform(0,0.03), min=0, max=1)
        params.add('kappa0', value=np.random.uniform(0,0.03), min=0, max=1)
        params.add('kappa1', value= np.random.uniform(0,0.3), min=0, max=1)
        params.add('tau0', value= np.random.uniform(0,0.3), min=0, max=1)
        params.add('tau1', value= np.random.uniform(0,0.03), min=0, max=1)
        params.add('rho1', value= VacRate, min=VacRate*.7, max=VacRate+VacRate*.1)
        params.add('rho2', value= VacRate, min=VacRate*.7, max=VacRate+VacRate*.1)
        params.add('deltaV1', value= 1/21,min=1/30, max=1/14)
        params.add('deltaV2', value= 1/21,min=1/30, max=1/14)
        params.add('nuV', value= 0.1,min=0, max=0.3)
        params.add('omega',value= 1.5,min=1, max=3)
        # --------------------------------------------------------------------
        dt=1/24
        # seting initial conditions 
        y0 = [S0,Ew0,EwV0,Ewm0,Iw0,IwV0,Iwm0,Qw0,QwV0,Qwm0,Rw0,Em0,EmV0,Emw0,Im0,
          ImV0,Imw0,Qm0,QmV0,Qmw0,Rm0,D0,Rb0,P0,V10,V20]
        y0opt =y0
        # defining the time vector for optimization 
        tf = len(time)
        tl = int(tf/dt)
        t = np.linspace(0,tf-1,tl)
        topt= np.linspace(1, len(time), len(time))
        # Runing the optimization
        sol = minimize(residual, params, args=(topt,y0opt,DataF),method='least_squares',max_nfev=10000,
                        ftol=1e-8,gtol=1e-8,xtol=1e-8,loss='soft_l1',diff_step=1e-4,verbose=1,tr_solver='lsmr')
        # sol = minimize(residual, params, args=(topt,y0opt,DataF),method='least_squares',max_nfev=1000,
                        # ftol=1e-10,gtol=1e-10,xtol=1e-10,loss='arctan',diff_step=1e-4,verbose=1,tr_solver='lsmr')
    
        paropt=sol.params
        # print(report_fit(sol))
        FullSetOptParam.append(paropt)  
        Rep=np.array([sol.aic,sol.bic,sol.chisqr,sol.redchi,sol.nvarys]).reshape(1,5)
        RepDf=pd.DataFrame(data=Rep,columns=('AIC','BIC','Chi Square','Reduced Chi Square'
                             ,'N vars'))
        Report=Report.append(RepDf,ignore_index=True)
        # Report.append(RepDf,ignore_index=True)
        
        sir_out = pd.DataFrame(simulate(t,dt,y0, paropt))
            
        ti= np.linspace(t[0],t[len(t)-1],int(t[len(t)-1]-t[0])+1)
        # sir_out = pd.DataFrame(Sys_interpolation(sir_out,t,ti))
        # Outputs.append(sir_out)
        sir_out = pd.DataFrame(Sys_interpolation(sir_out,t,ti))
        sir_out['N']=sir_out.iloc[:, 1:26].sum(axis=1)
        Outputs.append(sir_out)  
    # Just in case to plot the solution of the optimization
    Results=np.array(Outputs)
    vector=Results.mean(axis=0)
    std=Results.std(axis=0)
    # Report=Report.values.tolist()
    # t,s,e,i,q,r,d,p,v= std.transpose()
    # t,S,E,I,Q,R,D,P,V= vector.transpose()
    t,s,ew,ewV,ewm,iw,iwV,iwm,qw,qwV,qwm,rw,em,emV,emw,im,imV,imw,qm,qmV,qmw,rm,d,rb,p,v1,v2,nt = std.transpose()
    t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,Nt = vector.transpose()
    # t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,N
    Q=Qw+QwV+Qm+QmV+Qwm+Qmw
    q=qw+qwV+qm+qmV+qwm+qmw
    R=Rw+Rm+Rb
    r=rw+rm+rb           
    #-------------------------------------------------------------------------
    # Plot Reult
    FigName='Fittings/Plots/'+n+'.png'
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
                          
    # ax.plot(time, Q-q, '--r', alpha=0.5, lw=1)  
    ax.plot(time, Q, 'r', alpha=0.5, lw=2, label='Active')
    # ax.plot(time, Q+q, '--r', alpha=0.5, lw=1)
    ax.fill_between(time,  Q-q, Q+q,color='r', alpha=0.2)
    
    # ax.plot(time, R-r, '--g', alpha=0.5, lw=1)  
    ax.plot(time, R, 'g', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(time, R+r, '--g', alpha=0.5, lw=1)  
    ax.fill_between(time,  R-r, R+r,color='g', alpha=0.2)

    # ax.plot(time, D-d, '--y', alpha=0.5, lw=1)  
    ax.plot(time, D, 'y', alpha=0.5, lw=2, label='Deaths')
    # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
    ax.fill_between(time,  D-d, D+d,color='y', alpha=0.2)

    ax.plot(time,Active, 'or', alpha=0.5, lw=2, label='Active')
    ax.plot(time,Recovered, 'og', alpha=0.5, lw=2, label='Recovered')
    ax.plot(time,Deaths, 'oy', alpha=0.5, lw=2, label='Deaths')
    ax.plot(time,Confirmed, 'ok', alpha=0.5, lw=2, label='Total Cases')
    
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.set_title(n)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    fig.savefig(FigName,dpi=600)
    plt.close(fig)
    return FullSetOptParam,Outputs,Report
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def CheckParameters(countries):
    """
    Parameters
    ----------
    countries : List
        List of countries.

    Returns
    -------
    FullParamDF : Pandas Data Frame
        Data Frame with the countries parameters.

    """
    alpha0=list()
    alpha1=list()
    beta=list()
    betaQ=list()
    delta=list()
    deltaV1=list()
    deltaV2=list()
    gamma=list()
    kappa0=list()
    kappa1=list()
    lambda0=list()
    lambda1=list()
    nuV=list()
    omega=list()
    rho1=list()
    rho2=list()
    tau0=list()
    tau1=list()
    Country=list()
    
    for i in countries:
        File='Fittings/'+i+'.pckl'
        Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
        # Country.append(i)
        for j in np.arange(len(Params)):
            alpha0.append(Params[j]['alpha0'].value)
            alpha1.append(Params[j]['alpha1'].value)
            beta.append(Params[j]['beta'].value)
            betaQ.append(Params[j]['betaQ'].value)
            delta.append(Params[j]['delta'].value)
            deltaV1.append(Params[j]['deltaV1'].value)
            deltaV2.append(Params[j]['deltaV2'].value)
            gamma.append(Params[j]['gamma'].value)
            kappa0.append(Params[j]['kappa0'].value)
            kappa1.append(Params[j]['kappa1'].value)
            lambda0.append(Params[j]['lambda0'].value)
            lambda1.append(Params[j]['lambda1'].value)
            nuV.append(Params[j]['nuV'].value)
            omega.append(Params[j]['omega'].value)
            rho1.append(Params[j]['rho1'].value)
            rho2.append(Params[j]['rho2'].value)
            tau0.append(Params[j]['tau0'].value)
            tau1.append(Params[j]['tau1'].value)
            Country.append(i)
    
    ParamData={'Country':Country,
               'alpha_0':alpha0,
               'alpha_1':alpha1,
               'beta_I':beta,
               'beta_Q':betaQ,
               'delta':delta,
               'delta_{V1}':deltaV1,
               'delta_{V2}':deltaV2,
               'gamma':gamma,
               'k_0':kappa0,
               'k_1':kappa1,
               'lambda_0':lambda0,
               'lambda_1':lambda1,
               'nu_V':nuV,
               'omega':omega,
               'rho_1':rho1,
               'rho_2':rho2,
               'tau_0':tau0,
               'tau_1':tau1}
    
    FullParamDF=pd.DataFrame(ParamData, columns= ['Country','alpha_0','alpha_1',
                                                  'beta_I','beta_Q','delta',
                                                  'delta_{V1}','delta_{V2}',
                                                  'gamma','k_0','k_1','lambda_0',
                                                  'lambda_1','nu_V','omega','rho_1',
                                                  'rho_2','tau_0','tau_1'])
    MeanParamDF=FullParamDF.groupby('Country',as_index=False).mean()
    return FullParamDF, MeanParamDF
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PlotParameters(countriesInfo,FullParamSet):
    DataFrame= pd.merge(FullParamSet, countriesInfo, on="Country")
    # countries=pd.unique(DataFrame['Country'])
    # continents=pd.unique(DataFrame['Continent'])
    Dir='Fittings/Parameters Distributions/By continent/'
    FigName=Dir+'alpha_0.png'
    DataFrame['alpha_0'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\alpha_0$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'alpha_1.png'
    DataFrame['alpha_1'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\alpha_1$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'beta_I.png'
    DataFrame['beta_I'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\beta_I$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'beta_Q.png'
    DataFrame['beta_Q'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\beta_Q$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'delta.png'
    DataFrame['delta'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\delta$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'delta_V1.png'
    DataFrame['delta_{V1}'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\delta_{V1}$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'delta_V2.png'
    DataFrame['delta_{V2}'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\delta_{V2}$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'gamma.png'
    DataFrame['gamma'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\gamma$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'k_0.png'
    DataFrame['k_0'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$k_0$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'k_1.png'
    DataFrame['k_1'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$k_1$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'lambda_0.png'
    DataFrame['lambda_0'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\lambda_0$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'lambda_1.png'
    DataFrame['lambda_1'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\lambda_1$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'nu_V.png'
    DataFrame['nu_V'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\nu_V$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'omega.png'
    DataFrame['omega'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\omega$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'rho_1.png'
    DataFrame['rho_1'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\rho_1$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'rho_2.png'
    DataFrame['rho_2'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\rho_2$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'tau_0.png'
    DataFrame['tau_0'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10])
    plt.suptitle('$\\tau_0$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------
    FigName=Dir+'tau_1.png'
    DataFrame['tau_1'].hist(by=DataFrame['Continent'],
                              bins=40,
                              alpha=0.7,
                              xlabelsize=15,
                              ylabelsize=15,
                              figsize=[20,10],
                              xrot=45)
    plt.suptitle('$\\tau_1$',size=35)
    plt.savefig(FigName,dpi=600)
    plt.close()
    # plt.tight_layout()
    # -------------------------------------------------------------------------

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PlotFittings(countries):
    """

    Parameters
    ----------
    countries : list
        List of countries to plot.

    Returns
    -------
    None.

    """
    for i in countries:
        File='Fittings/'+i+'.pckl'
        Params,Outputs,Report,Data= pickle.load(open(File,"rb"))

        Results=np.array(Outputs)
        vector=Results.mean(axis=0)
        std=Results.std(axis=0)
        # Report=Report.values.tolist()
        # t,s,e,i,q,r,d,p,v= std.transpose()
        # t,S,E,I,Q,R,D,P,V= vector.transpose()
        t,s,ew,ewV,ewm,iw,iwV,iwm,qw,qwV,qwm,rw,em,emV,emw,im,imV,imw,qm,qmV,qmw,rm,d,rb,p,v1,v2,nt = std.transpose()
        t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,Nt = vector.transpose()
        # t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,N
        Q=Qw+QwV+Qm+QmV+Qwm+Qmw
        q=qw+qwV+qm+qmV+qwm+qmw
        R=Rw+Rm+Rb
        r=rw+rm+rb  
        time = pd.to_datetime(Data.Time)
        Recovered = np.asarray(Data.Recovered,dtype=int)
        Deaths = np.asarray(Data.Deaths,dtype=int)
        Confirmed = np.asarray(Data.Confirmed,dtype=int)
        Active = np.asarray(Data.Active,dtype=int)
        #-------------------------------------------------------------------------
        # Plot Reult
        FigName='Fittings/Plots/'+i+'.png'
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
                              
        # ax.plot(time, Q-q, '--r', alpha=0.5, lw=1)  
        ax.plot(time, Q, 'r', alpha=0.5, lw=2, label='Active')
        # ax.plot(time, Q+q, '--r', alpha=0.5, lw=1)
        ax.fill_between(time,  Q-q, Q+q,color='r', alpha=0.2)
        
        # ax.plot(time, R-r, '--g', alpha=0.5, lw=1)  
        ax.plot(time, R, 'g', alpha=0.5, lw=2, label='Recovered')
        # ax.plot(time, R+r, '--g', alpha=0.5, lw=1)  
        ax.fill_between(time,  R-r, R+r,color='g', alpha=0.2)
    
        # ax.plot(time, D-d, '--y', alpha=0.5, lw=1)  
        ax.plot(time, D, 'y', alpha=0.5, lw=2, label='Deaths')
        # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
        ax.fill_between(time,  D-d, D+d,color='y', alpha=0.2)
    
        ax.plot(time,Active, 'or', alpha=0.5, lw=2, label='Active')
        ax.plot(time,Recovered, 'og', alpha=0.5, lw=2, label='Recovered')
        ax.plot(time,Deaths, 'oy', alpha=0.5, lw=2, label='Deaths')
        ax.plot(time,Confirmed, 'ok', alpha=0.5, lw=2, label='Total Cases')
        
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Number')
        ax.set_ylim(bottom=0)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        ax.set_title(i)
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()
        fig.savefig(FigName,dpi=600)
        plt.close(fig)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PlotFittingsGlobal(countries):
    """

    Parameters
    ----------
    countries : list
        List of countries to plot.

    Returns
    -------
    None.

    """
    QT= list()
    RT= list()
    DT= list()
    qt= list()
    rt= list()
    dt= list()
    TOTAL=list()
    DEATHS=list()
    RECOVERED=list()
    for i in countries:
        File='Fittings/'+i+'.pckl'
        Params,Outputs,Report,Data= pickle.load(open(File,"rb"))

        Results=np.array(Outputs)
        vector=Results.mean(axis=0)
        std=Results.std(axis=0)
        # Report=Report.values.tolist()
        # t,s,e,i,q,r,d,p,v= std.transpose()
        # t,S,E,I,Q,R,D,P,V= vector.transpose()
        t,s,ew,ewV,ewm,iw,iwV,iwm,qw,qwV,qwm,rw,em,emV,emw,im,imV,imw,qm,qmV,qmw,rm,d,rb,p,v1,v2,nt = std.transpose()
        t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,Nt = vector.transpose()
        # t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,N
        Q=Qw+QwV+Qm+QmV+Qwm+Qmw
        q=qw+qwV+qm+qmV+qwm+qmw
        R=Rw+Rm+Rb
        r=rw+rm+rb  
        time = pd.to_datetime(Data.Time)
        Recovered = np.asarray(Data.Recovered,dtype=int)
        Deaths = np.asarray(Data.Deaths,dtype=int)
        Confirmed = np.asarray(Data.Confirmed,dtype=int)
        # Active = np.asarray(Data.Active,dtype=int)
        QT.append(Q)
        RT.append(R)
        DT.append(D)
        qt.append(q)
        rt.append(r)
        dt.append(d)
        TOTAL.append(Confirmed)
        DEATHS.append(Deaths)
        RECOVERED.append(Recovered)
    QT=pd.DataFrame(QT).sum()
    RT=pd.DataFrame(RT).sum()
    DT=pd.DataFrame(DT).sum()
    qt=pd.DataFrame(qt).sum()
    rt=pd.DataFrame(rt).sum()
    dt=pd.DataFrame(dt).sum()
    totalmodel=qt+rt+dt
    TOTALMODEL=QT+RT+DT
    TOTAL=pd.DataFrame(TOTAL).sum()
    DEATHS=pd.DataFrame(DEATHS).sum()
    RECOVERED=pd.DataFrame(RECOVERED).sum()
    #-------------------------------------------------------------------------
    # Plot Reult
    FigName='Fittings/Plots/WorldFitting.png'
    fig = plt.figure(figsize=(10, 10),facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
                          
    # ax.plot(time, Q-q, '--r', alpha=0.5, lw=1)  
    ax.plot(time, TOTALMODEL, 'k', alpha=0.5, lw=2, label='Total Cases')
    # ax.plot(time, Q+q, '--r', alpha=0.5, lw=1)
    ax.fill_between(time,  TOTALMODEL-totalmodel, TOTALMODEL+totalmodel,color='k', alpha=0.2)
    
    # ax.plot(time, R-r, '--g', alpha=0.5, lw=1)  
    ax.plot(time, RT, 'g', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(time, R+r, '--g', alpha=0.5, lw=1)  
    ax.fill_between(time,  RT-rt, RT+rt,color='g', alpha=0.2)

    # ax.plot(time, D-d, '--y', alpha=0.5, lw=1)  
    ax.plot(time, DT, 'y', alpha=0.5, lw=2, label='Deaths')
    # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
    ax.fill_between(time,  DT-dt, DT+dt,color='y', alpha=0.2)

    # ax.plot(time,Active, 'or', alpha=0.5, lw=2, label='Active')
    ax.plot(time,RECOVERED, 'og', alpha=0.5, lw=2, label='Recovered')
    ax.plot(time,DEATHS, 'oy', alpha=0.5, lw=2, label='Deaths')
    ax.plot(time,TOTAL, 'ok', alpha=0.5, lw=2, label='Total Cases')
    
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.set_ylim(bottom=0)
    plt.xticks(rotation=45)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.set_title('World fitting')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    fig.savefig(FigName,dpi=600)
    # plt.close(fig)
    # return QT,RT,DT,qt,rt,dt
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def PlotFittingsContinent(countries):
    """

    Parameters
    ----------
    countries : list
        List of countries to plot.

    Returns
    -------
    None.

    """
    for j in pd.unique(countries['Continent']):        
        QT= list()
        RT= list()
        DT= list()
        qt= list()
        rt= list()
        dt= list()
        TOTAL=list()
        DEATHS=list()
        RECOVERED=list()    
        listofcountries=countries['Country'][countries['Continent']==j]
        for i in listofcountries:
            File='Fittings/'+i+'.pckl'
            Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
    
            Results=np.array(Outputs)
            vector=Results.mean(axis=0)
            std=Results.std(axis=0)
            # Report=Report.values.tolist()
            # t,s,e,i,q,r,d,p,v= std.transpose()
            # t,S,E,I,Q,R,D,P,V= vector.transpose()
            t,s,ew,ewV,ewm,iw,iwV,iwm,qw,qwV,qwm,rw,em,emV,emw,im,imV,imw,qm,qmV,qmw,rm,d,rb,p,v1,v2,nt = std.transpose()
            t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,Nt = vector.transpose()
            # t,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2,N
            Q=Qw+QwV+Qm+QmV+Qwm+Qmw
            q=qw+qwV+qm+qmV+qwm+qmw
            R=Rw+Rm+Rb
            r=rw+rm+rb  
            time = pd.to_datetime(Data.Time)
            Recovered = np.asarray(Data.Recovered,dtype=int)
            Deaths = np.asarray(Data.Deaths,dtype=int)
            Confirmed = np.asarray(Data.Confirmed,dtype=int)
            # Active = np.asarray(Data.Active,dtype=int)
            QT.append(Q)
            RT.append(R)
            DT.append(D)
            qt.append(q)
            rt.append(r)
            dt.append(d)
            TOTAL.append(Confirmed)
            DEATHS.append(Deaths)
            RECOVERED.append(Recovered)
        QT=pd.DataFrame(QT).sum()
        RT=pd.DataFrame(RT).sum()
        DT=pd.DataFrame(DT).sum()
        qt=pd.DataFrame(qt).sum()
        rt=pd.DataFrame(rt).sum()
        dt=pd.DataFrame(dt).sum()
        totalmodel=qt+rt+dt
        TOTALMODEL=QT+RT+DT
        TOTAL=pd.DataFrame(TOTAL).sum()
        DEATHS=pd.DataFrame(DEATHS).sum()
        RECOVERED=pd.DataFrame(RECOVERED).sum()
        #-------------------------------------------------------------------------
        # Plot Reult
        FigName='Fittings/Plots/'+j+'.png'
        fig = plt.figure(figsize=(10, 10),facecolor='w')
        ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
                              
        # ax.plot(time, Q-q, '--r', alpha=0.5, lw=1)  
        ax.plot(time, TOTALMODEL, 'k', alpha=0.5, lw=2, label='Total Cases')
        # ax.plot(time, Q+q, '--r', alpha=0.5, lw=1)
        ax.fill_between(time,  TOTALMODEL-totalmodel, TOTALMODEL+totalmodel,color='k', alpha=0.2)
        
        # ax.plot(time, R-r, '--g', alpha=0.5, lw=1)  
        ax.plot(time, RT, 'g', alpha=0.5, lw=2, label='Recovered')
        # ax.plot(time, R+r, '--g', alpha=0.5, lw=1)  
        ax.fill_between(time,  RT-rt, RT+rt,color='g', alpha=0.2)
    
        # ax.plot(time, D-d, '--y', alpha=0.5, lw=1)  
        ax.plot(time, DT, 'y', alpha=0.5, lw=2, label='Deaths')
        # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
        ax.fill_between(time,  DT-dt, DT+dt,color='y', alpha=0.2)
    
        # ax.plot(time,Active, 'or', alpha=0.5, lw=2, label='Active')
        ax.plot(time,RECOVERED, 'og', alpha=0.5, lw=2, label='Recovered')
        ax.plot(time,DEATHS, 'oy', alpha=0.5, lw=2, label='Deaths')
        ax.plot(time,TOTAL, 'ok', alpha=0.5, lw=2, label='Total Cases')
        
        ax.set_xlabel('Time /days')
        ax.set_ylabel('Number')
        ax.set_ylim(bottom=0)
        plt.xticks(rotation=45)
        ax.yaxis.set_tick_params(length=0)
        ax.xaxis.set_tick_params(length=0)
        ax.grid(b=True, which='major', c='w', lw=2, ls='-')
        ax.set_title(j)
        legend = ax.legend()
        legend.get_frame().set_alpha(0.5)
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        plt.show()
        fig.savefig(FigName,dpi=600)
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def FitFullNetwork(Countries,days):
    print('algo para que no quede vacío')
    for i in Countries:
    # i='Brazil'
        print('Fitting model for: ',i)
        Data,Temp=GetCountriinfo(i, days)
        T=Temp.Temperature
        timeT= pd.to_datetime(Temp.Date)
        
        global f
        global ini
        ini=int(np.where(pd.to_datetime(Temp.Date) == pd.to_datetime(Data.Time.iloc[0]))[0])
        tb,Beta=T_inv(T,timeT)
        f = interp1d(tb, Beta, kind='cubic')
        
        Params,Outputs,Report=FitMoelforCountry(i,Data)
        # OutputsDF=pd.concat(Outputs)
       
        File='Fittings/'+i+'.pckl'
        pickle.dump([Params,Outputs,Report,Data], open(File, "wb"))
    # ParamsO,OutputsO,Data= pickle.load(open(File,"rb"))
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def GetNeighbours(g,country):
    """
    This function get the list of neighbours of a country with the number of air 
    connections 

    Parameters
    ----------
    g : NetworkX Graph
        Full network.
    country : str
        name of the country.

    Returns
    -------
    Data: Pandas Data Frame
        Dataframe with the information. Neighbours and number of Routes

    """
    Vecino=list()
    Rutas=list()
    for m in g._adj[country]:
        # v=m
        # print('\t\t Vecino: ',g._node[m]['Name'],'. Rutas: ', g.get_edge_data(u, v, default=None)['Routes']) 
        Vecino.append(g._node[m]['Name'])
        Rutas.append( g.get_edge_data(country, m, default=None)['Routes'])
        
    
    Data={'Neighbours':Vecino,
          'Routes': Rutas}
    
    Data=pd.DataFrame(Data, columns=['Neighbours','Routes'])
    return Data
     
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def Simulate_Full_network(g,Countries,days,dt,connectivity,ran):
    """
    This funtion is to simulate the full world network

    Parameters
    ----------
    g : Network X Graph
        Graph of the network.
    Countries : Pandas Data Frame
        Country information.
    days : int
        Number of days you want to simulate.
    dt : float
        time step .
        
    connectivity: float.
        number in the interval [0,1] that denote the grade of connectivity, being 0 no
        connectivity and 1 full connection
    ran: list.
        provide the arange of parameters use for simulations from 0 to len of 
        parameters vectors

    Returns
    -------
    TimeSimulation : Datetime vector
        Date time vector with the dates of simulations.
    S : Pandas Data Frame
        Susceptibles day by day, iteration by iteration, and country by country in the network.
    E : Pandas Data Frame
        Exposed day by day, iteration by iteration, and country by country in the network.
    I : Pandas Data Frame
        Infected day by day, iteration by iteration, and country by country in the network.
    Q : Pandas Data Frame
        Reported day by day, iteration by iteration, and country by country in the network.
    R : Pandas Data Frame
        Recovered day by day, iteration by iteration, and country by country in the network.
    D : Pandas Data Frame
        Deaths day by day, iteration by iteration, and country by country in the network.
    P : Pandas Data Frame
        Protected day by day, iteration by iteration, and country by country in the network.
    V : Pandas Data Frame
        Vaccinated day by day, iteration by iteration, and country by country in the network.

    """
    print('Simulating Full Network:')
    File='Fittings/Argentina.pckl'    
    Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
    datei=pd.to_datetime(Data['Time'][max(Data.index)])
    TimeSimulation=  (datei + np.arange(days) * timedelta(days=1)).astype('str')
    S=pd.DataFrame(columns=(TimeSimulation))
    S.insert(loc=0, column='Country',value='')
    # E,I,Q,R,D,P,V=S,S,S,S,S,S,S
    S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S
    indicec=0
    # for i in np.arange(len(Outputs)):
    for i in ran:
        # ----------------Set initial populations--------------------------
        print('Setting initial conditions for parameters ',i+1,' of ', len(ran),'parameters: ',i )
        for j in Countries['Country']:
            # print('Country: ',j)
            File='Fittings/'+j+'.pckl'    
            Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
            # -------------------------------------------------------------------
            new_rowS = {'Country':j, TimeSimulation[0]:Outputs[i].S[len(Outputs[0].S)-1]}
            S   = S.append(new_rowS, ignore_index=True)     #| Susceptible Wild
            # ---------------------Wild virus------------------------------------------
            new_rowEw = {'Country':j, TimeSimulation[0]:Outputs[i].Ew[len(Outputs[0].Ew)-1]}
            Ew  = Ew.append(new_rowEw, ignore_index=True)   

            new_rowEwV = {'Country':j, TimeSimulation[0]:Outputs[i].EwV[len(Outputs[0].EwV)-1]}
            EwV = EwV.append(new_rowEwV, ignore_index=True)    

            new_rowEwm = {'Country':j, TimeSimulation[0]:Outputs[i].Ewm[len(Outputs[0].Ewm)-1]}
            Ewm = Ewm.append(new_rowEwm, ignore_index=True)  
            
            new_rowIw = {'Country':j, TimeSimulation[0]:Outputs[i].Iw[len(Outputs[0].Iw)-1]}
            Iw  = Iw.append(new_rowIw, ignore_index=True)      

            new_rowIwm = {'Country':j, TimeSimulation[0]:Outputs[i].Iwm[len(Outputs[0].Iwm)-1]}
            Iwm = Iwm.append(new_rowIwm, ignore_index=True)
            
            new_rowQw = {'Country':j, TimeSimulation[0]:Outputs[i].Qw[len(Outputs[0].Qw)-1]}
            Qw  = Qw.append(new_rowQw, ignore_index=True)

            new_rowQwm = {'Country':j, TimeSimulation[0]:Outputs[i].Qwm[len(Outputs[0].Qwm)-1]}
            Qwm = Qwm.append(new_rowQwm, ignore_index=True)
            
            new_rowIwV = {'Country':j, TimeSimulation[0]:Outputs[i].IwV[len(Outputs[0].IwV)-1]}
            IwV = IwV.append(new_rowIwV, ignore_index=True)       

            new_rowQwV = {'Country':j, TimeSimulation[0]:Outputs[i].QwV[len(Outputs[0].QwV)-1]}
            QwV = QwV.append(new_rowQwV, ignore_index=True)   

            new_rowRw = {'Country':j, TimeSimulation[0]:Outputs[i].Rw[len(Outputs[0].Rw)-1]}
            Rw  = Rw.append(new_rowRw, ignore_index=True)                      #| Recovered Wild
            # ---------------------Mutant virus----------------------------------------
            new_rowEm = {'Country':j, TimeSimulation[0]:Outputs[i].Em[len(Outputs[0].Em)-1]}
            Em  = Em.append(new_rowEm, ignore_index=True)    

            new_rowEmV = {'Country':j, TimeSimulation[0]:Outputs[i].EmV[len(Outputs[0].EmV)-1]}
            EmV = EmV.append(new_rowEmV, ignore_index=True)

            new_rowEmw = {'Country':j, TimeSimulation[0]:Outputs[i].Emw[len(Outputs[0].Emw)-1]}
            Emw =  Emw.append(new_rowEmw, ignore_index=True)
            
            new_rowIm = {'Country':j, TimeSimulation[0]:Outputs[i].Im[len(Outputs[0].Im)-1]}
            Im  = Im.append(new_rowIm, ignore_index=True)   

            new_rowImw = {'Country':j, TimeSimulation[0]:Outputs[i].Imw[len(Outputs[0].Imw)-1]}
            Imw =  Imw.append(new_rowImw, ignore_index=True)
            
            new_rowQm = {'Country':j, TimeSimulation[0]:Outputs[i].Qm[len(Outputs[0].Qm)-1]}
            Qm  = Qm.append(new_rowQm, ignore_index=True)

            new_rowQmw = {'Country':j, TimeSimulation[0]:Outputs[i].Qmw[len(Outputs[0].Qmw)-1]}
            Qmw = Qmw.append(new_rowQmw, ignore_index=True)
            
            new_rowImV = {'Country':j, TimeSimulation[0]:Outputs[i].ImV[len(Outputs[0].ImV)-1]}
            ImV = ImV.append(new_rowImV, ignore_index=True)

            new_rowQmV = {'Country':j, TimeSimulation[0]:Outputs[i].QmV[len(Outputs[0].QmV)-1]}
            QmV = QmV.append(new_rowQmV, ignore_index=True)  

            new_rowRm = {'Country':j, TimeSimulation[0]:Outputs[i].Rm[len(Outputs[0].Rm)-1]}
            Rm  = Rm.append(new_rowRm, ignore_index=True)                                    #| Recovered Mutant
            # ---------------------Common states---------------------------------------
            new_rowD = {'Country':j, TimeSimulation[0]:Outputs[i].D[len(Outputs[0].D)-1]}
            D   = D.append(new_rowD, ignore_index=True)  

            new_rowRb = {'Country':j, TimeSimulation[0]:Outputs[i].Rb[len(Outputs[0].Rb)-1]}
            Rb  = Rb.append(new_rowRb, ignore_index=True)
            
            new_rowP = {'Country':j, TimeSimulation[0]:Outputs[i].P[len(Outputs[0].P)-1]}
            P   = P.append(new_rowP, ignore_index=True)    

            new_rowV1 = {'Country':j, TimeSimulation[0]:Outputs[i].V1[len(Outputs[0].V1)-1]}
            V1  = V1.append(new_rowV1, ignore_index=True)   

            new_rowV2 = {'Country':j, TimeSimulation[0]:Outputs[i].V2[len(Outputs[0].V2)-1]}
            V2  = V2.append(new_rowV2, ignore_index=True)       
        # Simulating the compleate network
        print('Simulating for parameters ',i+1,' of ', len(ran))
        start=len(Outputs[0].S)
        for k in np.arange(1,days):
            # print('Day ',k+1,' of ',TimeSimulation[k+1])
            # indicec=1
            tl = int(1/dt)
            # t = np.linspace(k-1,k-dt,tl)
            t = np.linspace(start+k-1,start+k-dt,tl)            
            v=0
            for j in Countries['Country']:
                # indicec=indicec*indicP-1
                Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
                indi=indicec+v
                # print('Simulating for country ',j,' day ',k+1,' of ', days,'indice ',indi)
                Neig=GetNeighbours(g,j)
                # print('País: ',j)
                Ei=0
                Eim=0
                # ------------------getting Temp profile-----------------------
                Daty,Temp=GetCountriinfo(j, days)
                T=Temp.Temperature
                timeT= pd.to_datetime(Temp.Date)
                global f
                global ini
                # ini=int(np.where(pd.to_datetime(Temp.Date) == pd.to_datetime(Daty.Time.iloc[0]))[0])
                ini=int(np.where(pd.to_datetime(Temp.Date) == pd.to_datetime(datei))[0])
                tb,Beta=T_inv(T,timeT)
                f = interp1d(tb, Beta, kind='cubic')
                # -------------------------------------------------------------
                
                for x in Neig['Neighbours']:
                    Filex='Fittings/'+x+'.pckl' 
                    # Parx,Outx,Datax= pickle.load(open(Filex,"rb"))
                    Parx,Outx,Repox,Datax= pickle.load(open(Filex,"rb"))
                    # Popx=sum(Countries[Countries['Country']==x]['Population'])
                    Popx=Countries.loc[Countries['Country']==x,'Population'].values[-1]
                    infx=Iw.loc[Iw['Country'] == x, TimeSimulation[k-1]].values[-1]+Iwm.loc[Iwm['Country'] == x, TimeSimulation[k-1]].values[-1]
                    repx=Qw.loc[Qw['Country'] == x, TimeSimulation[k-1]].values[-1]+Qwm.loc[Qwm['Country'] == x, TimeSimulation[k-1]].values[-1]
                    infxm=Im.loc[Iw['Country'] == x, TimeSimulation[k-1]].values[-1]+Imw.loc[Iwm['Country'] == x, TimeSimulation[k-1]].values[-1]
                    repxm=Qm.loc[Qw['Country'] == x, TimeSimulation[k-1]].values[-1]+Qmw.loc[Qmw['Country'] == x, TimeSimulation[k-1]].values[-1]
                   
                    betaIx=Parx[i]['beta'].value
                    betaQx=Parx[i]['betaQ'].value
                    
                    Beta=1/Popx*(betaIx*infx + betaQx * repx)
                    Betam=1/Popx*(betaIx*infxm + betaQx * repxm)*Parx[i]['omega'].value
                    efrac=abs(1.0 - m.exp(-Beta*dt)) 
                    efracm=abs(1.0 - m.exp(-Betam*dt)) 
                    Sx=int(400*int(Neig[Neig['Neighbours']==x]['Routes'].values[0]*connectivity))
                    Expo=np.random.binomial(Sx,efrac)
                    Expom=np.random.binomial(Sx,efracm)
                    # print('\t ',x,' - Infectados: ',infx,' - Reportados: ' ,repx, '- Popu: ',Popx, 
                    #       ' - Beta: ',Beta, ' - Exposed individuals ',Expo)
                    Ei=Ei+Expo
                    Eim=Eim+Expom
                # y0=[S.loc[S['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     E.loc[E['Country'] == j, TimeSimulation[k-1]].values[-1]+Ei,
                #     I.loc[I['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     Q.loc[Q['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     R.loc[R['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     D.loc[D['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     P.loc[P['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     V.loc[V['Country'] == j, TimeSimulation[k-1]].values[-1]]
                y0=[S.loc[S['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Ew.loc[Ew['Country'] == j, TimeSimulation[k-1]].values[-1]+Ei,
                    EwV.loc[EwV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Ewm.loc[Ewm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Iw.loc[Iw['Country'] == j, TimeSimulation[k-1]].values[-1],   
                    Iwm.loc[Iwm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qw.loc[Qw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qwm.loc[Qwm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    IwV.loc[IwV['Country'] == j, TimeSimulation[k-1]].values[-1],  
                    QwV.loc[QwV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Rw .loc[Rw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Em.loc[Em['Country'] == j, TimeSimulation[k-1]].values[-1]+Eim,
                    EmV.loc[EmV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Emw.loc[Emw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Im.loc[Im['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Imw.loc[Imw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qm.loc[Qm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qmw.loc[Qmw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    ImV.loc[ImV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    QmV.loc[QmV['Country'] == j, TimeSimulation[k-1]].values[-1], 
                    Rm.loc[Rm['Country'] == j, TimeSimulation[k-1]].values[-1],                                   #| Recovered Mutant
                    D.loc[D['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Rb.loc[Rb['Country'] == j, TimeSimulation[k-1]].values[-1],
                    P.loc[P['Country'] == j, TimeSimulation[k-1]].values[-1], 
                    V1.loc[V1['Country'] == j, TimeSimulation[k-1]].values[-1],
                    V2.loc[V2['Country'] == j, TimeSimulation[k-1]].values[-1] ]  

                print('Simuation in node : ',j, ' Iteration: ',i+1, ' of ',len(ran),
                      ' Day:',k+1, ' of ',days)
                sir_out = pd.DataFrame(simulate(t,dt,y0, Params[i]))
                
                S.at[indi,TimeSimulation[k]]=sir_out.S[len(sir_out.S)-1]
                Ew.at[indi,TimeSimulation[k]]=sir_out.Ew[len(sir_out.Ew)-1]
                EwV.at[indi,TimeSimulation[k]]=sir_out.EwV[len(sir_out.EwV)-1]
                Ewm.at[indi,TimeSimulation[k]]=sir_out.Ewm[len(sir_out.Ewm)-1]
                Iw.at[indi,TimeSimulation[k]]=sir_out.Iw[len(sir_out.Iw)-1]   
                Iwm.at[indi,TimeSimulation[k]]=sir_out.Iwm[len(sir_out.Iwm)-1]
                Qw.at[indi,TimeSimulation[k]]=sir_out.Qw[len(sir_out.Qw)-1]
                Qwm.at[indi,TimeSimulation[k]]=sir_out.Qwm[len(sir_out.Qwm)-1]
                IwV.at[indi,TimeSimulation[k]]=sir_out.IwV[len(sir_out.IwV)-1] 
                QwV.at[indi,TimeSimulation[k]]=sir_out.QwV[len(sir_out.QwV)-1]
                Rw.at[indi,TimeSimulation[k]]=sir_out.Rw[len(sir_out.Rw)-1]
                Em.at[indi,TimeSimulation[k]]=sir_out.Em[len(sir_out.Em)-1]
                EmV.at[indi,TimeSimulation[k]]=sir_out.EmV[len(sir_out.EmV)-1]
                Emw.at[indi,TimeSimulation[k]]=sir_out.Emw[len(sir_out.Emw)-1]
                Im.at[indi,TimeSimulation[k]]=sir_out.Im[len(sir_out.Im)-1]
                Imw.at[indi,TimeSimulation[k]]=sir_out.Imw[len(sir_out.Imw)-1]
                Qm.at[indi,TimeSimulation[k]]=sir_out.Qm[len(sir_out.Qm)-1]
                Qmw.at[indi,TimeSimulation[k]]=sir_out.Qmw[len(sir_out.Qmw)-1]
                ImV.at[indi,TimeSimulation[k]]=sir_out.ImV[len(sir_out.ImV)-1]
                QmV.at[indi,TimeSimulation[k]]=sir_out.QmV[len(sir_out.QmV)-1]
                Rm.at[indi,TimeSimulation[k]]=sir_out.Rm[len(sir_out.Rm)-1]                                 #| Recovered Mutant
                D.at[indi,TimeSimulation[k]]=sir_out.D[len(sir_out.D)-1]
                Rb.at[indi,TimeSimulation[k]]=sir_out.Rb[len(sir_out.Rb)-1]
                P.at[indi,TimeSimulation[k]]=sir_out.P[len(sir_out.P)-1]
                V1.at[indi,TimeSimulation[k]]=sir_out.V1[len(sir_out.V1)-1]
                V2.at[indi,TimeSimulation[k]]=sir_out.V2[len(sir_out.V2)-1]
                v=v+1
        indicec=indi+1

    S=S.sort_values(by=['Country']).reset_index(drop=True)
    Ew=Ew.sort_values(by=['Country']).reset_index(drop=True)
    EwV=EwV.sort_values(by=['Country']).reset_index(drop=True)
    Ewm=Ewm.sort_values(by=['Country']).reset_index(drop=True)
    Iw=Iw.sort_values(by=['Country']).reset_index(drop=True) 
    Iwm=Iwm.sort_values(by=['Country']).reset_index(drop=True)
    Qw= Qw.sort_values(by=['Country']).reset_index(drop=True)
    Qwm=Qwm.sort_values(by=['Country']).reset_index(drop=True)
    IwV=IwV.sort_values(by=['Country']).reset_index(drop=True)
    QwV=QwV.sort_values(by=['Country']).reset_index(drop=True)
    Rw=Rw.sort_values(by=['Country']).reset_index(drop=True)
    Em=Em.sort_values(by=['Country']).reset_index(drop=True)
    EmV=EmV.sort_values(by=['Country']).reset_index(drop=True)
    Emw=Emw.sort_values(by=['Country']).reset_index(drop=True)
    Im=Im.sort_values(by=['Country']).reset_index(drop=True)
    Imw=Imw.sort_values(by=['Country']).reset_index(drop=True)
    Qm=Qm.sort_values(by=['Country']).reset_index(drop=True)
    Qmw=Qmw.sort_values(by=['Country']).reset_index(drop=True)
    ImV=ImV.sort_values(by=['Country']).reset_index(drop=True)
    QmV=QmV.sort_values(by=['Country']).reset_index(drop=True)
    Rm=Rm.sort_values(by=['Country']).reset_index(drop=True)                            #| Recovered Mutant
    D=D.sort_values(by=['Country']).reset_index(drop=True)
    Rb=Rb.sort_values(by=['Country']).reset_index(drop=True)
    P=P.sort_values(by=['Country']).reset_index(drop=True)
    V1=V1.sort_values(by=['Country']).reset_index(drop=True)
    V2=V2.sort_values(by=['Country']).reset_index(drop=True)
    return TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def Simulate_Full_networkVaccinScenario(g,Countries,days,dt,connectivity,ran,Info,NewRate):
    """
    This funtion is to simulate the full world network

    Parameters
    ----------
    g : Network X Graph
        Graph of the network.
    Countries : Pandas Data Frame
        Country information.
    days : int
        Number of days you want to simulate.
    dt : float
        time step .
        
    connectivity: float.
        number in the interval [0,1] that denote the grade of connectivity, being 0 no
        connectivity and 1 full connection
    ran: list.
        provide the arange of parameters use for simulations from 0 to len of 
        parameters vectors

    Returns
    -------
    TimeSimulation : Datetime vector
        Date time vector with the dates of simulations.
    S : Pandas Data Frame
        Susceptibles day by day, iteration by iteration, and country by country in the network.
    E : Pandas Data Frame
        Exposed day by day, iteration by iteration, and country by country in the network.
    I : Pandas Data Frame
        Infected day by day, iteration by iteration, and country by country in the network.
    Q : Pandas Data Frame
        Reported day by day, iteration by iteration, and country by country in the network.
    R : Pandas Data Frame
        Recovered day by day, iteration by iteration, and country by country in the network.
    D : Pandas Data Frame
        Deaths day by day, iteration by iteration, and country by country in the network.
    P : Pandas Data Frame
        Protected day by day, iteration by iteration, and country by country in the network.
    V : Pandas Data Frame
        Vaccinated day by day, iteration by iteration, and country by country in the network.

    """
    print('Simulating Full Network:')
    File='Fittings/Argentina.pckl'    
    Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
    datei=pd.to_datetime(Data['Time'][max(Data.index)])
    TimeSimulation=  (datei + np.arange(days) * timedelta(days=1)).astype('str')
    S=pd.DataFrame(columns=(TimeSimulation))
    S.insert(loc=0, column='Country',value='')
    # E,I,Q,R,D,P,V=S,S,S,S,S,S,S
    S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S,S
    indicec=0
    # for i in np.arange(len(Outputs)):
    VaccinesCritic=list()
    for i in ran:
        # ----------------Set initial populations--------------------------
        print('Setting initial conditions for parameters ',i+1,' of ', len(ran),'parameters: ',i )
        for j in Countries['Country']:
            # print('Country: ',j)
            File='Fittings/'+j+'.pckl'    
            Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
            # -------------------------------------------------------------------
            new_rowS = {'Country':j, TimeSimulation[0]:Outputs[i].S[len(Outputs[0].S)-1]}
            S   = S.append(new_rowS, ignore_index=True)     #| Susceptible Wild
            # ---------------------Wild virus------------------------------------------
            new_rowEw = {'Country':j, TimeSimulation[0]:Outputs[i].Ew[len(Outputs[0].Ew)-1]}
            Ew  = Ew.append(new_rowEw, ignore_index=True)   

            new_rowEwV = {'Country':j, TimeSimulation[0]:Outputs[i].EwV[len(Outputs[0].EwV)-1]}
            EwV = EwV.append(new_rowEwV, ignore_index=True)    

            new_rowEwm = {'Country':j, TimeSimulation[0]:Outputs[i].Ewm[len(Outputs[0].Ewm)-1]}
            Ewm = Ewm.append(new_rowEwm, ignore_index=True)  
            
            new_rowIw = {'Country':j, TimeSimulation[0]:Outputs[i].Iw[len(Outputs[0].Iw)-1]}
            Iw  = Iw.append(new_rowIw, ignore_index=True)      

            new_rowIwm = {'Country':j, TimeSimulation[0]:Outputs[i].Iwm[len(Outputs[0].Iwm)-1]}
            Iwm = Iwm.append(new_rowIwm, ignore_index=True)
            
            new_rowQw = {'Country':j, TimeSimulation[0]:Outputs[i].Qw[len(Outputs[0].Qw)-1]}
            Qw  = Qw.append(new_rowQw, ignore_index=True)

            new_rowQwm = {'Country':j, TimeSimulation[0]:Outputs[i].Qwm[len(Outputs[0].Qwm)-1]}
            Qwm = Qwm.append(new_rowQwm, ignore_index=True)
            
            new_rowIwV = {'Country':j, TimeSimulation[0]:Outputs[i].IwV[len(Outputs[0].IwV)-1]}
            IwV = IwV.append(new_rowIwV, ignore_index=True)       

            new_rowQwV = {'Country':j, TimeSimulation[0]:Outputs[i].QwV[len(Outputs[0].QwV)-1]}
            QwV = QwV.append(new_rowQwV, ignore_index=True)   

            new_rowRw = {'Country':j, TimeSimulation[0]:Outputs[i].Rw[len(Outputs[0].Rw)-1]}
            Rw  = Rw.append(new_rowRw, ignore_index=True)                      #| Recovered Wild
            # ---------------------Mutant virus----------------------------------------
            new_rowEm = {'Country':j, TimeSimulation[0]:Outputs[i].Em[len(Outputs[0].Em)-1]}
            Em  = Em.append(new_rowEm, ignore_index=True)    

            new_rowEmV = {'Country':j, TimeSimulation[0]:Outputs[i].EmV[len(Outputs[0].EmV)-1]}
            EmV = EmV.append(new_rowEmV, ignore_index=True)

            new_rowEmw = {'Country':j, TimeSimulation[0]:Outputs[i].Emw[len(Outputs[0].Emw)-1]}
            Emw =  Emw.append(new_rowEmw, ignore_index=True)
            
            new_rowIm = {'Country':j, TimeSimulation[0]:Outputs[i].Im[len(Outputs[0].Im)-1]}
            Im  = Im.append(new_rowIm, ignore_index=True)   

            new_rowImw = {'Country':j, TimeSimulation[0]:Outputs[i].Imw[len(Outputs[0].Imw)-1]}
            Imw =  Imw.append(new_rowImw, ignore_index=True)
            
            new_rowQm = {'Country':j, TimeSimulation[0]:Outputs[i].Qm[len(Outputs[0].Qm)-1]}
            Qm  = Qm.append(new_rowQm, ignore_index=True)

            new_rowQmw = {'Country':j, TimeSimulation[0]:Outputs[i].Qmw[len(Outputs[0].Qmw)-1]}
            Qmw = Qmw.append(new_rowQmw, ignore_index=True)
            
            new_rowImV = {'Country':j, TimeSimulation[0]:Outputs[i].ImV[len(Outputs[0].ImV)-1]}
            ImV = ImV.append(new_rowImV, ignore_index=True)

            new_rowQmV = {'Country':j, TimeSimulation[0]:Outputs[i].QmV[len(Outputs[0].QmV)-1]}
            QmV = QmV.append(new_rowQmV, ignore_index=True)  

            new_rowRm = {'Country':j, TimeSimulation[0]:Outputs[i].Rm[len(Outputs[0].Rm)-1]}
            Rm  = Rm.append(new_rowRm, ignore_index=True)                                    #| Recovered Mutant
            # ---------------------Common states---------------------------------------
            new_rowD = {'Country':j, TimeSimulation[0]:Outputs[i].D[len(Outputs[0].D)-1]}
            D   = D.append(new_rowD, ignore_index=True)  

            new_rowRb = {'Country':j, TimeSimulation[0]:Outputs[i].Rb[len(Outputs[0].Rb)-1]}
            Rb  = Rb.append(new_rowRb, ignore_index=True)
            
            new_rowP = {'Country':j, TimeSimulation[0]:Outputs[i].P[len(Outputs[0].P)-1]}
            P   = P.append(new_rowP, ignore_index=True)    

            new_rowV1 = {'Country':j, TimeSimulation[0]:Outputs[i].V1[len(Outputs[0].V1)-1]}
            V1  = V1.append(new_rowV1, ignore_index=True)   

            new_rowV2 = {'Country':j, TimeSimulation[0]:Outputs[i].V2[len(Outputs[0].V2)-1]}
            V2  = V2.append(new_rowV2, ignore_index=True)       
        # Simulating the compleate network
        print('Simulating for parameters ',i+1,' of ', len(ran))
        start=len(Outputs[0].S)
        for k in np.arange(1,days):
            # print('Day ',k+1,' of ',TimeSimulation[k+1])
            # indicec=1
            tl = int(1/dt)
            # t = np.linspace(k-1,k-dt,tl)
            t = np.linspace(start+k-1,start+k-dt,tl)            
            v=0
            for j in Countries['Country']:
                # indicec=indicec*indicP-1
                Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
                indi=indicec+v
                # print('Simulating for country ',j,' day ',k+1,' of ', days,'indice ',indi)
                Neig=GetNeighbours(g,j)
                # print('País: ',j)
                Ei=0
                Eim=0
                # ------------------getting Temp profile-----------------------
                Daty,Temp=GetCountriinfo(j, days)
                T=Temp.Temperature
                timeT= pd.to_datetime(Temp.Date)
                global f
                global ini
                # ini=int(np.where(pd.to_datetime(Temp.Date) == pd.to_datetime(Daty.Time.iloc[0]))[0])
                ini=int(np.where(pd.to_datetime(Temp.Date) == pd.to_datetime(datei))[0])
                tb,Beta=T_inv(T,timeT)
                f = interp1d(tb, Beta, kind='cubic')
                # -------------------------------------------------------------
                
                for x in Neig['Neighbours']:
                    Filex='Fittings/'+x+'.pckl' 
                    # Parx,Outx,Datax= pickle.load(open(Filex,"rb"))
                    Parx,Outx,Repox,Datax= pickle.load(open(Filex,"rb"))
                    # Popx=sum(Countries[Countries['Country']==x]['Population'])
                    Popx=Countries.loc[Countries['Country']==x,'Population'].values[-1]
                    infx=Iw.loc[Iw['Country'] == x, TimeSimulation[k-1]].values[-1]+Iwm.loc[Iwm['Country'] == x, TimeSimulation[k-1]].values[-1]
                    repx=Qw.loc[Qw['Country'] == x, TimeSimulation[k-1]].values[-1]+Qwm.loc[Qwm['Country'] == x, TimeSimulation[k-1]].values[-1]
                    infxm=Im.loc[Iw['Country'] == x, TimeSimulation[k-1]].values[-1]+Imw.loc[Iwm['Country'] == x, TimeSimulation[k-1]].values[-1]
                    repxm=Qm.loc[Qw['Country'] == x, TimeSimulation[k-1]].values[-1]+Qmw.loc[Qmw['Country'] == x, TimeSimulation[k-1]].values[-1]
                   
                    betaIx=Parx[i]['beta'].value
                    betaQx=Parx[i]['betaQ'].value
                    
                    Beta=1/Popx*(betaIx*infx + betaQx * repx)
                    Betam=1/Popx*(betaIx*infxm + betaQx * repxm)*Parx[i]['omega'].value
                    efrac=abs(1.0 - m.exp(-Beta*dt)) 
                    efracm=abs(1.0 - m.exp(-Betam*dt)) 
                    Sx=int(400*int(Neig[Neig['Neighbours']==x]['Routes'].values[0]*connectivity))
                    Expo=np.random.binomial(Sx,efrac)
                    Expom=np.random.binomial(Sx,efracm)
                    # print('\t ',x,' - Infectados: ',infx,' - Reportados: ' ,repx, '- Popu: ',Popx, 
                    #       ' - Beta: ',Beta, ' - Exposed individuals ',Expo)
                    Ei=Ei+Expo
                    Eim=Eim+Expom
                # y0=[S.loc[S['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     E.loc[E['Country'] == j, TimeSimulation[k-1]].values[-1]+Ei,
                #     I.loc[I['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     Q.loc[Q['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     R.loc[R['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     D.loc[D['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     P.loc[P['Country'] == j, TimeSimulation[k-1]].values[-1],
                #     V.loc[V['Country'] == j, TimeSimulation[k-1]].values[-1]]
                VAC=V1.loc[V1['Country'] == j, TimeSimulation[k-1]].values[-1] + V2.loc[V2['Country'] == j, TimeSimulation[k-1]].values[-1]
                y0=[S.loc[S['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Ew.loc[Ew['Country'] == j, TimeSimulation[k-1]].values[-1]+Ei,
                    EwV.loc[EwV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Ewm.loc[Ewm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Iw.loc[Iw['Country'] == j, TimeSimulation[k-1]].values[-1],   
                    Iwm.loc[Iwm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qw.loc[Qw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qwm.loc[Qwm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    IwV.loc[IwV['Country'] == j, TimeSimulation[k-1]].values[-1],  
                    QwV.loc[QwV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Rw .loc[Rw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Em.loc[Em['Country'] == j, TimeSimulation[k-1]].values[-1]+Eim,
                    EmV.loc[EmV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Emw.loc[Emw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Im.loc[Im['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Imw.loc[Imw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qm.loc[Qm['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Qmw.loc[Qmw['Country'] == j, TimeSimulation[k-1]].values[-1],
                    ImV.loc[ImV['Country'] == j, TimeSimulation[k-1]].values[-1],
                    QmV.loc[QmV['Country'] == j, TimeSimulation[k-1]].values[-1], 
                    Rm.loc[Rm['Country'] == j, TimeSimulation[k-1]].values[-1],                                   #| Recovered Mutant
                    D.loc[D['Country'] == j, TimeSimulation[k-1]].values[-1],
                    Rb.loc[Rb['Country'] == j, TimeSimulation[k-1]].values[-1],
                    P.loc[P['Country'] == j, TimeSimulation[k-1]].values[-1], 
                    V1.loc[V1['Country'] == j, TimeSimulation[k-1]].values[-1],
                    V2.loc[V2['Country'] == j, TimeSimulation[k-1]].values[-1] ]  

                print('Simuation in node : ',j, ' Iteration: ',i+1, ' of ',len(ran),
                      ' Day:',k+1, ' of ',days)
                
                if  (VAC*100/sum(y0)>50) and (j in Info['Country'].values) and (j in VaccinesCritic == False):
                    Params[i]['rho1'].value=0.0
                    Params[i]['rho2'].value=0.0
                    VaccinesCritic.append(j)
                    
                if  (VAC*100/sum(y0)<=50) and (j not in Info['Country'].values) and (len(VaccinesCritic) >0):
                    if (Params[i]['rho1'].value <= NewRate):
                        Params[i]['rho1'].value=NewRate*len(VaccinesCritic)
                    if (Params[i]['rho1'].value > NewRate) and (Params[i]['rho1'].value <= NewRate*5):
                        Params[i]['rho1'].value=Params[i]['rho1'].value+Params[i]['rho1'].value*len(VaccinesCritic)/10
            
                            
                    # Params[i]['rho2'].value=0.0
                    
                sir_out = pd.DataFrame(simulate(t,dt,y0, Params[i]))
                
                S.at[indi,TimeSimulation[k]]=sir_out.S[len(sir_out.S)-1]
                Ew.at[indi,TimeSimulation[k]]=sir_out.Ew[len(sir_out.Ew)-1]
                EwV.at[indi,TimeSimulation[k]]=sir_out.EwV[len(sir_out.EwV)-1]
                Ewm.at[indi,TimeSimulation[k]]=sir_out.Ewm[len(sir_out.Ewm)-1]
                Iw.at[indi,TimeSimulation[k]]=sir_out.Iw[len(sir_out.Iw)-1]   
                Iwm.at[indi,TimeSimulation[k]]=sir_out.Iwm[len(sir_out.Iwm)-1]
                Qw.at[indi,TimeSimulation[k]]=sir_out.Qw[len(sir_out.Qw)-1]
                Qwm.at[indi,TimeSimulation[k]]=sir_out.Qwm[len(sir_out.Qwm)-1]
                IwV.at[indi,TimeSimulation[k]]=sir_out.IwV[len(sir_out.IwV)-1] 
                QwV.at[indi,TimeSimulation[k]]=sir_out.QwV[len(sir_out.QwV)-1]
                Rw.at[indi,TimeSimulation[k]]=sir_out.Rw[len(sir_out.Rw)-1]
                Em.at[indi,TimeSimulation[k]]=sir_out.Em[len(sir_out.Em)-1]
                EmV.at[indi,TimeSimulation[k]]=sir_out.EmV[len(sir_out.EmV)-1]
                Emw.at[indi,TimeSimulation[k]]=sir_out.Emw[len(sir_out.Emw)-1]
                Im.at[indi,TimeSimulation[k]]=sir_out.Im[len(sir_out.Im)-1]
                Imw.at[indi,TimeSimulation[k]]=sir_out.Imw[len(sir_out.Imw)-1]
                Qm.at[indi,TimeSimulation[k]]=sir_out.Qm[len(sir_out.Qm)-1]
                Qmw.at[indi,TimeSimulation[k]]=sir_out.Qmw[len(sir_out.Qmw)-1]
                ImV.at[indi,TimeSimulation[k]]=sir_out.ImV[len(sir_out.ImV)-1]
                QmV.at[indi,TimeSimulation[k]]=sir_out.QmV[len(sir_out.QmV)-1]
                Rm.at[indi,TimeSimulation[k]]=sir_out.Rm[len(sir_out.Rm)-1]                                 #| Recovered Mutant
                D.at[indi,TimeSimulation[k]]=sir_out.D[len(sir_out.D)-1]
                Rb.at[indi,TimeSimulation[k]]=sir_out.Rb[len(sir_out.Rb)-1]
                P.at[indi,TimeSimulation[k]]=sir_out.P[len(sir_out.P)-1]
                V1.at[indi,TimeSimulation[k]]=sir_out.V1[len(sir_out.V1)-1]
                V2.at[indi,TimeSimulation[k]]=sir_out.V2[len(sir_out.V2)-1]
                v=v+1
        indicec=indi+1

    S=S.sort_values(by=['Country']).reset_index(drop=True)
    Ew=Ew.sort_values(by=['Country']).reset_index(drop=True)
    EwV=EwV.sort_values(by=['Country']).reset_index(drop=True)
    Ewm=Ewm.sort_values(by=['Country']).reset_index(drop=True)
    Iw=Iw.sort_values(by=['Country']).reset_index(drop=True) 
    Iwm=Iwm.sort_values(by=['Country']).reset_index(drop=True)
    Qw= Qw.sort_values(by=['Country']).reset_index(drop=True)
    Qwm=Qwm.sort_values(by=['Country']).reset_index(drop=True)
    IwV=IwV.sort_values(by=['Country']).reset_index(drop=True)
    QwV=QwV.sort_values(by=['Country']).reset_index(drop=True)
    Rw=Rw.sort_values(by=['Country']).reset_index(drop=True)
    Em=Em.sort_values(by=['Country']).reset_index(drop=True)
    EmV=EmV.sort_values(by=['Country']).reset_index(drop=True)
    Emw=Emw.sort_values(by=['Country']).reset_index(drop=True)
    Im=Im.sort_values(by=['Country']).reset_index(drop=True)
    Imw=Imw.sort_values(by=['Country']).reset_index(drop=True)
    Qm=Qm.sort_values(by=['Country']).reset_index(drop=True)
    Qmw=Qmw.sort_values(by=['Country']).reset_index(drop=True)
    ImV=ImV.sort_values(by=['Country']).reset_index(drop=True)
    QmV=QmV.sort_values(by=['Country']).reset_index(drop=True)
    Rm=Rm.sort_values(by=['Country']).reset_index(drop=True)                            #| Recovered Mutant
    D=D.sort_values(by=['Country']).reset_index(drop=True)
    Rb=Rb.sort_values(by=['Country']).reset_index(drop=True)
    P=P.sort_values(by=['Country']).reset_index(drop=True)
    V1=V1.sort_values(by=['Country']).reset_index(drop=True)
    V2=V2.sort_values(by=['Country']).reset_index(drop=True)
    return TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2



#==============================================================================
#------------------------------------------------------------------------------
#                   The main code begins from here    
#------------------------------------------------------------------------------
#==============================================================================
# Create the network
# G,airports = Make_Network()
# sg = ReduceNetwork(G)
# Data=GetDataSets()
# -------------------save information-----------------------
File='Data/Simulation/Networks.pckl'
# pickle.dump([G,sg,Data], open(File, "wb"))
# -------------------load information-----------------------
G,sg,Data= pickle.load(open(File,"rb"))

# Plot the network
# PlotNetwork(G)
# PlotNetwork(sg)

# PlotReducedNetwork(sg)
# Plot the map
# PlotMap(airports,sg)

Countries=GetListOfCountries(sg)
CountriesInfo=GetNodeInformation(sg)
# CountriesInfo.to_csv('Paises.csv',sep=',')
# ShowCountryConections(sg)

# ShowNodeInfo(G)




Vaccines=ShowVaccination(G)
Vaccines['Per Full Vacc']=Vaccines['Total Full vaccinated']*100/Vaccines['Population']
Vaccines['Per Full Vacc Onde Dose']=Vaccines['Total vaccintaed']*100/Vaccines['Population']
Vaccines=Vaccines.sort_values(by=['Per Full Vacc Onde Dose'], ascending=False)
# print(Vaccines.to_latex(index=False))  

# FullSetofParameters,MeanParamDF=CheckParameters(Countries)

# print(MeanParamDF.to_latex(index=True,longtable=True,float_format="%.3f",col_space=1))


# PlotFittings(Countries)
# PlotFittingsContinent(CountriesInfo)
# PlotFittingsGlobal(Countries)
# PlotParameters(CountriesInfo, FullSetofParameters)
# print('Countries with vaccination ongoing: ', Nvac,' of ', len(Countries))
# for i in sorted(LVac):
#     print(i)
# lista=Countries[0:9]
# lista=['Ukraine' ,'Sweden' ,'Singapore' ,'Rwanda' ,'Oman' ,'Montenegro' ,
#        'Malaysia' ,'Kyrgyzstan' ,'Iran' ,'Grenada' ,'Estonia' ,'Cuba' ,
#        'Central African Republic' ,'Bosnia and Herzegovina' ,'Azerbaijan' ,
#        'Austria' ,'Australia' ,'Armenia' ]

# FitFullNetwork(lista[12:18],90)
# FitFullNetwork(['Cuba'],90)
FitFullNetwork(Countries[55:65],120)
# FitFullNetwork(Countries[7:9],90)

# FitFullNetwork(['United States'],30)
# TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_network(sg,CountriesInfo,180,1/24,1)

# --------------------------Current Scenarios---------------------------------
# ran=list(np.arange(10))
# # x=np.arange(0,1,.25)
# # y0=1-x
# # y0[1:len(y0)]=y0[1:len(y0)]-0.25
# # conec=y0
# conec=0.0
# days=360
# Dir='Data/Simulation/'
# for i in ran:                
#     # file='SimulationsCurrentFull0'+str(i)+'.pckl'
#     # file='SimulationsCurrentHalf0'+str(i)+'.pckl'
#     # file='SimulationsCurrentQuart0'+str(i)+'.pckl'
#     file='SimulationsCurrentNull0'+str(i)+'.pckl'
#     TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_network(sg,CountriesInfo,days,1/24,conec,[i])
#     File=Dir+file
#     pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))

# TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_network(sg,CountriesInfo,180,1/24,1,[0,1])
# File='Data/Simulation/SimulationsCurrent01.pckl'
# pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))

# TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_network(sg,CountriesInfo,180,1/24,1,[2,3])
# File='Data/Simulation/SimulationsCurrent02.pckl'
# pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))

# TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_network(sg,CountriesInfo,180,1/24,1,[4,5])
# File='Data/Simulation/SimulationsCurrent03.pckl'
# pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))

# TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_network(sg,CountriesInfo,180,1/24,1,[6,7])
# File='Data/Simulation/SimulationsCurrent04.pckl'
# pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))

# TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_network(sg,CountriesInfo,180,1/24,1,[8,9])
# File='Data/Simulation/SimulationsCurrent05.pckl'
# pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))


# --------------------------Vaccine Scenarios---------------------------------
# ListOfRichCountries=pd.read_csv('Data/VaccinCountries.csv')
# VacRate=Vaccines['Rate'].mean()
# for i in ran:
#     # print(i,type(i))
#     # file='SimulationsVacFull0'+str(i)+'.pckl'
#     # file='SimulationsVacHalf0'+str(i)+'.pckl'
#     # file='SimulationsVacQuart0'+str(i)+'.pckl'
#     file='SimulationsVaNullc0'+str(i)+'.pckl'
#     TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_networkVaccinScenario(sg,CountriesInfo,days,1/24,conec,[i],ListOfRichCountries,VacRate)
#     File=Dir+file
#     pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))


# ListOfRichCountries=pd.read_csv('Data/VaccinCountries.csv')
# VacRate=Vaccines['Rate'].mean()
# TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=Simulate_Full_networkVaccinScenario(sg,CountriesInfo,180,1/24,0.0,[8,9],ListOfRichCountries,VacRate)
# File='Data/Simulation/SimulationsVacNull05.pckl'
# pickle.dump([TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2], open(File, "wb"))



