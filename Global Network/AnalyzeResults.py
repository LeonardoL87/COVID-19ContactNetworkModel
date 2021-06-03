#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 14:44:27 2021

@author: leonardo
"""

#installation
import plotly.express as px
import plotly.io as pio
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================================
# =========================================================================
def TrasposeDFSimulations(Data):
    # Data=Data.sort_values(by='Country')
    # DataPivot=Data.pivot(columns='Population')
    
    # Data.index=np.arange(Data.shape[0])
    
    Columns=('Country','Time','S','Ew','EwV','Ewm','Iw','IwV','Iwm','Qw','QwV','Qwm','Rw',
         'Em','EmV','Emw','Im','ImV','Imw','Qm','QmV','Qmw','Rm','D','Rb','P',
         'V1','V2')
    
    SimulationsDF2=pd.DataFrame(columns=Columns)
    SimulationsDF=SimulationsDF2
    
    Time=pd.to_datetime(Data.columns[2:len(Data.columns)])
    Countries=pd.unique(Data['Country'])
    Population=pd.unique(Data['Population'])
    for i in Countries[[0]]:
        DFi=Data[Data['Country']==i]
        for j in Population[[0]]:
           DFi=DFi[DFi['Population']==j] 
        
    for i in Data.index:
       val=Data.iloc[i][2:len(Data.iloc[0])].values
       col=Data.iloc[i][0]
       count=Data.iloc[i][1]
       SimulationsDF2[col]=val
       SimulationsDF2['Country']=count
       SimulationsDF2['Time']=Time
       if SimulationsDF2.isnull().all().any() == False:
           SimulationsDF=SimulationsDF.append(SimulationsDF2,ignore_index=False)
           SimulationsDF2=pd.DataFrame(columns=Columns)
    return SimulationsDF
# =========================================================================
# =========================================================================
def plotWrold(Data,Information,Area):
    Columns=Data.columns
    countries=pd.unique(Data['Country'])
    Populations=pd.unique(Data['Population'])
    Dic={}
    dic={}
    
    countries=Information['Country'][Information['Continent']==Area].values
    s,ew,ewV,ewm,iw,iwV,iwm,qw,qwV,qwm,rw,em,emV,emw,im,imV,imw,qm,qmV,qmw,rm,d,rb,p,v1,v2 = np.zeros((26,Data.shape[1]-2))
    S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2 = np.zeros((26,Data.shape[1]-2))
    for i in countries:
        DataFiltred=Data[Data['Country']==i]
        Mean=DataFiltred.groupby('Population',as_index=False).mean()
        STD=DataFiltred.groupby('Population',as_index=False).std()
        for j in Populations:
            Dic[j]=Mean[Mean['Population']==j]
            dic[j]=STD[STD['Population']==j]
            # Dic=pd.DataFrame.from_dict(Dic)
            

        D+=Dic['D'].iloc[-1,1:len(Dic['D'].iloc[-1])].values.astype(float)
        Em+=Dic['Em'].iloc[-1,1:len(Dic['Em'].iloc[-1])].values.astype(float)
        EmV+=Dic['EmV'].iloc[-1,1:len(Dic['EmV'].iloc[-1])].values.astype(float)
        Emw+=Dic['Emw'].iloc[-1,1:len(Dic['Emw'].iloc[-1])].values.astype(float)
        Ew+=Dic['Ew'].iloc[-1,1:len(Dic['Ew'].iloc[-1])].values.astype(float)
        Ewm+=Dic['Ewm'].iloc[-1,1:len(Dic['Ewm'].iloc[-1])].values.astype(float)
        EwV+=Dic['EwV'].iloc[-1,1:len(Dic['EwV'].iloc[-1])].values.astype(float)
        Im+=Dic['Im'].iloc[-1,1:len(Dic['Im'].iloc[-1])].values.astype(float)
        ImV+=Dic['ImV'].iloc[-1,1:len(Dic['ImV'].iloc[-1])].values.astype(float)
        Imw+=Dic['Imw'].iloc[-1,1:len(Dic['Imw'].iloc[-1])].values.astype(float)
        Iw+=Dic['Iw'].iloc[-1,1:len(Dic['Iw'].iloc[-1])].values.astype(float)
        Iwm+=Dic['Iwm'].iloc[-1,1:len(Dic['Iwm'].iloc[-1])].values.astype(float)
        IwV+=Dic['IwV'].iloc[-1,1:len(Dic['IwV'].iloc[-1])].values.astype(float)
        P+=Dic['P'].iloc[-1,1:len(Dic['P'].iloc[-1])].values.astype(float)
        Qm+=Dic['Qm'].iloc[-1,1:len(Dic['Qm'].iloc[-1])].values.astype(float)
        QmV+=Dic['QmV'].iloc[-1,1:len(Dic['QmV'].iloc[-1])].values.astype(float)
        Qmw+=Dic['Qmw'].iloc[-1,1:len(Dic['Qmw'].iloc[-1])].values.astype(float)
        Qw+=Dic['Qw'].iloc[-1,1:len(Dic['Qw'].iloc[-1])].values.astype(float)
        Qwm+=Dic['Qwm'].iloc[-1,1:len(Dic['Qwm'].iloc[-1])].values.astype(float)
        QwV+=Dic['QwV'].iloc[-1,1:len(Dic['QwV'].iloc[-1])].values.astype(float)
        Rb+=Dic['Rb'].iloc[-1,1:len(Dic['Rb'].iloc[-1])].values.astype(float)
        Rm+=Dic['Rm'].iloc[-1,1:len(Dic['Rm'].iloc[-1])].values.astype(float)
        Rw+=Dic['Rw'].iloc[-1,1:len(Dic['Rw'].iloc[-1])].values.astype(float)
        S+=Dic['S'].iloc[-1,1:len(Dic['S'].iloc[-1])].values.astype(float)
        V1+=Dic['V1'].iloc[-1,1:len(Dic['V1'].iloc[-1])].values.astype(float)
        V2+=Dic['V2'].iloc[-1,1:len(Dic['V2'].iloc[-1])].values.astype(float)
        
        d+=dic['D'].iloc[-1,1:len(dic['D'].iloc[-1])].values.astype(float)
        em+=dic['Em'].iloc[-1,1:len(dic['Em'].iloc[-1])].values.astype(float)
        emV+=dic['EmV'].iloc[-1,1:len(dic['EmV'].iloc[-1])].values.astype(float)
        emw+=dic['Emw'].iloc[-1,1:len(dic['Emw'].iloc[-1])].values.astype(float)
        ew+=dic['Ew'].iloc[-1,1:len(dic['Ew'].iloc[-1])].values.astype(float)
        ewm+=dic['Ewm'].iloc[-1,1:len(dic['Ewm'].iloc[-1])].values.astype(float)
        ewV+=dic['EwV'].iloc[-1,1:len(dic['EwV'].iloc[-1])].values.astype(float)
        im+=dic['Im'].iloc[-1,1:len(dic['Im'].iloc[-1])].values.astype(float)
        imV+=dic['ImV'].iloc[-1,1:len(dic['ImV'].iloc[-1])].values.astype(float)
        imw+=dic['Imw'].iloc[-1,1:len(dic['Imw'].iloc[-1])].values.astype(float)
        iw+=dic['Iw'].iloc[-1,1:len(dic['Iw'].iloc[-1])].values.astype(float)
        iwm+=dic['Iwm'].iloc[-1,1:len(dic['Iwm'].iloc[-1])].values.astype(float)
        iwV+=dic['IwV'].iloc[-1,1:len(dic['IwV'].iloc[-1])].values.astype(float)
        p+=dic['P'].iloc[-1,1:len(dic['P'].iloc[-1])].values.astype(float)
        qm+=dic['Qm'].iloc[-1,1:len(dic['Qm'].iloc[-1])].values.astype(float)
        qmV+=dic['QmV'].iloc[-1,1:len(dic['QmV'].iloc[-1])].values.astype(float)
        qmw+=dic['Qmw'].iloc[-1,1:len(dic['Qmw'].iloc[-1])].values.astype(float)
        qw+=dic['Qw'].iloc[-1,1:len(dic['Qw'].iloc[-1])].values.astype(float)
        qwm+=dic['Qwm'].iloc[-1,1:len(dic['Qwm'].iloc[-1])].values.astype(float)
        qwV+=dic['QwV'].iloc[-1,1:len(dic['QwV'].iloc[-1])].values.astype(float)
        rb+=dic['Rb'].iloc[-1,1:len(dic['Rb'].iloc[-1])].values.astype(float)
        rm+=dic['Rm'].iloc[-1,1:len(dic['Rm'].iloc[-1])].values.astype(float)
        rw+=dic['Rw'].iloc[-1,1:len(dic['Rw'].iloc[-1])].values.astype(float)
        s+=dic['S'].iloc[-1,1:len(dic['S'].iloc[-1])].values.astype(float)
        v1+=dic['V1'].iloc[-1,1:len(dic['V1'].iloc[-1])].values.astype(float)
        v2+=dic['V2'].iloc[-1,1:len(dic['V2'].iloc[-1])].values.astype(float)
    Q=Qw+QwV+Qm+QmV+Qwm+Qmw
    q=qw+qwV+qm+qmV+qwm+qmw
    R=Rw+Rm+Rb
    r=rw+rm+rb
    Total=Q+R+D
    total=q+r+d
    V=V1+V2
    v=v1+v2
    time = pd.to_datetime(Data.columns[2:len(Data.columns)])
    # Recovered = np.asarray(Data.Recovered,dtype=int)
    # Deaths = np.asarray(Data.Deaths,dtype=int)
    # Confirmed = np.asarray(Data.Confirmed,dtype=int)
    # Active = np.asarray(Data.Active,dtype=int)
    #-------------------------------------------------------------------------
    # Plot Reult
    FigName='Fittings/Plots/'+i+'.png'
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
                          
    # ax.plot(time, Q-q, '--r', alpha=0.5, lw=1)  
    ax.plot(time, Total, 'r', alpha=0.5, lw=2, label='Total Cases')
    # ax.plot(time, Q+q, '--r', alpha=0.5, lw=1)
    ax.fill_between(time,  Total-total, Total+total,color='r', alpha=0.2)
    
    # ax.plot(time, R-r, '--g', alpha=0.5, lw=1)  
    ax.plot(time, R, 'g', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(time, R+r, '--g', alpha=0.5, lw=1)  
    ax.fill_between(time,  R-r, R+r,color='g', alpha=0.2)

    # ax.plot(time, D-d, '--y', alpha=0.5, lw=1)  
    ax.plot(time, D, 'y', alpha=0.5, lw=2, label='Deaths')
    # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
    ax.fill_between(time,  D-d, D+d,color='y', alpha=0.2)
    
    ax.plot(time, V, 'b', alpha=0.5, lw=2, label='Vaccinated')
    # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
    ax.fill_between(time,  V-v, V+v,color='b', alpha=0.2)

    # ax.plot(time,Active, 'or', alpha=0.5, lw=2, label='Active')
    # ax.plot(time,Recovered, 'og', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(time,Deaths, 'oy', alpha=0.5, lw=2, label='Deaths')
    # ax.plot(time,Confirmed, 'ok', alpha=0.5, lw=2, label='Total Cases')
    
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.set_title(Area)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    # fig.savefig(FigName,dpi=600)
    # plt.close(fig)
# =========================================================================
# =========================================================================
def plotCountry(Data,Country):
    Columns=Data.columns
    countries=pd.unique(Data['Country'])
    Populations=pd.unique(Data['Population'])
    Dic={}
    dic={}
    
    # countries=Information['Country'][Information['Continent']==Area].values
    s,ew,ewV,ewm,iw,iwV,iwm,qw,qwV,qwm,rw,em,emV,emw,im,imV,imw,qm,qmV,qmw,rm,d,rb,p,v1,v2 = np.zeros((26,Data.shape[1]-2))
    S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2 = np.zeros((26,Data.shape[1]-2))
    # for i in countries:
    DataFiltred=Data[Data['Country']==Country]
    Mean=DataFiltred.groupby('Population',as_index=False).mean()
    STD=DataFiltred.groupby('Population',as_index=False).std()
    for j in Populations:
        Dic[j]=Mean[Mean['Population']==j]
        dic[j]=STD[STD['Population']==j]
        # Dic=pd.DataFrame.from_dict(Dic)
        

    D+=Dic['D'].iloc[-1,1:len(Dic['D'].iloc[-1])].values.astype(float)
    Em+=Dic['Em'].iloc[-1,1:len(Dic['Em'].iloc[-1])].values.astype(float)
    EmV+=Dic['EmV'].iloc[-1,1:len(Dic['EmV'].iloc[-1])].values.astype(float)
    Emw+=Dic['Emw'].iloc[-1,1:len(Dic['Emw'].iloc[-1])].values.astype(float)
    Ew+=Dic['Ew'].iloc[-1,1:len(Dic['Ew'].iloc[-1])].values.astype(float)
    Ewm+=Dic['Ewm'].iloc[-1,1:len(Dic['Ewm'].iloc[-1])].values.astype(float)
    EwV+=Dic['EwV'].iloc[-1,1:len(Dic['EwV'].iloc[-1])].values.astype(float)
    Im+=Dic['Im'].iloc[-1,1:len(Dic['Im'].iloc[-1])].values.astype(float)
    ImV+=Dic['ImV'].iloc[-1,1:len(Dic['ImV'].iloc[-1])].values.astype(float)
    Imw+=Dic['Imw'].iloc[-1,1:len(Dic['Imw'].iloc[-1])].values.astype(float)
    Iw+=Dic['Iw'].iloc[-1,1:len(Dic['Iw'].iloc[-1])].values.astype(float)
    Iwm+=Dic['Iwm'].iloc[-1,1:len(Dic['Iwm'].iloc[-1])].values.astype(float)
    IwV+=Dic['IwV'].iloc[-1,1:len(Dic['IwV'].iloc[-1])].values.astype(float)
    P+=Dic['P'].iloc[-1,1:len(Dic['P'].iloc[-1])].values.astype(float)
    Qm+=Dic['Qm'].iloc[-1,1:len(Dic['Qm'].iloc[-1])].values.astype(float)
    QmV+=Dic['QmV'].iloc[-1,1:len(Dic['QmV'].iloc[-1])].values.astype(float)
    Qmw+=Dic['Qmw'].iloc[-1,1:len(Dic['Qmw'].iloc[-1])].values.astype(float)
    Qw+=Dic['Qw'].iloc[-1,1:len(Dic['Qw'].iloc[-1])].values.astype(float)
    Qwm+=Dic['Qwm'].iloc[-1,1:len(Dic['Qwm'].iloc[-1])].values.astype(float)
    QwV+=Dic['QwV'].iloc[-1,1:len(Dic['QwV'].iloc[-1])].values.astype(float)
    Rb+=Dic['Rb'].iloc[-1,1:len(Dic['Rb'].iloc[-1])].values.astype(float)
    Rm+=Dic['Rm'].iloc[-1,1:len(Dic['Rm'].iloc[-1])].values.astype(float)
    Rw+=Dic['Rw'].iloc[-1,1:len(Dic['Rw'].iloc[-1])].values.astype(float)
    S+=Dic['S'].iloc[-1,1:len(Dic['S'].iloc[-1])].values.astype(float)
    V1+=Dic['V1'].iloc[-1,1:len(Dic['V1'].iloc[-1])].values.astype(float)
    V2+=Dic['V2'].iloc[-1,1:len(Dic['V2'].iloc[-1])].values.astype(float)
    
    d+=dic['D'].iloc[-1,1:len(dic['D'].iloc[-1])].values.astype(float)
    em+=dic['Em'].iloc[-1,1:len(dic['Em'].iloc[-1])].values.astype(float)
    emV+=dic['EmV'].iloc[-1,1:len(dic['EmV'].iloc[-1])].values.astype(float)
    emw+=dic['Emw'].iloc[-1,1:len(dic['Emw'].iloc[-1])].values.astype(float)
    ew+=dic['Ew'].iloc[-1,1:len(dic['Ew'].iloc[-1])].values.astype(float)
    ewm+=dic['Ewm'].iloc[-1,1:len(dic['Ewm'].iloc[-1])].values.astype(float)
    ewV+=dic['EwV'].iloc[-1,1:len(dic['EwV'].iloc[-1])].values.astype(float)
    im+=dic['Im'].iloc[-1,1:len(dic['Im'].iloc[-1])].values.astype(float)
    imV+=dic['ImV'].iloc[-1,1:len(dic['ImV'].iloc[-1])].values.astype(float)
    imw+=dic['Imw'].iloc[-1,1:len(dic['Imw'].iloc[-1])].values.astype(float)
    iw+=dic['Iw'].iloc[-1,1:len(dic['Iw'].iloc[-1])].values.astype(float)
    iwm+=dic['Iwm'].iloc[-1,1:len(dic['Iwm'].iloc[-1])].values.astype(float)
    iwV+=dic['IwV'].iloc[-1,1:len(dic['IwV'].iloc[-1])].values.astype(float)
    p+=dic['P'].iloc[-1,1:len(dic['P'].iloc[-1])].values.astype(float)
    qm+=dic['Qm'].iloc[-1,1:len(dic['Qm'].iloc[-1])].values.astype(float)
    qmV+=dic['QmV'].iloc[-1,1:len(dic['QmV'].iloc[-1])].values.astype(float)
    qmw+=dic['Qmw'].iloc[-1,1:len(dic['Qmw'].iloc[-1])].values.astype(float)
    qw+=dic['Qw'].iloc[-1,1:len(dic['Qw'].iloc[-1])].values.astype(float)
    qwm+=dic['Qwm'].iloc[-1,1:len(dic['Qwm'].iloc[-1])].values.astype(float)
    qwV+=dic['QwV'].iloc[-1,1:len(dic['QwV'].iloc[-1])].values.astype(float)
    rb+=dic['Rb'].iloc[-1,1:len(dic['Rb'].iloc[-1])].values.astype(float)
    rm+=dic['Rm'].iloc[-1,1:len(dic['Rm'].iloc[-1])].values.astype(float)
    rw+=dic['Rw'].iloc[-1,1:len(dic['Rw'].iloc[-1])].values.astype(float)
    s+=dic['S'].iloc[-1,1:len(dic['S'].iloc[-1])].values.astype(float)
    v1+=dic['V1'].iloc[-1,1:len(dic['V1'].iloc[-1])].values.astype(float)
    v2+=dic['V2'].iloc[-1,1:len(dic['V2'].iloc[-1])].values.astype(float)
    
    Q=Qw+QwV+Qm+QmV+Qwm+Qmw
    q=qw+qwV+qm+qmV+qwm+qmw
    R=Rw+Rm+Rb
    r=rw+rm+rb
    Total=Q+R+D
    total=q+r+d
    V=V1+V2
    v=v1+v2
    time = pd.to_datetime(Data.columns[2:len(Data.columns)])
    # Recovered = np.asarray(Data.Recovered,dtype=int)
    # Deaths = np.asarray(Data.Deaths,dtype=int)
    # Confirmed = np.asarray(Data.Confirmed,dtype=int)
    # Active = np.asarray(Data.Active,dtype=int)
    #-------------------------------------------------------------------------
    # Plot Reult
    FigName='Fittings/Plots/'+Country+'.png'
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
                          
    # ax.plot(time, Q-q, '--r', alpha=0.5, lw=1)  
    ax.plot(time, Total, 'r', alpha=0.5, lw=2, label='Total Cases')
    # ax.plot(time, Q+q, '--r', alpha=0.5, lw=1)
    ax.fill_between(time,  Total-total, Total+total,color='r', alpha=0.2)
    
    # ax.plot(time, R-r, '--g', alpha=0.5, lw=1)  
    ax.plot(time, R, 'g', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(time, R+r, '--g', alpha=0.5, lw=1)  
    ax.fill_between(time,  R-r, R+r,color='g', alpha=0.2)

    # ax.plot(time, D-d, '--y', alpha=0.5, lw=1)  
    ax.plot(time, D, 'y', alpha=0.5, lw=2, label='Deaths')
    # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
    ax.fill_between(time,  D-d, D+d,color='y', alpha=0.2)
    
    ax.plot(time, V, 'b', alpha=0.5, lw=2, label='Vaccinated')
    # ax.plot(time, D+d, '--y', alpha=0.5, lw=1)  
    ax.fill_between(time,  V-v, V+v,color='b', alpha=0.2)

    # ax.plot(time,Active, 'or', alpha=0.5, lw=2, label='Active')
    # ax.plot(time,Recovered, 'og', alpha=0.5, lw=2, label='Recovered')
    # ax.plot(time,Deaths, 'oy', alpha=0.5, lw=2, label='Deaths')
    # ax.plot(time,Confirmed, 'ok', alpha=0.5, lw=2, label='Total Cases')
    
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number')
    ax.set_ylim(bottom=0)
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    ax.set_title(Country)
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
    # fig.savefig(FigName,dpi=600)
    # plt.close(fig)
    

# =========================================================================
# =========================================================================
def MergeDataSimulations(Path):
    path, dirs, files = next(os.walk(Path))
    # file_count = len(files)
    # DataFramei=list()
    itera=0
    for i in files:
        File=Path+i
        TimeSimulation,S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2=pickle.load(open(File,"rb"))
        S['Population']='S'
        Ew['Population']='Ew'
        EwV['Population']='EwV'
        Ewm['Population']='Ewm'
        Iw['Population']='Iw'
        IwV['Population']='IwV'
        Iwm['Population']='Iwm'
        Qw['Population']='Qw'
        QwV['Population']='QwV'
        Qwm['Population']='Qwm'
        Rw['Population']='Rw'
        Em['Population']='Em'
        EmV['Population']='EmV'
        Emw['Population']='Emw'
        Im['Population']='Im'
        ImV['Population']='ImV'
        Imw['Population']='Imw'
        Qm['Population']='Qm'
        QmV['Population']='QmV'
        Qmw['Population']='Qmw'
        Rm['Population']='Rm'
        D['Population']='D'
        Rb['Population']='Rb'
        P['Population']='P'
        V1['Population']='V1'
        V2['Population']='V2'
        # DataFramei.append([S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,Emw,
                            # Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2])
        Simulations= pd.concat([S,Ew,EwV,Ewm,Iw,IwV,Iwm,Qw,QwV,Qwm,Rw,Em,EmV,
              Emw,Im,ImV,Imw,Qm,QmV,Qmw,Rm,D,Rb,P,V1,V2])
        if itera == 0:
            SimulationsDF=pd.DataFrame(columns=Simulations.columns)
            
        SimulationsDF=pd.concat([SimulationsDF,Simulations])
        itera+=1
            
        # SimulationsDF=
    # SimulationsDF=pd.DataFrame(DataFramei,columns=Simulations.columns)
    first_column = SimulationsDF.pop('Population')
    SimulationsDF.insert(0, 'Population', first_column)
    return TimeSimulation,SimulationsDF

# =========================================================================
# =========================================================================
def MergeData(CountriesInformation):
    print('Analizinf the fittings:')
    DataModelFinal=pd.DataFrame()
    DataFinal=pd.DataFrame()
    DataParams=pd.DataFrame()
    # Columns=('alpha0','alpha1','beta','betaQ','delta',
    # 'gamma','kappa0','kappa1','lambda0',
    # 'lambda1','rho','tau0','tau1')
    Columns= ('alpha0','alpha1','beta','betaQ','delta','deltaV1','deltaV2',
              'gamma','kappa0','kappa1','lambda0','lambda1','nuV','omega',
              'rho1','rho2','tau0','tau1')
    # -------------------------------------------
    for i in CountriesInformation['Country']:
        # i='Brazil'
        print('Country: ',i)
        File='Fittings/'+i+'.pckl'    
        # Params,Outputs,Data= pickle.load(open(File,"rb"))
        Params,Outputs,Report,Data= pickle.load(open(File,"rb"))
        Parameters=pd.DataFrame(columns= Columns,index=np.arange(len(Params)))
        Time=Data['Time'].tolist()
        Data['Country']=i
        # -------------------------------------
        for j in np.arange(len(Outputs)):
            Outputs[j]['Date']=Time      
        OutputsDF=pd.concat(Outputs)
        OutputsDF=OutputsDF.groupby('Date',as_index=False).mean()
        OutputsDF['Country']=i
        for j in np.arange(len(Params)):
            for k in np.arange(len(Columns)):
                # print(Columns[k], Params[j][Columns[k]].value)
                Parameters.at[j,Columns[k]]=Params[j][Columns[k]].value
        # Parameters=Parameters.mean(axis=0)
        Parameters['Country']=i
        # Parameters=Parameters.groupby('Country',as_index=False).mean()
    
        # --------------------------------------------------
        DataModelFinal=pd.concat([DataModelFinal,OutputsDF])  
        DataFinal=pd.concat([DataFinal,Data])  
        DataParams=pd.concat([DataParams,Parameters])   
    return DataModelFinal,DataFinal,DataParams
# =========================================================================
# =========================================================================
def PlotParamsGroup(Data,group_by):
    Dir='Fittings/Parameters Distributions/'
    Countries=pd.unique(Data[group_by])
    Columns=Data.columns
    Columns=Columns[0:13]
    for i in Countries:
        print('Ploting distributions for: ',i)
        Name=Dir+i+'.png'         
        Params=Data[Data[group_by]==i]
        pos=1
        fig=plt.figure()
        manager = plt.get_current_fig_manager()
        manager.window.showMaximized()
       
        for j in Columns[0:len(Columns)]:
            ax=plt.subplot(2,7,pos)
            if sum(Params[j])>(1e-6*len(Columns)):
                Params[j].plot.density()
                Params[j].plot.hist()
                if j=='betaQ':
                    j='beta_Q'
                ax=plt.title("$"+"\\"+j+"$")
                ax=plt.suptitle(i)
                plt.show()
            pos=pos+1
        
        fig.set_size_inches((12, 11), forward=True)
        fig.savefig(Name, dpi=300) # Change is over here
        plt.close(fig)
# =========================================================================
# =========================================================================
      
pio.renderers.default='browser'

# Path='Data/Simulation/Current scenario 2/'
# Path='Data/Simulation/360 Days/Current/Full/'
# Path='Data/Simulation/360 Days/Current/Half/'
# Path='Data/Simulation/360 Days/Current/Quart/'
Path='Data/Simulation/360 Days/Current/Null/'

Tsimul,FullSimulations= MergeDataSimulations(Path)



FullSimulationsFinal=TrasposeDFSimulations(FullSimulations)

CountriesInfo=pd.read_csv('Paises.csv', header=0)
Vaccines=pd.read_csv('Vaccines.csv', header=0)

# FullSimulationsFinal=CountriesInfo.merge(FullSimulationsFinal, how='right', on='Country')

# Clean DF
for i in CountriesInfo.index:
    print(i)
    CountriesInfo['Code'][i]=CountriesInfo['Code'][i].replace('\'', '')
    CountriesInfo['Continent'][i]=CountriesInfo['Continent'][i].replace('\'', '')
    
# CountriesInfo.set_index('Country').join(Vaccines.set_index('Country'))
# CountriesInfo.join(Vaccines.set_index('Country'), on='Country')
CountriesInfo.set_index('Country')
Vaccines.set_index('Country')
CountriesInfoVac=CountriesInfo.merge(Vaccines, how='left', on='Country')

CountriesInfoVac=CountriesInfoVac.fillna(0)


DataModelFinal,DataFinal,DataParams=MergeData(CountriesInfoVac)

DataModelFinalforPlot=DataModelFinal.merge(CountriesInfoVac, how='left', on='Country')
DataParamsForPlot=DataParams.merge(CountriesInfoVac, how='left', on='Country')
Q=DataModelFinalforPlot['Qw']+DataModelFinalforPlot['QwV']+DataModelFinalforPlot['Qwm']+DataModelFinalforPlot['Qm']+DataModelFinalforPlot['QmV']+DataModelFinalforPlot['Qmw']
R=DataModelFinalforPlot['Rw']+DataModelFinalforPlot['Rm']+DataModelFinalforPlot['Rb']
D=DataModelFinalforPlot['D']
DataModelFinalforPlot['Total Cases']=Q+R+D



# PlotParamsGroup(DataParamsForPlot,'Continent')
# 


# fig = px.scatter_geo(DataModelFinalforPlot, 
#                          color= "Total Cases",
#                          locations="Code",
#                          hover_name="Country", 
#                          size="Total Cases",
#                          animation_frame="Date",
#                          title='Total Cases',
#                          projection="natural earth",)
# fig.show()
# fig.write_html("Mapa/MapTotal.html")

for j in pd.unique(CountriesInfo['Continent']):    
    plotWrold(FullSimulations,CountriesInfo,j)
    

plotCountry(FullSimulations,'Argentina')    
# plotWrold(FullSimulations,CountriesInfo,'North America')    
    
# plotWrold(FullSimulations,CountriesInfo,'Europe')    
    
    