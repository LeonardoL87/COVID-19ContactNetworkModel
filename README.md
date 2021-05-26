# COVID-19 Contact Network Model
Full Contact Network for SARS-CoV-2 
The proposed model is based on a contact network approach. In our model, each node represents a country within a list of $175$ countries. The number of countries comes from taking those that report epidemiological data (see data section below). The connections between nodes are deduced from the air connectivity between countries, the number of flights and their frequency. The air routes determine the connection between the nodes of the network and the number of flights and frequency of the same determines the degree of connection. In this way, the weight $w_ {ji} \in [0, N] \subseteq  \mathcal{N}$ that connects node $i$ with node $j$ is defined as the number of air routes between both nodes. 

At the node level, to model the infection dynamics of SARS-CoV-2 in a population, a stochastic compartmental model was implemented with a discrete population consisting of $26$ equations and $16$ parameters as can be seen in Eq. \ref{eq:system_equations}. The model contemplates the emergence of new strains of the virus. To model this aspect, a logistic-type function was used that calculates the probability that a new strain will emerge in a given country (network node) at time $t$ (Eq. \ref{eq:probM}). On the other hand, a stochastic variable implemented by means of a binomial probability function calculates the appearance of individuals infected with a mutant strain, taking into account the probability as a function of $t$ given by the logistic-type function. The probability is conditioned by the mean of the total number of infected individuals per million inhabitants with the wild strain, in this way the incidence over time is considered and the circulation of the virus in the population is taken into account. To establish the probability that a mutant strain will emerge, the work carried out by \cite{garcia2021sword} was taken into account, where the time required for a mutant strain to appear in a given population is determined given the number of generations per instant of time. In this way, it is established that the probability approaches $1$ as time gets longer. 

In Fig. \ref{fig:fig1} you can see the diagram that illustrates the proposed mechanistic model. The compartments are:
\begin{itemize}
    \item $S$: susceptible population, 
    \item $Ew$:  population exposed to wild strain,                              
    \item $EwV$: population vaccinated exposed to wild strain,                         
    \item $Ewm$: population recovered from wild strain exposed to mutant strain,
    \item $Iw$:  population infected with wild strain (non detected),                           
    \item $Iwm$: population recovered from wild strain infected with mutant strain (non detected),
    \item $Qw$:  population infected with wild strain (detected),                 
    \item $Qwm$: population recovered from wild strain infected with mutant strain (detected),
    \item $IwV$: population vaccinated infected to wild strain (non detected),                      
    \item $QwV$: population vaccinated infected to wild strain (detected),                              
    \item $Rw$:   population recovered from wild strain,                 
    \item $Em$:  population exposed to mutant strain,                             
    \item $EmV$: population vaccinated exposed to mutant strain,                                
    \item $Emw$: population recovered from mutant strain exposed to wild strain,
    \item $Im$:  population infected with mutant strain (non detected),                             
    \item $Imw$: population recovered from mutant strain infected with wild strain (non detected),
    \item $Qm$:  population infected with mutant strain (detected),                        
    \item $Qmw$: population recovered from mutant strain infected with wild strain (detected),
    \item $ImV$: population vaccinated infected to mutant strain (non detected),                            
    \item $QmV$: population vaccinated infected to mutant strain (detected),                                 
    \item $Rm$:  population recovered from mutant strain,                    
    \item $D$:   population of deaths as a result of the virus,      
    \item $Rb$:  population recovered from both strains,
    \item $P$:   protected or isolated susceptible population,                                   
    \item $V1$:  population vaccinated with one dose,       
    \item $V2$: population vaccinated with two doses. 
\end{itemize}

The susceptible population ($S$) can move to the protected compartment ($P$) at rate $\alpha$, and come back to $S$ at rate $\tau$. Susceptible population can be infected by both viral strains (wild and mutant) at rates $\beta_w$ and $\beta_w$ respectively. The exposed population ($E_w, E_m, E_{wm}$ and $E_{mw}$) move to infected compartment ($I_w, I_m, I_{wm}$ and $I_{mw}$) ) after the incubation period $\gamma$. The infected population is detected at rate $\delta$ and moved to compartments ($Q_w, Q_m, Q_{wm}$ and $Q_{mw}$) depending on the strain which they are infected. Only non vaccinated population can move to death compartment($D$) and we assume that first dose also provides a partial protection against infection of both strains with the introduction of parameter $\nu_V$. For simplicity, population fully vaccinated assume immune to both strains. The population that has been infected by both strains acquires total immunity and passes into the $R_T$ compartment. On the other hand, having undergone a cycle of infection by one of the two strains (wild or mutant) provides immunity against that strain but not against the other. In this way, the population passes to the $R_w$ or $R_m$ compartment depending on the strain with which they were infected.

The infection rate in the model was modeled in two different ways. On the one hand, a seasonal rate is contemplated in, which is calculated taking into account a sinusoidal function dependent on time $t$ as can bee seen in the model equations bellow. On the other hand, when the effect of temperature is contemplated, a function $T(t)$ is used. This function measures the effect that daily temperature has on the infection rate and is calculated as an inverse relationship to the mean average temperature of the last $10$ years as shown in Eq. \ref{eq:T_inv}.

The model runs on a daily time scale, the passage of time between days t0 and t1 is $1/24$, which means that each day is divided into $24$ hours. Transitions between nodes, that is, the population shift from one node to another is carried out at the beginning of each integration step. This assumes a number of daily flights arriving at node $x_i$ from its neighbors $x_j$ at the beginning of each integration step. These income and expenses are taken into account when setting the initial values $y_0$ of the system for each node. 

Only the exposed population to wild or mutant strains are taken into account for the transfer function between nodes. This hypothesis attempts to simulate individuals who may escape the control of airlines and governments through PCRs.

The fraction of exposed individuals that can move from node $i$ to node $j$ on a daily basis is calculated taking into account the probability of infection at node i. To calculate this fraction, proceed in the same way as before (Eq. \ref{eq:Prob} and Eq. \ref{eq:betas}). The value of the population exposed in this case corresponds to the cargo capacity of the airplanes (between $200$ and $400$ passengers) and the number of airplanes that daily communicate between node $i$ and node $j$

The model's algorithm can be summarize as follows:

\begin{algorithm}[H]
\SetAlgoLined
 \KwData{Create Data Frame with simulations}
 \For{$i$ $\in$ Parameters}{
%   Set initial population \;
      \For{$j$ $\in$ Countries}{
      initialization\;
      \KwData{Load parameters $i$ for country $j$}
      \KwData{Set initial conditions $y^i_0$ for country $j$}
          \For{$k$ $\in$ Days}{
              \For{$l$ $\in$ Countries}{
              \KwData{Neighbors: list of neighbors of $l$}
              \KwData{Importations: number of new exposed population imported to country $l$}
              \KwData{Day Simulation: Simulate country $l$}
              }
          }
      }
%   \eIf{condition}{
%   instructions1\;
%   instructions2\;
%   }{
%   instructions3\;
%   }
 }
 \caption{Model's algorithm}
 \label{alg:ModelAlg}
\end{algorithm}

\clearpage
% ========================================================================================================
\subsection{Model equations}
The discrete stochastic equations of the model are showed below.
\begin{equation}
\begin{split}
\centering
    S(t)   =& S(t-\delta t)   + released(t)    - exposedw(t)  - exposedm(t) - protected(t)- vaccinated1(t), \\
    Ew(t)  =& Ew(t-\delta t)  + exposedw(t)    - infectionw(t), \\                                   
    EwV(t) =& EwV(t-\delta t) + exposedVw(t)   - infectionVw(t), \\                                  
    Ewm(t) =& Ewm(t-\delta t) + exposedwm(t)   - infectionwm(t), \\
    Iw(t)  =& Iw(t-\delta t)  + infectionw(t)  - detectedw(t),\\                                     
    Iwm(t) =& Iwm(t-\delta t) + infectionwm(t) - detectedwm(t)\\
    Qw(t)  =& Qw(t-\delta t)  + detectedw(t)   - recoveryw(t)   - deathsw(t),\\                         
    Qwm(t) =& Qwm(t-\delta t) + detectedwm(t)  - recoverywm(t)  - deathswm(t),\\
    IwV(t) =& IwV(t-\delta t) + infectionVw(t) - detectedVw(t),\\                                    
    QwV(t) =& QwV(t-\delta t) + detectedVw(t)  - recoveryVw(t),\\                                    
    Rw(t)  =& Rw(t-\delta t)  + recoveryw(t)   + recoveryVw(t)  - exposedmw(t),\\                        
    Em(t)  =& Em(t-\delta t)  + exposedm(t)    - infectionm(t),\\                                    
    EmV(t) =& EmV(t-\delta t) + exposedVm(t)   - infectionVm(t),\\                                   
    Emw(t) =& Emw(t-\delta t) + exposedmw(t)   - infectionmw(t),\\
    Im(t)  =& Im(t-\delta t)  + infectionm(t)  - detectedm(t),\\                                    
    Imw(t) =& Imw(t-\delta t) + infectionmw(t) - detectedmw(t),\\
    Qm(t)  =& Qm(t-\delta t)  + detectedm(t)   - recoverym(t)   - deathsm(t),\\                           
    Qmw(t) =& Qmw(t-\delta t) + detectedmw(t)  - recoverymw(t)  - deathsmw(t),\\
    ImV(t) =& ImV(t-\delta t) + infectionVm(t) - detectedVm(t),\\                                    
    QmV(t) =& QmV(t-\delta t) + detectedVm(t)  - recoveryVm(t),\\                                   
    Rm(t)  =& Rm(t-\delta t)  + recoverym(t)   + recoveryVm(t)  - exposedwm(t),\\                        
    D(t)   =& D(t-\delta t)   + deathsw(t)     + deathsm(t)     + deathswm(t)  + deathsmw(t), \\      
    Rb(t)  =& Rb(t-\delta t)  + recoverywm(t)  + recoverymw(t),\\
    P(t)   =& P(t-\delta t)   + protected(t)   - released(t),\\                                        
    V1(t)  =& V1(t-\delta t)  + vaccinated1(t) - vaccinated2(t) - exposedVw(t) - exposedVm(t),  \\         
    V2(t)  =& V2(t-\delta t)  + vaccinated2(t),  \\
\end{split}
\label{eq:system_equations}
\end{equation}
where each term in the previous system arise from a binomial distribution as follows:

\clearpage
\begin{equation}
    \begin{split}
        exposedw(t) \sim & Bin(S(t-\delta t),P(x_i=Ew)),\\
        exposedm(t) \sim & Bin(S(t-\delta t),P(x_i=Em)),\\
        exposedmw(t) \sim & Bin(Rw(t-\delta t),P(x_i=Em)),\\
        exposedwm(t) \sim & Bin(Rm(t-\delta t),P(x_i=Ew)),\\
        protected(t) \sim & Bin(S(t-\delta t),P(x_i=P)),\\
        released(t) \sim & Bin(P(t-\delta t),P(x_i=S)),\\
        infectionw(t) \sim & Bin(Ew(t-\delta t),P(x_i=Iw)), \\
        infectionm(t) \sim & Bin(Em(t-\delta t),P(x_i=Im)), \\
        infectionwm(t) \sim & Bin(Ewm(t-\delta t),P(x_i=Iwm)), \\
        infectionmw(t) \sim & Bin(Emw(t-\delta t),P(x_i=Imw)),\\
        detectedw(t) \sim & Bin(Iw(t-\delta t),P(x_i=Qw)),\\
        detectedm(t)  &\sim Bin(Im(t-\delta t),P(x_i=Qm)),\\
        detectedwm(t) \sim & Bin(Iwm(t-\delta t),P(x_i=Qwm)),\\
        detectedmw(t) \sim & Bin(Imw(t-\delta t),P(x_i=Qmw)),\\
        recoveryw(t) \sim & Bin(Qw(t-\delta t),P(x_i=Rw)),\\
        recoverym(t) \sim & Bin(Qm(t-\delta t),P(x_i=Rm)),\\
        recoverywm(t) \sim & Bin(Qwm(t-\delta t),P(x_i=Rb)),\\
        recoverymw(t) \sim & Bin(Qmw(t-\delta t),P(x_i=Rb)),\\
        deathsw(t) \sim & Bin(Qw(t-\delta t),P(x_i=D)),\\
        deathsm(t) \sim & Bin(Qm(t-\delta t),P(x_i=D)),\\
        deathswm(t) \sim & Bin(Qwm(t-\delta t),P(x_i=D)),\\
        deathsmw(t) \sim & Bin(Qmw(t-\delta t),P(x_i=D)),\\
        vaccinated1(t) \sim & Bin(S(t-\delta t),P(x_i=V1)),\\
        vaccinated2(t) \sim & Bin(V1(t-\delta t),P(x_i=V2)),\\
        exposedVw(t) \sim & Bin(V1(t-\delta t),P(x_i=EwV)),\\
        exposedVm(t) \sim & Bin(V1(t-\delta t),P(x_i=EmV)),\\
        infectionVw(t) \sim & Bin(EwV(t-\delta t),P(x_i=IwV)),\\
        infectionVm(t) \sim & Bin(EmV(t-\delta t),P(x_i=ImV)),\\
        detectedVw(t) \sim & Bin(IwV(t-\delta t),P(x_i=QwV)),\\
        detectedVm(t) \sim & Bin(ImV(t-\delta t),P(x_i=PmV)),\\
        recoveryVw(t) \sim & Bin(QwV(t-\delta t),P(x_i=Rw)),\\
        recoveryVm(t) \sim & Bin(QmV(t-\delta t),P(x_i=Rm)),\\
    \end{split}
    \label{eq:Terms}
\end{equation}
and the probabilities of transition are defined as:

\begin{equation}
    \begin{split}
        P(x_i=Ew)   =& (1.0 - exp(-\beta_w(t) \ dt)),\\
        P(x_i=Em)   =& (1.0 - exp(-\beta_m(t) \ dt)),\\
        P(x_i=EwV)  =& (1.0 - exp(-\nu_V \ \beta_w(t) \ dt)), \\
        P(x_i=EmV)  =& (1.0 - exp(-\nu_V \ \beta_m(t) \ dt)), \\
        P(x_i=Iw,Im,Iwm,Imw,IwV,ImV)    =& (1.0 - m.exp(-\gamma \ dt)),\\
        P(x_i=Rw,Rm,Rb)    =& (1.0 - m.exp(-\lambda(t) \ dt)), \\
        P(x_i=P)    =& (1.0 - exp(-\alpha(t) \ dt)), \\
        P(x_i=D)    =& (1.0 - exp(-k(t) \ dt)), \\
        P(x_i=S)  =& (1.0 - exp(-\tau(t) \ dt)) , \\
        P(x_i=Qm,Qw,Qmw,Qwm,QwV,QmV)  =& (1.0 - exp(-\delta \ dt)),\\
        P(x_i=V1) =& (1.0 - exp(-\rho_1*\delta_{V1} \ dt)), \\
        P(x_i=V2) =& (1.0 - exp(-\rho_2*\delta_{V2} \ dt)), \\
    \end{split}
    \label{eq:Prob}
\end{equation}

where $P(x_i=Ew)$ is the probability to be exposed  to wild strain, $P(x_i=Em)$ is probability to be exposed  to mutant strain, $P(x_i=EwV)$ is the probability to be exposed  to wild strain of vaccinated population, $P(x_i=EmV)$ is the probability of vaccinated population be exposed to mutant strain, $P(x_i=Iw,Im,Iwm,Imw,IwV,ImV)$ is the infection probability, $P(x_i=Rw,Rm,Rb)$ is the recovery probability, $P(x_i=P)$  is the protection probability, $P(x_i=D)$ is the death by infection probability probability, $P(x_i=S)$ is the release probabilityof protected population,$P(x_i=Qm,Qw,Qmw,Qwm,QwV,QmV)$  is detecttion of infected population probability,$P(x_i=V1)$  is the vaccination first dose probability, and $P(x_i=V2)$  is the vaccination second dose probability. On the other hand, we define the infections rates corresponding to the wild strain ($\beta_w(t)$) and the mutant ($\beta_m(t)$) as follow: 

The infection rate is calculated having into account both infected populations, reported and not reported infected. Not reported infected populations are those who still are note detected by the health system while reported infected are those who already were detected by the health system. 

\begin{equation}
    \begin{split}
    \beta_w(t)    =& (1/N) (\beta_I(t) \ IW + \beta_Q \ QW), \\                    
    \beta_m(t)    =& \omega \ (1/N)*( \beta_I(t) \ IM + \beta_Q \ QM), \\
    \end{split}
    \label{eq:betas}
\end{equation}
where, $IW=Iw+IwV+Iwm$, $QW=Qw+QwV+Qwm$, $IM=Im+ImV+Imw$ and $QM=Qm+QmV+Qmw$.

The effect of temperature on the infection rate is modeled as follows:
\begin{equation}
    \begin{split}
    T(t) =& 1-\frac{\hat{T(t)}}{(\overline{\hat{T}})}+c,\\
    \hat{T(t)} =& f(T),\\
    \end{split}
    \label{eq:T_inv}
\end{equation}
where $f(.)$ is a Savgol filter function with a window length of $51$ and a polynomial order of $3$ used to smooth the temperature profile in time $t$ and $c=1.5$ is a scalar use to normalize the function. The length of the Savgol filter window must be a positive odd integer. This value must be less than or equal to the size of of the signal length (Temperature).

The parameters used to calculate each probability are described next:
\begin{itemize}
    \item $\delta$ : detection rate
    \item $\nu_V$ : partial protection provided by the first dose of the vaccine
    \item $\alpha(t)  = \alpha_0 \ exp(-\alpha_1 \ t)$ : lockdown rate
    \item $\beta_Q(t) = \beta_Q \ ((1+sin(2 \ \pi \ t \ 1/365))) \ \frac{1}{2}$ : infection rate by detected infectious individuals when no temperature influence is modeled
    \item $\beta_I(t) = \beta_I \ ((1+sin(2 \ \pi \ t \ 1/365))) \ \frac{1}{2}$ : infection rate by non-detected infectious individuals when no temperature influence is modeled
    \item $\beta_Q(t) = \beta_Q \ T(t)$ : infection rate by detected infectious individuals when temperature influence is modeled
    \item $\beta_I(t) = \beta_I \ T(t)$ : infection rate by non-detected infectious individuals when temperature influence is modeled
    \item $\gamma$ : incubation rate
    \item $\lambda(t) = \lambda_0 \ (1 - exp(\lambda \ t))$: recovery rate
    \item $k (t) = k_0 * exp(- k_1 \ t)$: death by virus rate   
    \item $\tau (t) = \tau_0(1 - exp(-\tau_1 \ t))$: releasing rate
    \item $\rho1$ : vaccination rate for the first dose
    \item $\rho2$ : vaccination rate for the second dose
    \item $\delta_{V1}$ : time lag for the vaccine first dose
    \item $\delta_{V2}$ : time lag for the vaccine second dose
    \item $\omega$ : severity of the mutant strain
\end{itemize}

The probability that a mutant strain will emerge that can infect the population is modeled as shown by the equations below. 
\begin{equation}
    \begin{split}
     P_m(t)=&1/(1+(365*100) \ exp(-(t))),\\
     I_m(t)\sim & Bin(Tot_{I}), P_m(t)),\\
    \end{split}
    \label{eq:probM}
\end{equation}
where, $P_m(t)$ is the probability of a mutant strain to emerge in the population given a the time since the wild strain is circulating within the population, $I_m(t)$ is the number of new mutant infected individuals and $Tot_{I}$ is the mean infected individuals per million  with the wild strain in the last two weeks. 
% \subsubsection{Headings: third level}
