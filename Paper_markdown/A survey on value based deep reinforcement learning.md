# A survey on value-based deep reinforcement learning

# Abstract

Reinforcement learning(RL) address the problem of how to make a sequential decision in the real world. It aims to maximize the total reward in the task. In recent years, deep learning is one of the attractive fields, and it has pushed the limits of what was possible in the domain of Machine Learning and Artificial Intelligence. Combine with RL and deep learning is a trendy field in the regime of artificial intelligence(AI). It named as deep RL. It represents a step towards developing autonomous systems with a higher-level understanding of the observed world. There are various subfields for deep RL, but the arise of deep RL should back to the DQN algorithm, which is proposed by \cite{1} . In this survey, we begin with an introduction to the general field or deep RL, then focus on value-based deep RL, which is one of the bases in deep RL. Our survey will cover the central algorithm in the value-based deep RL field. We will review the latest research results and also recall the issue in value-based method, especially, the problems in using function approximation for RL. We hope to inspire novel research methods of the value-based algorithm.  

# Introduction

First, let us consider the question of how to build an efficient artificial intelligence(AI) system. The AI system must face the complicated structure of the real world and also must deal with the sequential decision-making, which is the task of deciding from the sequence of actions performed in an uncertain environment to maximize the total reward. In recent years, deep reinforcement learning considered as a step towards developing an efficient AI system. Deep reinforcement learning uses deep learning to deal with the unstructured of the data. In the past decades, researchers have designed massive manual designed features for the specific application. But the rise of the deep learning rises a revolution to let the network do the feature extract task. Reinforcement learning(RL) proposed from the inspired of the behavioural psychology \cite{sutton98} and able to deal with the sequential decision-making problem。The newborn deep RL fills us with visions of the future.
Several of these achievements are due to the combination of RL with deep learning techniques \cite{LeCun et al., 2015; Schmidhuber,2015; Goodfellow et al., 2016} . For instance,  \cite{Mnih 2015} have used the deep RL agent successfully to learn raw pixels to achieve human-level in atari game.  \cite{Alphago} can defeat human champions in the go game.  \cite{ 2} use a general deep RL algorithm to mater chess and Shogi by self-play.  \cite{starcraft} have applied deep RL method in the complicated strategy games and achieve a fantastic result. Deep RL also has potential for real-world applications such as robotics  \cite{Levine et al., 2016; Gandhi et al., 2017; Pinto et al., 2017}, self-driving cars  \cite{You et al., 2017}, finance  \cite{Deng et al., 2017} and smart grids  \cite{François-Lavet, 2017}. 

**Deep RL Alogrithm Taxonomy:** 
It is not easy to draw an accurate taxonomy now for the fast-developing field. But generally, we can have an intuition for the current deep RL taxonomy, which ignore the most advanced area of deep RL (meta-learning, transfer learning, MAML exploration, etc.).
One of the essential branching points in a deep RL algorithm is according to the question of whether the agent has access to a model of the environment. If the state transitions and rewards can be predicted, then it will be the model-based deep RL alogrithms, else will be the model-free deep RL alogrithms.

* Model-free deep RL alogrithms: including policy optimization(eg. Policy Gradient, A2C, A3C, PPO, TRPO), Q-Learning(eg. DQN, CS1, QR-DQN, HER), and hybird alogrithms(DDPG, TD3, SAC)
* Model-based deep RL alogrithms: Model is given(eg. AlphaZero) and Model is unknown(eg. I2A, MBMF, MBVE)

This article survey the literature along these three axes: DQN alogrithms, combination DQN with policy optimization alogrithms and combination DQN with other methods. 

# Problem formulation
Reinforcement Learning \cite{sutton98} is concerned with learning control policies for agents interacting with unknown environments $\mathcal{E}$. Such environments are often formalized as a Markov Decision Processes (MDPs), described by a 4-tuple $(\mathcal{S},\mathcal{A},\mathcal{P},\mathcal{R})$. At each timestep $t$ an agent interacting with the MDP observes a state $s_t \in \mathcal{S}$, and chooses an action $a_t \in \mathcal{A}$ which determines the reward $r_t \sim \mathcal{R}(s_t,a_t)$ and next state $s_{t+1} \sim \mathcal{P}(s_t,a_t)$.

The goal of the agent is to interact with the environments $\mathcal{E}$ by selecting actions in a way that maximises future rewards. We make the standard assumption that future rewards are discounted by a factor of $\gamma$ per time-step, and define the future discounted return at time $t$ as $R_t = \sum_{t'=t}^{T} \gamma^{t'-t} r_{t'}$, where $T$ is the time-step. We define the optimal action-value function $Q^*(s,a)$ as the maximum expected return achievable by following any strategy, after seeing some sequence $s$ and then taking some action $a$, $Q^*(s,a) = \max_{\pi} \mathbb{E}[{ R_t | s_t=s, a_t=a, \pi }]$, where $\pi$ is a policy mapping sequences to actions (or distributions over actions).

The optimal action-value function obeys an important identity known as the Bellman equation. This is based on the following intuition: if the optimal value $Q^*(s',a')$ of the sequence $s'$ at the next time-step was known for all possible actions $a'$, then the optimal strategy is to select the action $a'$ maximising the expected value of $r + \gamma Q^*(s',a')$,
$$\begin{align}
Q^*(s,a) &= \mathbb{E}_{s' \sim \mathcal{E}}[{ r + \gamma \max_{a'} Q^*(s', a') \Big| s, a }]
\end{align}$$

The basic idea behind many reinforcement learning algorithms is to estimate the action-value function, by using the Bellman equation as an iterative update, $Q_{i+1}(s,a) = \mathbb{E}\left[ r + \gamma \max_{a'} Q_i(s', a') | s, a \right]$. Such value iteration algorithms converge to the optimal action-value function, $Q_i \rightarrow Q^*$ as $i \rightarrow \infty$ \cite{sutton:book}. In practice, this basic approach is totally impractical, because the action-value function is estimated separately for each sequence, without any generalisation. Instead, it is common to use a function approximator to estimate the action-value function, $Q(s,a; \theta) \approx Q^*(s,a)$. In the reinforcement learning community this is typically a linear function approximator, but sometimes a non-linear function approximator is used instead, such as a neural network. We refer to a neural network function approximator with weights $\theta$ as a Q-network. A Q-network can be trained by minimising a sequence of loss functions $L_i(\theta_i)$ that changes at each iteration $i$,

$$\begin{align}
L_i\left(\theta_i\right) &= \mathbb{E}_{s,a \sim \rho(\cdot)}{\left(y_i - Q \left(s,a ; \theta_i \right) \right)^2}
\end{align}$$

where $y_i = \mathbb{E}_{s' \sim \mathcal{E}}[{r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) | s, a }]$ is the target for iteration $i$ and $\rho(s,a)$ is a probability distribution over sequences $s$ and actions $a$ that we refer to as the behaviour distribution. The parameters from the previous iteration $\theta_{i-1}$ are held fixed when optimising the loss function $L_i\left(\theta_i\right)$. 
Note that the targets depend on the network weights; this is in contrast with the targets used for supervised learning, which are fixed before learning begins. 
Differentiating the loss function with respect to the weights we arrive at the following gradient,
$$\begin{align}
\nabla_{\theta_i} L_i\left(\theta_i\right) &= \mathbb{E}_{s,a \sim \rho(\cdot); s' \sim \mathcal{E}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta_{i-1}) - Q(s,a ; \theta_i ) \right) \nabla_{\theta_i} Q(s,a;\theta_{i}) \right] .
\label{eq:q-learning-gradient}
\end{align}$$
Rather than computing the full expectations in the above gradient, it is often computationally expedient to optimise the loss function by stochastic gradient descent.



# Related work
The researchers have dive into the fields of RL for decades, the traditional RL algorithms have been well developed and also work very well when we combine with deep learning currently. There are many article have surveied the traditional RL, such as [Abhijit Gosavi],[Michael L.Littman 1996],[sutton], [Algorithms for reinforcement learning by szepesvari].
Due to the advent of deep learning has had a significant impact on many areas in machine learning, it also dramatically improving the state-of-the-art in the RL. So, there are also some survey conver the seminal and recent developments in deep RL，such as [1708.05866]. For a more comprehensive survey of recent efforts in DRL, including applications of DRL to areas such as natural language processing \citep{ranzato2016sequence, bahdanau2017actor}, we refer readers to the overview by Li \citep{li2017deep}.

# Value-based RL

The value-based RL algorithms will build a value function, which subsequently define a policy. The basis and most popular value-based algorithms is the Q-learning algorithm \citep{watkins1989learning} and its variant, the fitted Q-learning, that uses parameterized function approximators \citep{gordon1996stable}.

Before we dive into the Q-learning, we should consider the Sarsa algorithm at firt. Sarsa is on-policy algorithm and Q-learning is always considered as  an off-policy algorithm.  On-policy alogrithms

**Sarsa Algorithm**
Sarsa algorithm is to learn an action-value function, $q_\pi (s,a)$, rather than a state-value function. We get the reward from $q_pi (s_t,a_t)$ and lead to new state-action pair (s_{t+1},a_{t+1}), so that we get the chain according to the state, action and reward. We shown the firgue of Sarsa in Figure x.
[Sarsa figure]
The SARSA algorithm is given in algorithm-x:
[Algorithm in P130 of sutton]

**Q-learning Algorithm:** 
The basic version of Q-learning will lookup table of values $Q(s, a)$. The alogrithms aims to learn the optimal Q-value function, and the Q-learning algorithm makes use of the Bellman equation for the Q-value function \citep{bellman1962applied}.
By Banach's theorem, the fixed point of the Bellman operator exists when the two conditions are satisified. The state-action pairs are represented discretely and  all actions are repeatedly sampled in all states.
But this simple alogrithms can not work well when we face the high-dimensional state-action space. We must consider the parameterized value function $Q(s, a; \theta)$, where $\theta$ refers to some parameters that define the Q-values.
The Q-learning algorithm is given in algorithm-x:
[[Algorithm in P131 of sutton]]
**Fitted Q-learning:**
As we could get many experience $(s,a,r,s')$ from the given dataset $D$, where the state at the next time-step $s'$ is generated based on transistion probability $T(s,a,\cdot)$ and the reward $r$ is given by the reward function $R(s,a,s')$. In fitted Q-learning \citep{gordon1996stable}, the algorithm based on random initialization of the Q-values $Q(s, a; \theta_0 )$ where $\theta_0$ with respec to the initial parameter.  Then,  at each iteration  $Q(s, a; \theta_k )$ is updated towards the target value with an approximation of the Q-values.
$$
Y_k^Q = r +\gamma \operatorname*{max}_{a' \in \mathcal A} Q(s',a';\theta_{k})
$$
where $\theta_k$ withs respect to  the parameters in the  $k_{th}$ iteration. The update is performed form the whole set of experience. 
The Fitted Q-learning algorithm is given in algorithm-x:
---
---
---
based on silver lecture 6
** linear combinations of features**
slide 13

**linear value function approximaiton
slide 13
**table lookup features
** monte-carlo with value function approximaiton
**TD learning and TD lambda
** comparei on/off policy  MC TD0 TDLAMBDA

** Decision tree**

**Nearest neighbour**

**Fourier/wavelet bases""
---
---
---
**Neural Fitted Q-learning (NFQ)**

In neural fitted Q-learning (NFQ) \citep{riedmiller2005neural},  we feed the state to the Q-network and output the probability of each action. This structure  can efficiently obtaining the computation of the maximum Q-value in state $s'$ . The Q-values are parameterized with a neural network $Q(s,a;\theta_k)$ where the parameters $\theta_k$ are updated by SGD to minimize the square loss:
$$
\begin{align}
\text{Loss} = \left( Q(s, a; \theta_k )-Y_k^Q \right)^2.
\end{align}
$$

$$
\theta_{k+1}=\theta_{k}+\alpha \left(Y_k^Q - Q (s,a; \theta_k)\right) \nabla_{\theta_k} Q(s ,a ; \theta_k),
$$

where $\alpha$ is the learning rate. Using the square loss to ensures that $Q(s, a; \theta_k )$ should tend without bias to the expected value of the random variable $Y_k^Q$ 

When the weights was updated, the target was also been changed, so this approach can generate errors in  the state-action space. Therefore, the contraction mapping property of the Bellman operator is not enough to guarantee convergence.  In the previous article, it was proved in experiment that these errors will propagate rapidly and the convergence will be  slow and unstable \citep{baird1995residual, tsitsiklis1997analysis, gordon1999approximate, riedmiller2005neural}.  Another disadvantage is the overestimate issue.  When the function approximateors was used to estimate the $max$ operator,the Q-values will tend to be overestimate \citep{van2016deep}.

The Neural Fitted Q-learning (NFQ) algorithm is given in algorithm-x:



# Value-based Deep RL

Here we specifically survey the deep Q-network (DQN) algorithm \citep{mnih2015human} which has achieved superhuman-level control when playing atari games from the pixels by using neural networks as function approximators. We will also survey various improvements of the DQN algorithm.
**Deep Q-networks**

%from 1810.07862
Leveraging ideas from NFQ, the deep Q-network (DQN) algorithm introduced by \citet{mnih2015human} is able to obtain strong performance in an online setting for a variety of ATARI games, directly by learning from the raw pixels of the video. Due to the reason that the Q-learning algorithm can not find the optimal policy in the high dimension, so the DQN algorithm is introduced to overcome this problem. 
As we disscussed in the classic alogrithms, the reward obtained is not stable or divergence when a nonlinear function approximator is used. A little change of the Q-values will greatly affect the policy. Thus, the data distribution and the correlations between the Q-values and the target values $R+\gamma \max_{a'} \mathcal{Q}(s',a')$ are varied. To address this issue, \citep{mnih2015human} have proposed two technique, experience replay and target Q-network.

**Experience replay mechanism:** The algorithm first initializes a replay memory $\mathbf{D}$, i.e., the memory pool, with transitions $(s_t, a_t, r_t, s_{t+1})$, i.e., experiences, generated randomly, e.g., through using $\epsilon$-greedy policy. Then, the algorithm randomly selects samples, i.e., minibatches, of transitions from $\mathbf{D}$ to train the DNN. The Q-values obtained by the trained DNN will be used to obtain new experiences, i.e., transitions, and these experiences will be then stored in the memory pool $\mathbf{D}$. This mechanism allows the DNN trained more efficiently by using both old and new experiences. In addition, by using the experience replay, the transitions are more independent and identically distributed, and thus the correlations between observations can be removed.

**Fixed target $Q$-network:** In the training process, the $\mathcal{Q}$-value will be shifted. Thus, the value estimations can be out of control if a constantly shifting set of values is used to update the $Q$-network. This leads to the destabilization of the algorithm. To address this issue, the target $Q$-network is used to update frequently but slowly the primary $Q$-networks' values. In this way, the correlations between the target and estimated $\mathcal{Q}$-values are significantly reduced, thereby stabilizing the algorithm.	


The DQL algorithm with experience replay and fixed target $Q$-network is presented in Algorithm~\ref{alg:DQL_ER_FTN}. DQL inherits and promotes advantages of both reinforcement and deep learning techniques, and thus it has a wide range of applications in practice such as game development~\cite{alphago}, transportation~\cite{lin2018efficient}, and robotics~\cite{gu2017deep}.


%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
\begin{algorithm}[!]
	\caption{The DQL Algorithm with Experience Replay and Fixed Target $Q$-Network}
	\label{alg:DQL_ER_FTN}
	\begin{algorithmic}[1]
		\STATE Initialize replay memory $\mathbf{D}$.
		\STATE Initialize the $Q$-network $\mathbf{Q}$ with random weights $\boldsymbol{\theta}$.
		\STATE Initialize the target $Q$-network $\hat{\mathbf{Q}}$ with random weights $\boldsymbol{\theta'}$.
		\FOR{\textit{episode=1 to T}}
		\STATE With probability $\epsilon$ select a random action $a_t$, otherwise select $a_t=\arg \max \mathcal{Q}^*(s_t, a_t, \boldsymbol{\theta})$.
		\STATE Perform action $a_t$ and observe immediate reward $r_t$ and next state $s_{t+1}$.
		\STATE Store transition $(s_t, a_t, r_t, s_{t+1})$ in $\mathbf{D}$.
		\STATE Select randomly samples c$(s_j, a_j, r_j, s_{j+1})$ from $\mathbf{D}$.
		\STATE The weights of the neural network then are optimized by using stochastic gradient descent with respect to the network parameter $\boldsymbol{\theta}$ to minimize the loss:
		\begin{equation}
		\label{eq:DQL}
		\Big[ r_j + \gamma\max_{a_{j+1}} \hat{\mathcal{Q}}(s_{j+1},a_{j+1};\boldsymbol{\theta'}) - \mathcal{Q}(s_j,a_j;\boldsymbol{\theta}) \Big]^2 .
		\end{equation}
		\STATE Reset $\hat{\mathbf{Q}} = \mathbf{Q}$ after every a fixed number of steps.
		\ENDFOR
	\end{algorithmic}
\end{algorithm}
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


**dueling **ls.


**double 

**averaged deep q-network


# problem
## overestimation
slide14










# Combine value-based and policy-based
In this chapter, we discuss the work that combine the value-based algorithms and policy-based alogrithms.


# Different architecture

\begin{itemize}
\item The target Q-network in Equation \ref{eq:target_Y} is replaced by $Q(s',a';\theta_k^{-})$ where its parameters $\theta_k^{-}$ are updated only every $C \in \mathbb N$ iterations with the following assignment: $\theta_k^{-}=\theta_k$. This prevents the instabilities to propagate quickly and it reduces the risk of divergence as the target values $Y_k^Q$ are kept fixed for $C$ iterations. The idea of target networks can be seen as an instantiation of fitted Q-learning, where each period between target network updates corresponds to a single fitted Q-iteration.
\item In an online setting, the replay memory \citep{lin1992self} keeps all information for the last $N_{\text{replay}} \in \mathbb N$ time steps, where the experience is collected by following an $\epsilon$-greedy policy\footnote{It takes a random action with probability $\epsilon$ and follows the policy given by $\operatorname*{argmax}_{a \in \mathcal A} Q(s, a; \theta_k )$ with probability $1-\epsilon$.}.
%This allows to make updates from a reasonable stable data distribution kept in the replay memory (for $N_{\text{replay}}$ sufficiently large). In addition, 
The updates are then made on a set of tuples $<s,a,r,s'>$ (called mini-batch) selected randomly within the replay memory. This technique allows for updates that cover a wide range of the state-action space. In addition, one mini-batch update has less variance compared to a single tuple update. Consequently, it provides the possibility to make a larger update of the parameters, while having an efficient parallelization of the algorithm.
\end{itemize}
A sketch of the algorithm is given in Figure \ref{ONFQ_schema}. 

In addition to the target Q-network and the replay memory, DQN uses other important heuristics.
To keep the target values in a reasonable scale and to ensure proper learning in practice, rewards are clipped between -1 and +1.
Clipping the rewards limits the scale of the error derivatives and makes it easier to use the same learning rate across multiple games (however, it introduces a bias).
In games where the player has multiple lives, one trick is also to associate a terminal state to the loss of a life such that the agent avoids these terminal states (in a terminal state the discount factor is set to 0).

In DQN, many deep learning specific techniques are also used.
In particular, a preprocessing step of the inputs is used to reduce the input dimensionality, to normalize inputs (it scales pixels value into [-1,1]) and to deal with some specificities of the task.
In addition, convolutional layers are used for the first layers of the neural network function approximator and the optimization is performed using a variant of stochastic gradient descent called RMSprop \citep{rmsprop}.



\begin{figure}[ht!]
 \centering
\scalebox{0.8}{
\begin{tikzpicture}
 [	NN/.style={circle,draw=blue!50,fill=blue!20,thick, inner sep=5pt,minimum size=6mm},
     inputs/.style={rectangle,draw=black!50,fill=black!20,thick, inner sep=5pt,minimum size=4mm},
     target/.style={rectangle,draw=black!50,fill=green!20,thick, inner sep=5pt,minimum size=4mm},
   Vhat/.style={circle,draw=black!50,fill=black!20,thick, inner sep=5pt,minimum size=4mm},
   ddt/.style={rectangle,draw=black!50,thick, inner sep=5pt,minimum size=4mm}];

  \node at (0,0) 	[NN, align=center] (NN) {Update \\$Q(s,a;\theta_k)$};
  \node [NN, right=of NN, align=center] (NN_t) {Every C:\\$\theta_k^{-}:=\theta_k$};
  \node [left=of NN,yshift=-7mm, xshift=-11mm,] 	[inputs] (rt) {$r_{1}, \ldots, r_{N_{replay}}$};
  \node [above=of rt, yshift=-7mm] 	[inputs] (st) {$s_{1}, \ldots, s_{N_{replay}}, a_{1}, \ldots, a_{N_{replay}}$};
  \node [below=of rt,yshift=7mm] 	[inputs] (st1) {$s_{1+1}, \ldots, s_{N_{replay+1}}$};
  \node [below=of NN,yshift=-2mm] 	[target] (target) {$r_t + \gamma \underset{a' \in A}{\operatorname{max}} ( Q(s_{t+1},a'; \theta_k^{-}) )$};
  \node[inner sep=0,minimum size=0,above=of NN, yshift=-3mm] (k) {}; % invisible node
  \node[draw,left=of rt, xshift=-8mm, align=center] (k3) {Policy}; % invisible node
  \node[draw,below=of k3, align=center] (env) {Environment}; % invisible node
  \node[inner sep=0,minimum size=0] (k1) at (k -| k3) {}; % invisible node
  \node[inner sep=0,minimum size=0] (k2) at (target -| k3) {}; % invisible node

 \draw [->] (st.east) -- (NN.west) node[midway, above] {};
 \draw [->] (st1.east) -- (target);
 \draw [->] (rt.east) -- (target);
 \draw [->] (target) -- (NN) node[midway, right, yshift=1mm, xshift=-1mm] {};
 \draw [-] (NN) -- (k);
 \draw [-] (k) -- (k1);
  \draw [-] (k1) -- (k3);
  \draw [->] (k3) -- (st.west);  \draw [->] (k3) -- (rt.west);  \draw [->] (k3) -- (st1.west);
  \draw [->] (NN) -- (NN_t);
  \draw [->] (NN_t) -- (target);
  \draw [<->] (k3) -- (env);

\end{tikzpicture}
}
 \caption[Sketch of the DQN algorithm.]{Sketch of the DQN algorithm. $Q(s,a;\theta_k)$ is initialized to random values (close to 0) everywhere in its domain and the replay memory is initially empty; the target Q-network parameters $\theta_k^-$ are only updated every C iterations with the Q-network parameters $\theta_k$ and are held fixed between updates; the update uses a mini-batch (e.g., 32 elements) of tuples $<s, a>$ taken randomly in the replay memory along with the corresponding mini-batch of target values for the tuples.}
 \label{ONFQ_schema}
\end{figure}


%Many additional improvements can be found in the literature and a few of them are cited hereafter.
\section{Double DQN}
\label{sec:double}
The max operation in Q-learning (Equations \ref{op_mapping_Bellman_Q}, \ref{eq:target_Y}) uses the same values both to select and to evaluate an action.
This makes it more likely to select overestimated values in case of inaccuracies or noise, resulting in overoptimistic value estimates.
Therefore, the DQN algorithm induces an upward bias.
The double estimator method uses two estimates for each variable, which allows for the selection of an estimator and its value to be uncoupled \citep{hasselt2010double}.
Thus, regardless of whether errors in the estimated Q-values are due to stochasticity in the environment, function approximation, non-stationarity, or any other source, this allows for the removal of the positive bias in estimating the action values.
%This is important, because any method incurs some inaccuracies and this results in instabilities and lower performance in practice.
In Double DQN, or DDQN \citep{van2016deep}, the target value $Y_k^Q$ is replaced by
\begin{equation}
Y_k^{DDQN} = r +\gamma Q(s',\operatorname*{argmax}_{a \in \mathcal A} Q(s',a;\theta_k);\theta_k^{-}),
\label{eq:target_YDDQN}
\end{equation}
which leads to less overestimation of the Q-learning values, as well as improved stability, hence improved performance.
As compared to DQN, the target network with weights $\theta_t^{-}$ are used for the evaluation of the current greedy action. Note that the policy is still chosen according to the values obtained by the current weights $\theta$.

\section{Dueling network architecture}
\label{sec:dueling}
In \citep{wang2015dueling}, the neural network architecture decouples the value and advantage function $A^{\pi}(s,a)$ (Equation \ref{def_A}), which leads to improved performance.
The Q-value function is given by
\begin{equation}
\begin{split}
Q(s,a;\theta^{(1)} & , \theta^{(2)},\theta^{(3)}) = V\left(s;\theta^{(1)},\theta^{(3)}\right) \\
& + \left( A\left(s,a;\theta^{(1)},\theta^{(2)}\right) - \operatorname*{max}_{a' \in \mathcal A} A\left(s,a';\theta^{(1)},\theta^{(2)}\right) \right).
\end{split}
\label{eq:QDueling}
\end{equation}

Now, for $a^* = \operatorname*{argmax}_{a' \in \mathcal A} Q(s, a'; \theta^{(1)},\theta^{(2)},\theta^{(3)})$, we obtain $Q(s, a^*;\theta^{(1)},\theta^{(2)},\theta^{(3)}) = V (s; \theta^{(1)},\theta^{(3)})$. As illustrated in Figure \ref{fig:dueling}, the stream $V (s; \theta^{(1)},\theta^{(3)})$ provides an estimate of the value function, while the other stream produces an estimate of the advantage function. The learning update is done as in DQN and it is only the structure of the neural network that is modified.



\begin{figure}[ht!]
 \centering
\scalebox{0.8}{
\begin{tikzpicture}

  \pic [fill=blue!50!green, draw=green!25!black] at (-4-0.1,0.5-0.1) {annotated cuboid={width=6, height=30, depth=6}};

  \pic [fill=blue!50, draw=green!25!black] at (-2-0.1,1.5-0.1) {annotated cuboid={width=6, height=24, depth=6}};
  \pic [fill=green!75!black, draw=green!25!black] at (-2-0.1,-1.5-0.1) {annotated cuboid={width=6, height=24, depth=6}};

  \pic [fill=blue!50, draw=green!25!black] at (0-0.1,1-0.1) {annotated cuboid={width=6, height=6, depth=6}};
  \pic [fill=green!75!black, draw=green!25!black] at (0-0.1,-1.6-0.1) {annotated cuboid={width=6, height=20, depth=6}};

  \pic [fill=black!25, draw=green!25!black] at (2-0.1,0-0.1) {annotated cuboid={width=6, height=20, depth=6}};


\node[font=\Large] (dots) at (-6,-1) {\ldots};

 \draw [dashed] node[] {} (-4-1.5,0.6) -- node[] {} (-4-0.6,0.5);
 \draw [dashed] node[] {} (-4-1.5,-2.6) -- node[] {} (-4-0.6,-2.5);

  \draw [dashed] node[] {} (-4,0.5) -- node[] {} (-2-0.6,1.5);
 \draw [dashed] node[] {} (-4,-2.5) -- node[] {} (-2-0.6,-0.9);
 \draw [dashed] node[] {} (-4,0.5) -- node[] {} (-2-0.6,-1.5);
 \draw [dashed] node[] {} (-4,-2.5) -- node[] {} (-2-0.6,-3.9);

\draw [dashed] node[] {} (-2,1.5) -- node[] {} (0-0.6,1);
 \draw [dashed] node[] {} (-2,-0.9) -- node[] {} (0-0.6,1-0.6);
 \draw [dashed] node[] {} (-2,-1.5) -- node[] {} (0-0.6,-1.6);
 \draw [dashed] node[] {} (-2,-3.9) -- node[] {} (0-0.6,-1.6-2);

 \draw [dashed] node[] {} (0,1) -- node[] {} (2-0.6,0);
 \draw [dashed] node[] {} (0,1-0.6) -- node[] {} (2-0.6,0-2);
 \draw [dashed] node[] {} (0,-1.6) -- node[] {} (2-0.6,0);
 \draw [dashed] node[] {} (0,-1.6-2) -- node[] {} (2-0.6,0-2);

\node[] (Q) at (2,-2.5) {$Q(s,a)$};
\node[color=green!50!black] (A) at (0,-4.1) {$A(s,a)$};
\node[color=blue] (V) at (0,1.5) {$V(s)$};

\draw [thick, blue,decorate,decoration={brace,amplitude=10pt},xshift=0.4pt,yshift=-0.4pt](-3,2.) -- (1,2.) node[blue,midway,yshift=0.7cm] {$\theta^{(3)}$};
\draw [thick, green!50!black,decorate,decoration={brace,amplitude=10pt,mirror},xshift=0.4pt,yshift=-0.4pt](-3,-4.5) -- (1,-4.5) node[green!50!black,midway,yshift=-0.7cm] {$\theta^{(2)}$};
\draw [thick, blue!50!green,decorate,decoration={brace,amplitude=10pt,mirror},xshift=0.4pt,yshift=-0.4pt](-6,-4) -- (-3.5,-4) node[blue!50!green,midway,yshift=-0.7cm] {$\theta^{(1)}$};

\end{tikzpicture}
}
 \caption{Illustration of the dueling network architecture with the two streams that separately estimate the value $V(s)$ and the advantages $A(s,a)$. The boxes represent layers of a neural network and the grey output implements equation \ref{eq:QDueling} to combine $V(s)$ and $A(s,a)$.}
 \label{fig:dueling}
\end{figure}

In fact, even though it loses the original semantics of $V$ and $A$, a slightly different approach is preferred in practice because it increases the stability of the optimization:
\begin{equation}
\begin{split}
Q(s,a &; \theta^{(1)} ,\theta^{(2)} ,\theta^{(3)}) = V\left(s;\theta^{(1)},\theta^{(3)}\right)\\
& + \left( A\left(s,a;\theta^{(1)},\theta^{(2)}\right) - \frac{1}{\lvert \mathcal A \rvert} \sum_{a' \in \mathcal A} A \left(s,a';\theta^{(1)},\theta^{(2)}\right) \right).
\end{split}
\label{eq:QDueling2}
\end{equation}
In that case, the advantages only need to change as fast as the mean, which appears to work better in practice \citep{wang2015dueling}.


\section{Distributional DQN}
\label{sec:distrib}
The approaches described so far in this chapter all directly approximate the expected return in a value function.
Another approach is to aim for a richer representation through a value distribution, i.e. the distribution of possible cumulative returns \citep{jaquette1973markov, morimura2010nonparametric}.
This value distribution provides more complete information of the intrinsic randomness of the rewards and transitions of the agent within its environment (note that it is not a measure of the agent's uncertainty about the environment).


The value distribution $Z^\pi$ is a mapping from state-action pairs to distributions of returns when following policy $\pi$. It has an expectation equal to $Q^\pi$:
$$Q^\pi(s,a) = \mathbb E Z^\pi(s,a).$$
This random return is also described by a recursive equation, but one of a distributional nature:
\begin{equation}
Z^\pi(s,a)=R(s,a,S')+\gamma Z^\pi(S',A'),
\label{eq:distribQit}
\end{equation}
where we use capital letters to emphasize the random nature of the next state-action pair $(S',A')$ and $A' \sim \pi(\cdot|S')$.
The distributional Bellman equation states that the distribution of $Z$ is characterized by the interaction of three random variables: the reward $R(s,a,S')$, the next state-action $(S', A')$, and its random return $Z^\pi(S',A')$.


It has been shown that such a distributional Bellman equation can be used in practice, with deep learning as the function approximator \citep{bellemare2017distributional, dabney2017distributional,rowland2018analysis}.
This approach has the following advantages:
\begin{itemize}
\item It is possible to implement risk-aware behavior (see e.g., \cite{morimura2010nonparametric}).
\item It leads to more performant learning in practice. This may appear surprising since both DQN and the distributional DQN aim to maximize the expected return (as illustrated in Figure \ref{fig:distrib}). One of the main elements is that the distributional perspective naturally provides a richer set of training signals than a scalar value function $Q(s,a)$. These training signals that are not a priori necessary for optimizing the expected return are known as \textit{auxiliary tasks} \citep{jaderberg2016reinforcement} and lead to an improved learning (this is discussed in \S\ref{sec:aux_tasks}).
\end{itemize}

\begin{figure}[ht!]
\centering
\subfloat[Example MDP.]{ %and transition probabilities for two policies $\pi_1$ and $pi_2$
\begin{tikzpicture}[->, >=stealth', auto, semithick, node distance=3cm, baseline]
\tikzstyle{state}=[fill=white, draw=black, thick, text=black, scale=1]
\node[state]    (A)      at (0,1.5)               {$(s,a)$};
\node    (A1)      [color=green!75!black]  at (0,2)            {$\pi_1$};
\node    (A2)      [color=cyan]   at (0,1)            {$\pi_2$};
\node[state]    (B)[above right of=A, yshift=-1cm]   {$s^{(1)}$};
\node[state]    (C)[below right of=A, yshift=1cm]   {$s^{(2)}$};
\node[state]    (D) at (2.5,1.5)   {$s^{(3)}$};
\node[]    (dummy)      at (4.5,-0.7)               {};
\path
(A) edge[bend left, color=green!75!black]  node[above, yshift=0.2cm] {$\mathbb P=1$}      (B)
    edge[color=cyan]  node[above, yshift=0.2cm] {$\mathbb P=0.2$}     (D)
    edge[bend right, color=cyan] node[below, yshift=-0.2cm] {$\mathbb P=0.8$}      (C)
(B) edge[loop right]  node {$\frac{R_{\text{max}}}{5}$}   (B)
(C) edge[loop right]  node {$0$}   (C)
(D) edge[loop right]  node {$R_{\text{max}}$}   (D)
;
\end{tikzpicture}
}
\subfloat[Sketch (in an idealized version) of the estimate of resulting value distribution $\hat Z^{\pi_1}$ and $\hat Z^{\pi_2}$ as well as the estimate of the Q-values $\hat Q^{\pi_1}, \hat Q^{\pi_2}$.]{
\begin{tikzpicture}
[	distrib/.style={rectangle,draw=black!50,fill=black!10,thick, inner sep=5pt,minimum size=4mm, align=center},
  task/.style={circle,draw=black!50,fill=black!5,thick, inner sep=5pt,minimum size=25mm, align=center},
  RLalgo/.style={rectangle,draw=black!50,fill=black!10,thick, inner sep=10pt,minimum width=30mm, align=center},
    declare function={fct1(\x,\k,\theta) = 0.8*1/((2*3.14)^0.5*0.1)*exp(-(1/2)*((\x-0)/(0.1))^2);},
    declare function={fct2(\x,\k,\theta) = 0.2*1/((2*3.14)^0.5*0.1)*exp(-(1/2)*((\x-5)/(0.1))^2);},
    declare function={fct3(\x,\k,\theta) = 1*1/((2*3.14)^0.5*0.1)*exp(-(1/2)*((\x-1)/(0.1))^2);},
    baseline
];
\node[color=cyan]  (C) at (-0.3,3.5)   {$\hat Z^{\pi_1}$};
\node[color=green!75!black]  (C) at (-0.3,3.)   {$\hat Z^{\pi_2}$};

\begin{axis}[
  no markers, domain=-0.3:6, samples=100,
  axis lines=left, xlabel=, ylabel=,
  every axis y label/.style={at=(current axis.above origin),anchor=east},
  every axis x label/.style={at=(current axis.right of origin),anchor=north},
  height=5cm, width=6cm,
  xtick={0,5}, ytick=\empty,
  xticklabels={0,$\frac{R_{\text{max}}}{1-\gamma}$},
  xmax=6.5,
  ymax=5.
  ]
\addplot [very thick,cyan!20!black] {fct1(x,2,2)};
\addplot [fill=cyan!20, draw=none] {fct1(x,2,2)} \closedcycle;
\addplot [very thick,cyan!20!black] {fct2(x,2,2)};
\addplot [fill=cyan!20, draw=none] {fct2(x,2,2)} \closedcycle;
\addplot [very thick,green!20!black] {fct3(x,2,2)};
\addplot [fill=green!20, draw=none] {fct3(x,2,2)} \closedcycle;
\end{axis}
\draw[color=green!50!black, thick] (0.83,0) -- (0.83,3.5);
\draw[color=cyan, thick] (0.86,0) -- (0.86,3.5);

\node[color=green!50!black]  (C) at (0.55,3.8)   {$\hat Q^{\pi_1}$};
\node[] at (0.95,3.7)   {$\approx$};
\node[color=cyan]  (C) at (1.45,3.8)   {$\hat Q^{\pi_2}$};

\end{tikzpicture}
}

 \caption{For two policies illustrated on Fig (a), the illustration on Fig (b) gives the value distribution $Z^{(\pi)}(s,a)$ as compared to the expected value $Q^\pi(s,a)$. On the left figure, one can see that $\pi_1$ moves with certainty to an absorbing state with reward at every step $\frac{R_{\text{max}}}{5}$, while $\pi_2$ moves with probability $0.2$ and $0.8$ to absorbing states with respectively rewards at every step $R_{\text{max}}$ and $0$. From the pair $(s,a)$, the policies $\pi_1$ and $\pi_2$ have the same expected return but different value distributions.}
 \label{fig:distrib}
\end{figure}


%In \citep{pritzel2017neural}, a differentiable memory module allows to rapidly integrate recent experience by interpolating between Monte Carlo value estimates and backed up off-policy estimates.

\section{Multi-step learning}
\label{sec:multi_step}
In DQN, the target value used to update the Q-network parameters (given in Equation \ref{eq:target_Y}) is estimated as the sum of the immediate reward and a contribution of the following steps in the return. That contribution is estimated based on its own value estimate at the next time-step. For that reason, the learning algorithm is said to \textit{bootstrap} as it recursively uses its own value estimates \citep{sutton1988learning}.

This method of estimating a target value is not the only possibility.
Non-bootstrapping methods learn directly from returns (Monte Carlo) and an intermediate solution is to use a multi-step target \citep{sutton1988learning,watkins1989learning,peng1994incremental,singh1996reinforcement}.
Such a variant in the case of DQN can be obtained by using the n-step target value given by:
\begin{equation}
Y_k^{Q,n} = \sum_{t=0}^{n-1} \gamma^t r_{t} +\gamma^{n} \operatorname*{max}_{a' \in A} Q(s_{n},a';\theta_k)
\label{eq:target_Y_multi}
\end{equation}
where $(s_0, a_0, r_0,\cdots, s_{n-1}, a_{n-1}, r_{n-1},s_{n})$ is any trajectory of $n+1$ time steps with $s=s_0$ and $a=a_0$.
A combination of different multi-steps targets can also be used:
\begin{equation}
Y_k^{Q,n} = \sum_{i=0}^{n-1} \lambda_i \left( \sum_{t=0}^{i} \gamma^t r_{t} +\gamma^{i+1} \operatorname*{max}_{a' \in A} Q\left(s_{i+1},a';\theta_k\right) \right)
\label{eq:target_Y_combi_multi}
\end{equation}
with $\sum_{i=0}^{n-1} \lambda_i=1$. In the method called $TD(\lambda)$ \citep{sutton1988learning}, $n \rightarrow \infty$ and $\lambda_i$ follow a geometric law: $\lambda_i \propto \lambda^i$ where $0 \le \lambda \le 1$.

\paragraph{To bootstrap or not to bootstrap?}

Bootstrapping has both advantages and disadvantages. On the negative side, using pure bootstrapping methods (such as in DQN) are prone to instabilities when combined with function approximation because they make recursive use of their own value estimate at the next time-step. On the contrary, methods such as n-step Q-learning rely less on their own value estimate because the estimate used is decayed by $\gamma^n$ for the n$^{th}$ step backup. In addition, methods that rely less on bootstrapping can propagate information more quickly from delayed rewards as they learn directly from returns \citep{sutton1996generalization}. Hence they might be more computationally efficient.

Bootstrapping also has advantages. The main advantage is that using value bootstrap allows learning from off-policy samples.
Indeed, methods that do not use pure bootstrapping, such as n-step Q-learning with $n>1$ or $TD(\lambda)$, are in principle on-policy based methods that would introduce a bias when used with trajectories that are not obtained solely under the behavior policy $\mu$ (e.g., stored in a replay buffer).

The conditions required to learn efficiently and safely with eligibility traces from off-policy experience are provided by \cite{munos2016safe,harutyunyan2016}.
%In the policy evaluation setting, we are given a fixed policy $\pi$ whose value $Q^\pi$ we wish to estimate from sample trajectories drawn from a behavior policy $\mu$. 
In the control setting, the retrace operator \citep{munos2016safe} considers a sequence of target policies $\pi$ that depend on the sequence of Q-functions (such as $\epsilon$-greedy policies), and seek to approximate $Q^*$ (if $\pi$ is greedy or becomes increasingly greedy w.r.t. the Q estimates).
%In both cases, 
It leads to the following target:
\begin{small}
\begin{equation}
%\mathcal R Q(s,a)
%Y = Q(s,a) + \left[ \sum_{t \ge 0} \gamma^t \left( \prod_{c_s=1}^{t} c_s \right) \left(r_t + \gamma \operatorname*{max}_{a' \in A} Q(s_{t+1},a') - Q(s_t,a_t)\right) \right]
Y = Q(s,a) + \left[ \sum_{t \ge 0} \gamma^t \left( \prod_{c_s=1}^{t} c_s \right) \left(r_t + \gamma \mathbb E_\pi Q(s_{t+1},a') - Q(s_t,a_t)\right) \right]
\end{equation}
\end{small}%
where $c_s=\lambda \min \left(1,\frac{\pi(s,a)}{\mu(s,a)}\right)$ with $0 \le \lambda \le 1$ and $\mu$ is the behavior policy (estimated from observed samples). This way of updating the Q-network has guaranteed convergence, does not suffer from a high variance and it does not cut the traces unnecessarily when $\pi$ and $\mu$ are close. Nonetheless, one can note that estimating the target is more expansive to compute as compared to the one-step target (such as in DQN) because the Q-value function has to be estimated on more states.

\section{Combination of all DQN improvements and variants of DQN}

The original DQN algorithm can combine the different variants discussed in \S \ref{sec:double} to \S \ref{sec:multi_step} (as well as some discussed in Chapter \ref{sec:explo-exploit}) and that has been studied by \cite{hessel2017rainbow}. Their experiments show that the combination of all the previously mentioned extensions to DQN provides state-of-the-art performance on the Atari 2600 benchmarks, both in terms of sample efficiency and final performance. Overall, a large majority of Atari games can be solved such that the deep RL agents surpass the human level performance. %The main difficulties challenges are related to exploration \ref{ch:challenges_online}.

Some limitations remain with DQN-based approaches. Among others, these types of algorithms are not well-suited to deal with large and/or continuous action spaces. In addition, they cannot explicitly learn stochastic policies. %(see Section \ref{sec:comb_pol_Q} for a simple modification that deals with this).
Modifications that address these limitations will be discussed in the following Chapter \ref{ch:policy-based_methods}, where we discuss policy-based approaches.
Actually, the next section will also show that value-based and policy-based approaches can be seen as two facets of the same model-free approach. Therefore, the limitations of discrete action spaces and deterministic policies are only related to DQN.

One can also note that value-based or policy-based approaches do not make use of any model of the environment, which limits their sample efficiency. %while, for some tasks, using a model is more efficient.
Ways to combine model-free and model-based approaches will be discussed in Chapter \ref{ch:model-based}.










DQN: from 1810.07862 or 1904.06736

survey
rainbow

# Algorithms review

rainbow
++new