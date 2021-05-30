# <font color="purple">Model-Agnostic Meta-Learning (MAML)</font>
### Paper
* [[ICML 2017] Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks](https://arxiv.org/pdf/1703.03400.pdf)
### Github
* https://github.com/cbfinn/maml
* https://github.com/AdrienLE/ANIML/blob/master/ANIML.ipynb
### Reference
* [Reference 1](https://towardsdatascience.com/paper-repro-deep-metalearning-using-maml-and-reptile-fd1df1cc81b0)
* [Reference 2](https://zhuanlan.zhihu.com/p/57864886?fbclid=IwAR27m394_8JgLTz7bwvWDMVIKG2FWUqg2-vbv2gVjWi3cXK_dmzys-4no64)
* [Reference 3](https://youtu.be/vUwOA3SNb_E?list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4)

## Note
### **Abstract**
此work提出一種meta-learning演算法，該演算法是<font color="red">與模型無關的(model-agnostic)</font>，適用於任何<font color="red">利用梯度下降的方法來訓練的模型</font>，並且<font color="red">適用於各式各樣的學習問題</font>，包括：classification、regression和reinforcement learning。meta-learning的目標是在不同的學習任務上訓練一個模型，使得該模型可以僅僅利用少量的training samples，就可以解決new learning tasks(新的學習任務)。在該方法中，對模型的參數(the parameters of the model)進行了顯式訓練(are explicitly trained)，以使用來自新任務的少量訓練資料進行少量gradient steps，將使模型參數對該新任務產生良好的泛化性能(generalization performance)。從效果上來說，該方法得到的模型更加容易進行微調(fine-tune)。該work證明了這種方法在兩個few-shot image classification benchmarks（也就是Omniglot和MiniImagenet）上具有最先進的性能、在few-shot regression上產生了良好的結果，並加速了使用neural network policies(神經網絡策略)進行policy gradient reinforcement learning(策略梯度強化學習)的fine-tuning(微調)。

### **1. Introduction**

### **2. Model-Agnostic Meta-Learning**
我們的目標是訓練可以實現<font color="red">rapid adaptation(快速適應)</font>的模型，而問題的設置通常被歸類為few-shot learning。在本節中，將定義問題的設置並介紹演算法的general form。

#### 2.1. Meta-Learning Problem Set-Up
few-shot meta-learning的目標是訓練僅需幾個datapoints和training iterations就可以快速適應新任務的模型。為此，model或learner在meta-learning階段接受了一組任務的培訓，從而使經過訓練的模型僅使用少量examples或trials就可以快速適應新任務。事實上，<font color="red">meta-learning problem將整個任務視為training examples</font>。（個人想法：在一般機器學習中，我們常將task中的一筆data視為一個training example，但考慮meta-learning問題時，會將整個task的所有data視為一個training example。此觀念可以參考[[Y. WANG et al. 2020]](https://arxiv.org/pdf/1904.05046.pdf)的Fig. 16。）在本節中，將以一般的方式來形式化此meta-learning problem設置，包括不同學習領域的簡要範例。將在第3節中詳細討論兩個不同的學習領域（即：監督式回歸與分類、強化學習）。

考慮一個以$f$表示的模型，該模型將observations(觀測值)$\mathbf{x}$映射到outputs(輸出)$\mathbf{a}$，即：<font color="red">$f(\mathbf{x})=\mathbf{a}$</font>。在meta-learing階段的期間，對模型進行訓練，使其能夠適應大型或無限數量的任務。由於，希望將此框架應用於從分類到強化學習的各種學習問題，因此，以下將介紹<font color="red">learning task(學習任務)的generic notion(一般概念)</font>。形式上，將每個任務表示為<font color="red">$\mathcal{T}=\left\{ \mathcal{L}(\mathbf{x}_1,\mathbf{a}_1,\cdots,\mathbf{x}_H,\mathbf{a}_H),q(\mathbf{x}_1),q(\mathbf{x}_{t+1}|\mathbf{x}_{t},\mathbf{a}_t),H \right\}$</font>，其中：$\mathcal{T}$由損失函數(loss function) $\mathcal{L}$、在intial observations(初始觀測值)$\mathbf{x}_1$的distribution(機率分佈) $q(\mathbf{x}_1)$、transition distribution $q(\mathbf{x}_{t+1}|\mathbf{x}_{t},\mathbf{a}_t)$、episode length(具dependent關係的樣本長度、觀測值數量) $H$組成。在獨立且同分布(independent and identically distributed, i.i.d.)的監督式學習問題，$H=1$。
p.s. 甚麼是episode length $H$呢？個人在閱讀section 3.2.時，內容有提到：timestep(時間步長) $t\in \left\{ 1,\cdots,H \right\}$，故推測$H$為timestep的長度。更直覺來說，$H$就是該任務$\mathcal{T}$所考慮的，具有dependent關係的觀測值$\mathbf{x}$數量、樣本長度。
該模型可以透過在每個時間$t$，選擇輸出$\mathbf{a}_t$，來生成樣本長度(samples of length) $H$。
p.s. 我覺得這句話的意思就是指：考慮任務$\mathcal{T}$，我如果選擇了$H$個時間的$\mathbf{x}$及$\mathbf{a}$，這意味著，該任務的樣本長度為$H$。
損失函數$\mathcal{L}(\mathbf{x}_1,\mathbf{a}_1,\cdots,\mathbf{x}_H,\mathbf{a}_H) \rightarrow\mathbb{R}$提供了task-specific feedback(特定於任務的反饋)，在Markov decision process中，可能以misclassification loss(錯誤分類損失)或cost function(成本函數)的形式表示。

在meta-learning的情境中，考慮一個在任務$\mathcal{T}$的機率分佈 $p(\mathcal{T})$，而我們希望我們的模型能夠適應於該分佈。<font color="green">在K-shot learning的設置中，模型以學習從$p(\mathcal{T})$提取的新任務$\mathcal{T}_i$進行訓練，其中：$\mathcal{T}_i$是從$q_i=q(\mathbf{x}_i)$中提取的K個樣本得出，並且該模型以feedback(反饋)由$\mathcal{T}_i$生成的$\mathcal{L}_{\mathcal{T}_i}$進行訓練。(不確定我對於原文的翻譯及理解是否正確)</font>
p.s. 該論文把
1. 在觀測值$\mathbf{x}$的機率分佈以$q$表示。
2. 在任務$\mathcal{T}$的機率分佈以$p$表示。

個人心得：從觀測值$\mathbf{x}_i$的機率分佈$q(\mathbf{x}_i)$中提取的K個樣本，可以獲得$\mathcal{T}_i$。而從$p(\mathcal{T})$提取新任務$\mathcal{T}_i$，我覺得可以藉由上述提到的$\mathcal{T}=\left\{ \mathcal{L}(\mathbf{x}_1,\mathbf{a}_1,\cdots,\mathbf{x}_H,\mathbf{a}_H),q(\mathbf{x}_1),q(\mathbf{x}_{t+1}|\mathbf{x}_{t},\mathbf{a}_t),H \right\}$作為理解的出發點，從$\mathcal{T}$中，藉由某種數學關係進行轉換，可能可以取得$q(\mathbf{x}_i)$，再從$q(\mathbf{x}_i)$中提取K個樣本，就可以獲得$\mathcal{T}_i$。

在meta-learning期間，從$p(\mathcal{T})$取樣任務$\mathcal{T}_i$<font color="green">（個人想法：$\mathcal{T}_i$是從$q_i=q(\mathbf{x}_i)$中提取的K個樣本得出，所以後面才會提到使用K個樣本來訓練模型。）</font>，使用K個樣本和來自$\mathcal{T}_i$的對應損失$\mathcal{L}_{\mathcal{T}_i}$的反饋來訓練模型，然後對來自$\mathcal{T}_i$的新樣本進行測試。然後，透過考慮來自$q_i$的新資料的測試誤差，如何針對於(w.r.t.)參數變化來改進模型$f$。實際上，$\mathcal{T}_i$上的測試誤差是meta-learning過程中的訓練誤差。在meta-training結束時，從$p(\mathcal{T})$中取樣新任務，並根據從K個樣本中學習後的模型性能來衡量meta-performance。通常，用於meta-testing的任務會在meta-training期間保留下來。(我認為，該句話的意思就是指：我們通常不會把testing的資料拿進去training使用，testing與training所使用的task是分開的。)

個人總結：本section 2.1.主要重點是，meta-learning會將整個任務視為training example、定義learning task、few/K-shot meta-learning的基本設置、meta-testing與meta-training所使用的任務是分開的。

#### 2.2. A Model-Agnostic Meta-Learning Algorithm
在過去的works中，探討recurrent neural networks(RNN)（在此work中，試圖訓練吸收整個datasets的RNN）及feature embeddings（在測試時，可以與nonparametric methods結合的feature embeddings），相較於這些work，該論文提出一種<font color="blue">可以透過meta-learing來學習the parameters of any standard model(任何標準模型的參數)，從而實現fast adaptation(快速適應)的模型</font>的方法。這種方法背後的intuition(直覺)是，某些internal representations(內部表示形式)比其他內部表示形式more transferrable(更易於轉移)。例如：神經網絡可能會學習到能廣泛適用於$p(\mathcal{T})$中所有任務而不是單個任務的internal features(內部特徵)。那麼，我們要如何encourage(促使)這種general-purpose representations(通用表示形式)的emergence(出現)？該work對這個問題採取了明確的做法：由於，將在新任務上使用gradient-based learning rule(基於梯度的學習規則)對模型進行fine-tuned(微調)，因此，將致力於以一種使基於梯度的學習規則能夠rapid progress(快速發展)的way(方式)來學習模型，並使模型不會對從$p(\mathcal{T})$提取的新任務overfitting(過度擬合)。實際上，該work的目標是找到<font color="red">對任務的變化sensitive(敏感的)模型參數</font>，如此一來，當altered in the direction of the gradient of that loss(在損失的梯度方向上進行更改)時，在參數上的小變化，將對從$p(\mathcal{T})$提取的任何任務的損失函數產生較大的改善，如Figure 1。
![](https://i.imgur.com/7ZyHVqp.png)
除了<font color="red">假設模型由某些參數向量$\theta$參數化</font>之外，<font color="red">不對模型的形式進行任何假設</font>（個人理解：也就是Model-Agnostic的概念），並且損失函數在$\theta$中足夠平滑，因此，可以使用<font color="red">gradient-based learning techniques(基於梯度的學習技術)</font>。

形式上，以一個帶參數$\theta$的parametrized function(參數化函數)$f_{\theta}$表示模型。當適應於新任務$\mathcal{T}_i$時，模型的參數$\theta$變為$\theta_{i}^{'}$。在該paper的方法中，使用任務$\mathcal{T}_i$上的一個或多個gradient descent updates(梯度下降更新)來計算updated parameter vector(更新的參數向量)$\theta_{i}^{'}$。例如，當使用一個gradient update時，<font color="red">$\theta_{i}^{'}=\theta-\alpha\nabla_{\theta}\mathcal{L}_{\mathcal{T}_i}(f_{\theta})$</font>。（個人理解：$\alpha$為step size(步長)、learning rate。）而step size $\alpha$可能被固定為hyperparameter或meta-learned。為了簡化符號表示，在本節的其餘部分中，將考慮一個梯度更新，但是要將其延伸為使用多個gradient updates(梯度更新)是輕而易舉的。透過最佳化<font color="blue">從$p(\mathcal{T})$取樣的tasks、針對於$\theta$的$f_{\theta_{i}^{'}}$性能</font>來訓練模型參數。更具體而言，可以將meta-objective表示如下：
$\displaystyle \min_{\theta} \sum_{\mathcal{T}_i \sim p(\mathcal{T})}\mathcal{L}_{\mathcal{T}_i}(f_{\theta_{i}^{'}})=  \sum_{\mathcal{T}_i \sim p(\mathcal{T})}\mathcal{L}_{\mathcal{T}_i}(f_{\theta-\alpha\nabla_{\theta}\mathcal{L}_{\mathcal{T}_i}(f_{\theta})})$
注意到上式，這裡對模型參數$\theta$進行了meta-optimization，接著，使用updated model parameters(更新後的模型參數)$\theta^{'}$計算objective(目標)。事實上，該論文提出的方法致力於優化模型參數，以使新任務上的一個或少量gradient steps，將在該任務上產生最大的有效行為。across tasks(跨任務)的meta-optimization是藉由stochastic gradient descent(隨機梯度下降，SGD)執行的，因此模型參數$\theta$被更新如下：
$\displaystyle \theta \leftarrow \theta - \beta\nabla_{\theta}\sum_{\mathcal{T}_i \sim p(\mathcal{T})}\mathcal{L}_{\mathcal{T}_i}(f_{\theta_{i}^{'}})$
(Equation (1))
其中，$\beta$為meta step size（個人理解：$\beta$與$\alpha$都是learning rate）。在一般情況下，完整演算法在Algorithm 1中概述。
![](https://i.imgur.com/Bbw7u9T.png)
MAML的meta-gradient update涉及<font color="red">a gradient through a gradient(gradient by gradient、梯度的梯度)</font>。（個人理解：也就是說，MAML的計算會涉及到<font color="red">二階導數</font>，但二階導數計算成本較高。）在計算上，這會需要透過額外的backward pass(反向遍歷)$f$，來計算Hessian-vector products，這將由標準深度學習庫（如TensorFlow）支援。在後面section 5.2.中，該work將會試圖用first-order approximation的方法，來避免backward pass。

[Reference 2](https://zhuanlan.zhihu.com/p/57864886?fbclid=IwAR27m394_8JgLTz7bwvWDMVIKG2FWUqg2-vbv2gVjWi3cXK_dmzys-4no64)：
先定義兩個模型：$\mathcal{M}_{meta}$（用於學習一個好的inital parameters）、$\mathcal{M}_{fine-tune}$（用於預測問題的輸出）；
接著定義兩個dataset：$\mathcal{D}_{meta-train}$（用於訓練$\mathcal{M}_{meta}$）、$\mathcal{D}_{meta-test}$（用於訓練及測試$\mathcal{M}_{fine-tune}$）；
而在一個task $\mathcal{T}$中，會分為support set（用於訓練）及query set（用於第二次更新或測試）。

Algorithm 1是MAML取得$\mathcal{M}_{meta}$的過程。

在Algorithm 1中，第一個Require的$p(\mathcal{T})$，是指在$\mathcal{D}_{meta-train}$中，抽取出的task $\mathcal{T}$的分佈。
第二個Require的$\alpha$及$\beta$為learning rate，會需要兩個learning rate的原因，是因為MAML是基於gradient by gradient所設計，在每次的iteration中，會包含兩次參數更新。

Algorithm 1的step 3：從$\mathcal{D}_{meta-train}$中，抽取出數個tasks，形成一個batch。

Algorithm 1的step 4~7：利用batch中的每一個task，分別對模型進行參數更新。

Algorithm 1的step 5：利用batch中的某一個task中的support set，來計算每個參數的梯度。考慮N-way K-shot，support set會有NK個樣本。

Algorithm 1的step 6：進行MAML的第一次梯度更新。

Algorithm 1的step 4~7完成後，MAML完成了第一次的梯度更新。

Algorithm 1的step 8：MAML的第二次梯度更新。這裡是使用task中的query set，目的是增強$\mathcal{M}_{meta}$對於不同task之間的泛化能力、避免$\mathcal{M}_{meta}$對於support set產生overfitting的現象。

MAML取得$\mathcal{M}_{fine-tune}$的過程（參考Alogrithm 1的過程，做以下調整）：


| Step | Description                        |
| ---- | ---------------------------------- |
| 1    | 利用$\mathcal{M}_{meta}$來初始化參數。|
| 3    | 只需從$\mathcal{D}_{meta-test}$抽取一個task進行學習，故不需要batch。此task的support set用來訓練$\mathcal{M}_{fine-tune}$、query set用來測試$\mathcal{M}_{fine-tune}$。|
| 8    | MAML取得$\mathcal{M}_{fine-tune}$的過程，不需要step 8。因為task的query set是用來測試$\mathcal{M}_{fine-tune}$，對$\mathcal{M}_{fine-tune}$而言，query set是unlabeled data。因此，$\mathcal{M}_{fine-tune}$沒有第二次的梯度更新，而是利用第一次梯度計算的結果更新參數。|

舉例：
![](https://i.imgur.com/FJXBSMG.jpg)
![](https://i.imgur.com/QyHmEem.jpg)


### **3. Species of MAML**
#### 3.1. Supervised Regression and Classification
在supervised tasks(監督式任務)的領域中，few-shot learning被充分研究(well-studied)，其goal(目標)是<font color="red">使用similar tasks(類似任務)的prior data(先驗資料)進行meta-learning</font>，從該任務的幾個input/output pairs(輸入/輸出對)中學習new function。例如，goal(目標)可能是在僅看到一個或幾個Segway圖片之後，使用先前已看到許多其他物件類型的模型對Segway圖片進行分類。同樣地，在few-shot regression中，goal(目標)是在對許多具有similar statistical properties(相似統計特性)的函數進行訓練後，從該函數取樣的幾個datapoints中預測continuous-valued function(連續值函數)的輸出。

如同在section 2.1.中的描述（在獨立且同分布的監督式學習問題，$H=1$），這裡考慮$H=1$（具dependent關係的樣本長度為1），並將timestep $t$標記於$\textbf{x}$上，如$\textbf{x}_t$，因為模型接受單個輸入並產生單個輸出，而不是輸入和輸出序列。

對於使用mean-squared error的regression tasks，損失採用以下形式：
$\displaystyle \mathcal{L}_{\mathcal{T}_i}(f_{\phi})=\sum_{\textbf{x}^{(j)},\textbf{y}^{(j)} \sim \mathcal{T}_i} \left\Vert f_{\phi}(\textbf{x}^{(j)})-\textbf{y}^{(j)} \right\Vert_{2}^{2}$
(Equation (2))

同樣地，對於具有cross-entropy loss的離散分類任務，損失採用以下形式：
$\displaystyle \mathcal{L}_{\mathcal{T}_i}(f_{\phi})=\sum_{\textbf{x}^{(j)},\textbf{y}^{(j)} \sim \mathcal{T}_i} \textbf{y}^{(j)}\log(f_{\phi}(\textbf{x}^{(j)}))+(1-\textbf{y}^{(j)})\log(1-f_{\phi}(\textbf{x}^{(j)}))$
(Equation (3))

根據慣用術語，K-shot classification tasks使用來自每個類別的K個input/output pairs(輸入/輸出對)，共計NK個data points(資料點)用於N-way classification。給定$p(\mathcal{T}_i)$，這些損失函數可以直接插入section 2.2.的方程式中，以執行meta-learning，如Algorithm 2.中所述。
![](https://i.imgur.com/5zMa860.png)
個人理解：利用Algorithm 2，嘗試考慮regression tasks，則屬於one-way K-shot，如果是我們的work，可能是classification與regression混合。

#### 3.2. Reinforcement Learning

### 5. Experimental Evaluation
#### 5.1. Regression
我們從一個簡單的回歸問題開始，該問題說明了MAML的基本原理。每個任務都涉及從正弦波的輸入到輸出的回歸，其中，正弦波的振幅和相位在任務之間變化，即：$y=a\sin(x+b)$。因此，$p(\mathcal{T})$是連續的，其中：幅度$a$在$[0.1,5.0]$內變化、相位$b$在$[0,\pi]$內變化，並且輸入$x$和輸出$y$的維數均為1。在訓練和測試過程中，從$[-5.0,5.0]$中uniformly(均勻)取樣datapoints $\textbf{x}$。損失是預測值$f(\textbf{x})$與真實值$\textbf{y}$之間的mean-squared error(均方誤差)。 regressor是具有ReLU非線性、2個大小為40的隱藏層之神經網絡模型。在使用MAML進行訓練時，該work使用一個gradient update(梯度更新)（個人理解：我認為是指Algorithm 2的step 7），其中：$K=10$個examples、固定step size(步長) $\alpha= 0.01$，並使用Adam作為meta-optimizer。baselines同樣由Adam訓練。為了評估性能，該work針對不同數量的$K$ examples微調了一個single meta-learned model，並與兩個baselines比較性能：
1. 對所有任務進行pretraining(預訓練)，這需要訓練network以回歸到隨機正弦函數，然後在測試時，使用自動調整的step size對於被提供的K個points進行梯度下降的微調。（個人理解：就是一般的pretraining、fine-tuning的transfer learning方法。）
2. 接收實際幅度和相位作為輸入的oracle。（個人理解：就是想像成，已經知道$a$跟$b$，可以直接得到$y=a\sin(x+b)$。）

在$K = \left\{5, 10, 20\right\}$個datapoints上，透過微調MAML學習的模型（即Algorithm 2）和預訓練模型來評估性能。在微調期間，每個gradient step都使用相同的K個datapoints進行計算。定性結果（如Figure 2所示，並在Appendix B中進一步延伸）顯示，學習的模型能快速適應於僅有5個datapoints（顯示為紫色三角形）的任務；而對所有任務使用標準監督式學習進行預訓練的模型，則無法充分適應於僅有極少datapoints的任務，而不會發生災難性的過度擬合。
![](https://i.imgur.com/js3c55y.png)
更重要的是，當K個datapoints都在輸入範圍的一半邊時，使用MAML訓練的模型仍然可以推斷出該範圍另一半的振幅和相位，這表明MAML訓練的模型$f$已經學會了對正弦波的periodic nature(周期性)進行建模。此外，我們在定性和定量結果（Figure 3和Appendix B）中都觀察到，儘管經過了一個gradient step的maximal performance(最大性能)訓練，但使用MAML學習的模型仍會隨著gradient step的增加而不斷改進。這種改進，表明了MAML對參數進行了優化，以使其位於可快速適應且對$p(\mathcal{T})$的損失函數敏感的區域（如section 2.2.所述），而不是過度擬合僅在一步之後才能改善的參數。
![](https://i.imgur.com/dbZ6ckE.png)

### Cited in Papers (maybe similar or related with our work)
