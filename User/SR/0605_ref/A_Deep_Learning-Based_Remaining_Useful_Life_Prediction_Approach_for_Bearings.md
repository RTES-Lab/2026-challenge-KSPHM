Logo

<!-- image -->

Logo

<!-- image -->

## A Deep Learning-Based Remaining Useful Life Prediction Approach for Bearings

Cheng Cheng , Guijun Ma , Yong Zhang , Mingyang Sun , Member, IEEE , Fei Teng , Member, IEEE, Han Ding, Senior Member, IEEE, and Ye Yuan , Member, IEEE

Abstract— t— In industrial applications, nearly half the failures of motors are caused by the degradation of rolling element bearings (REBs). Therefore, accurately estimating the remaining useful life (RUL) for REBs is of crucial importance to ensure the reliability and safety of mechanical systems. To tackle this challenge, model-based approaches are limited by the complexity of mathematical modeling. Conventional data-driven approaches, on the other hand, require massive efforts to extract the degradation features and construct the health index. In this article, a novel data-driven framework is proposed to exploit the adoption of deep convolutional neural networks (CNNs) in predicting the RULs of bearings. More concretely, raw vibrations of training bearings are first processed using the Hilbert–Huang transform to construct a novel nonlinear degradation energy indicator which can be used as the training label. The CNN is then employed to identify the hidden pattern between the extracted degradation energy indicator and the raw vibrations of training bearings, which makes it possible to estimate the degradation of the test bearings automatically. Finally, testing bearings' RULs are

Manuscript received August 19, 2019; accepted January 23, 2020. Date of publication February 4, 2020; date of current version June 15, 2020. Recommended by Technical Editor J. Scruggs. This work was supported in part by the National Natural Science Foundation of China under Grant 91748112 and in part by the Primary Research and Development Plan of Jiangsu Province under Grant BE2017002. (Corresponding author: Ye Yuan.)

- C. Cheng is with the Key Laboratory of Image Processing and Intelligent Control, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan 430074, China (e-mail: c\_cheng@hust.edu.cn).
- G. Ma and H. Ding are with the State Key Laboratory of Digital Manufacturing Equipment and Technology, Huazhong University of Science and Technology, Wuhan 430074, China, and also with the School of Mechanical Science and Engineering, Huazhong University of Science and Technology, Wuhan 430074, China (e-mail: mgj@hust.edu.cn; dinghan@hust.edu.cn).
- Y. Zhang is with the School of Information Science and Engineering, Wuhan University of Science and Technology, Wuhan 430081, China (e-mail: zhangyong77@wust.edu.cn).
- M. Sun is with the College of Control Science and Engineering, Zhejiang University, Hangzhou 310007, China (e-mail: mingyangsun@zju.edu.cn).
- F. Teng is with the Department of Electrical and Electronic Engineering, Imperial College London SW7 2AZ, London, U.K. (e-mail: f.teng@imperial.ac.uk).
- Y. Yuan is with the Key Laboratory of Image Processing and Intelligent Control, School of Artificial Intelligence and Automation, Huazhong University of Science and Technology, Wuhan 430074, China, and also with the State Key Laboratory of Digital Manufacturing Equipment and Technology, Huazhong University of Science and Technology, Wuhan 430074, China (e-mail: yye@hust.edu.cn).

Color versions of one or more of the figures in this article are available online at https://ieeexplore.ieee.org.

Digital Object Identifier 10.1109/TMECH.2020.2971503

predicted through using an --support vector regression model. The superior performance of the proposed RUL estimation framework, compared with the state-of-the-art approaches, is demonstrated through the experimental results. The generality of the proposed CNN model is also validated by performance test on other bearings undergoing different operating conditions.

Index Terms— s— Convolutional neural networks (CNNs), Hilbert–Huang transform (HHT), remaining useful life (RUL) estimation, rolling bearings.

## NOMENCLATURE

|              | CNN Convolutional neural network.               |
|--------------|-------------------------------------------------|
|              | DEI Degradation energy indicator.               |
|              | EMD Empirical mode decomposition.               |
|              | ETA Exponential transformed accuracy.           |
|              | HHT Hilbert–Huang transform.                    |
|              | IMF Intrinsic mode function.                    |
|              | MAE Mean average error.                         |
|              | NRMSE Normalized root-mean-square error.        |
|              | MSE Mean-square error.                          |
|              | REB Rolling element bearing.                    |
|              | RUL Remaining useful life.                      |
|              | SVR Support vector regression.                  |
| FT or Lf t   | Failure threshold.                              |
| N            | Length of historical units of training bearing. |
| Q            | Length of historical units of test bearing.     |
| U            | Number of predicted units of test bearing.      |
| S i          | Sensor measurement signal of ith unit.          |
| Δt           | Sampling time.                                  |
| P            | Number of measurements in ith unit.             |
| τ            | Time interval between two recording phases.     |
| L            | DEI of training bearing.                        |
| L norm       | Normalized DEI of training bearing.             |
| L test       | Estimated DEI of test bearing.                  |
| L  U , test | Predicted DEI of test bearing.                  |
| K            | Total number of layers.                         |
| W k          | Weights in kth convolutional layer.             |
| B k         | Bias in kth convolutional layer.                |
| T            | Predicted RUL of test bearing.                  |
|  f T         |                                                 |
|  failure     |                                                 |
| Tf Tfailure  | Real RUL of test bearing.                       |
| β            | Number of test bearings.                        |
| E r          | % Relative percentage error of prediction.      |
| Sm Smean     | Average score of prediction.                    |
| X            | Training set for SVR.                           |

1083-4435 © 2020 IEEE. Personal use is permitted, but republication/redistribution requires IEEE permission. See https://www.ieee.org/publications/rights/index.html for more information.

R Set of real numbers.

Z

Set of positive integers.

## I. INTRODUCTION

T O CONSTRAIN relative motion while reducing friction between moving parts, rolling element bearings (REBs) are one of the most widely used elements in industrial machinery. Prognostics and health management of bearings is of significance for safety, reliability, and effectiveness of the mechanical systems [1], [2]. In [3], it is shown that nearly half of motor failures are related to the degradation of bearings. As such, estimating the remaining useful life (RUL) (i.e., time-to-failure prognostics) of bearings has attracted a great deal of attention in recent years [4]. RUL prediction helps users monitor the condition of the bearings and provides an estimation of time left before a failure occurs. Compared with fault diagnosis, which has been well investigated over past few decades [5], the problem of RUL prediction studied in this article is a relatively new and challenging topic due to the huge amount of uncertainties of environment and operating condition.

In general, RUL prediction approaches can be categorized into model-based and data-driven approaches. Model-based approaches aim to build a physical model to represent the degradation of the rolling bearing [6]. Li et al. [7] predicted the defect growth on a bearing unit using Paris's law for fatigue. However, it is difficult to construct a precise physical degradation model due to the sensitivity of the model parameters and noised operating environments. This limits the practical applications of the model-based approaches. On the other hand, data-driven approaches benefit from the extensive expertise in signal processing and machine learning [8], and infer the degradation process of bearings without knowing any physics of degradation failure. The prognostic framework of the data-driven approaches mainly consists of three stages: 1) feature extraction from noisy sensory signals, which helps to build up the health indicator for the learning of system degradation behavior; 2) degradation models are trained on the training bearing using statistical or machine learning techniques; and 3) the degradation indicator of the test bearings can be estimated based on the model trained in the second stage. Then, the unknown degradation process can be predicted by applying regression techniques (i.e., -SVR).

To extract features from raw signals, time-domain, frequencydomain, and time–frequency-domain analysis are commonly adopted. Among them, the time–frequency analysis has been found to be the most efficient due to its ability to characterize transient signals over time and frequency domains [9]. Well known time–frequency techniques for extracting bearing features include short-time Fourier transform [10], wavelets [11], Wigner–Ville distribution [12], and Hilbert–Huang transform (HHT) [13]. Implementation of the short-time Fourier transform is limited by its time-frequency resolution capability; for instance, low frequencies are difficult to identify with short windows. On the other hand, wavelets and the Winger–Ville distribution provide richer pictures than short-time Fourier transform; however, their effectiveness depends on estimating the

Hurst parameter and the quality of the analyzed signal. HHT shows better computational efficiency and resolution over other time–frequency analysis which uses the techniques of empirical mode decomposition (EMD) and Hilbert transform (HT) to decompose the original vibration signal into a number of intrinsic mode functions (IMFs) in various frequency scales. Frequency components of each IMF are related to both the sampling frequency and the signal itself, thus demonstrating that HHT is a self-adaptive signal processing technique perfectly suited to nonstationary signals. Wu et al. [14] analyzed the timeto-failure prognostics of REBs, which extracts ten statistical features using time and frequency analysis and 11 IMF features using HHT time–frequency analysis. The gear fault identification method proposed in [15] is based on the HHT, and the first six IMFs are selected as inputs for SOM neural networks for fault diagnosis.

Data-driven RUL prediction approaches are mainly based on statistical and machine learning techniques, such as artificial neural networks (ANN) [16], fuzzy logic systems [17], and autoregressive (AR) models [18]. The computational cost of ANN is relatively high in terms of optimizing the weights of the model. The performances of the AR models and fuzzy logic systems require a precise trend of historical observations and high-quality training data, respectively. Recently, deep learning has merged into research and industry fields, and has beaten other machine learning techniques in speech recognition and image recognition tasks [19]. Deep learning model is good at discovering highlevel abstractions from labeled data using a back-propagation algorithm [20]. Specifically, it learns feature representations automatically rather than designing the hand-created features by experience. As the most well known model in deep learning, in recent years, CNN dominates the recognition and detection problems in computer vision domain, which is distinguished by three characteristics, namely, local connections, shared weights, and local pooling [21], [22]. The first two characteristics show that the CNN model requires fewer parameters to detect local information than multilayer perceptron, while the last characteristic ensures shift invariance to the networks. Typically, one-dimensional (1-D) CNN will be employed to this article to learn the latent space of input sensory time-series vibrations, which has been applied with great success to speech recognition and document reading tasks. Few attempts have been made for the prediction problem using CNN-based models [23]–[26]. This article exploits the adoption of CNN technique in estimating the RUL of bearings, as a prognosis problem, to learn about the nonlinear degradation behavior according to raw vibration data and an extracted label. Instead of using the CNN technique to perform the time-series prediction, the main function of the CNN model in this article is to reveal the hidden dependencies between the vibration data and the DEI of the training bearing, which makes full use of the advantages of CNN in automatic feature extraction.

In this article, we propose a data-driven framework for predicting the RUL of REBs by applying the HHT, CNN, and  -SVR. The raw vibration signals collected from sensors are processed by the HHT method and a novel time-series degradation indicator, i.e., DEI, is constructed. Subsequently, a CNN

Line chart

Fig. 1. Linear time degradation indicator (blue) versus nonlinear DEI (black), as a label for network training. The nonlinear DEI experiences a long time flat curve before a sharp degradation trend when it gets close to the end of bearing lifetime, which reflects the real degradation process of most machinery systems [27]. With the aggregation of damages at different bearing components close to the end of lifetime, the simple time degradation indicator is less effective than DEI for RUL estimation.

<!-- image -->

model is trained to learn the features from the input raw vibration to the DEI label on the training bearings, and used to predict the DEIs of testing bearings. Then, an -SVR model is introduced so that the evolution of the degradation can be forecast till the bearing failure. The effectiveness of the proposed framework for RUL prediction is validated on an experimental platform (i.e., PRONOSTIA). Much lower RUL prediction errors are achieved, compared with eight existing approaches in previous papers and two tested methods designed in this article, indicating the superior performance of the proposed method.

This article makes the following contributions.

- 1) The proposed method successfully extracts a novel nonlinear DEI (see Fig. 1, compared with the linear time degradation indicator) to describe the degradation trend of the training bearing, according to the natural frequencies of bearing components.
- 2) The proposed CNN architecture is general and robust for similar operating conditions—it can transfer to another bearing undergoing different operating conditions and obtain good prediction results, without changing CNN hyperparameters and the depth of layers.
- 3) The proposed DEI is an integrated indicator with regard to the maximum vibration levels among different bearing components which considers all the possible defects on the REB. This is a more realistic indicator as the localized defects are not initially initiated in real industrial applications, meaning that all the types of defects have to be considered.
- 4) CNN scales all the indicators of training bearings and test bearings into a consistent latent space. Thus, training and testing can share the same failure threshold (FT), i.e., the maximum value of the indicator for the training bearing.

The rest of this article is organized as follows. Section II presents the proposed RUL prediction framework with technical details. In Section III, experimental results obtained from bearing degradation tests are carried out. Thereby, the performance of the proposed framework is validated and the results show improved accuracy in predicting the RUL compared with eight state-of-the-art approaches and two designed test methods in this article. Finally, Section IV concludes this article.

## II. DEGRADATION INDICATOR TRAINING AND RUL PREDICTION ALGORITHM

The overall framework for the prediction of the RUL can be decomposed into three parts. The schematic of the overall framework is shown in Fig. 2. The key challenges of this article involve: 1) obtain the DEI to represent the degradation behaviour; 2) establish a CNN model to map raw vibration signal to the DEI; and 3) construct an -SVR to predict the RUL. Thus, in the following subsections, the explicit expression of degradation feature extraction, CNN model, and -SVR forecasting model will be derived in Sections II-A–II-C.

## A. Degradation Indicator Extraction

To begin with, for a training bearing, it is assumed that the raw vibration signal till the end of lifetime with N ∈ Z historical units have been acquired. The sensory vibration signal S i = {Si(t p )} P p=1 , with sampling time Δt, is measured at each historical unit fori ∈ D = {1 , 2, . . ., N}, whereP is the number of measurements recorded in each historical unit.

EMD is a self-adaptive method which is normally applied to analyze nonstationary and nonlinear signals. It decomposes the raw vibration data S i into n number of IMFs, illustrating the natural oscillation modes from fast to low oscillations.

For the ith unit, the jth mode (j = 1,...,n) of IMF, IMF i,j (t p ) = a v i,j (t p ) for p = 1,...,P, is calculated iteratively associated with the iteration number v = 0 , 1 ,... .

First, let v = 0 and initialize a 0 i,j (t p ) for p = 1,...,P by

<!-- formula-not-decoded -->

Define IMF i,j (t p ) = a 0 i,j (t p ) for p = 1,...,P only if the IMF meets the following two conditions.

- I) The IMF should have one or zero difference between the extrema number and the number of zero crossing.
2. II) Along the time axis, the average value of upper and lower bound of the IMF should be zero everywhere.

Otherwise, update the IMFi,j (t p ) = a v i,j (t p )for p = 1,...,P through the following iteration procedure with v = 1 , 2 ,... until the IMF satisfies both the aforementioned conditions (I) and (II)

<!-- formula-not-decoded -->

where ω v − 1 i,j (t1),...,ω v − 1 i,j (tP ) are the mean values of the upper envelope and the lower envelope of a v − 1 i,j (t1),...,a v − 1 i,j (tP ) .

Once obtaining all IMFs, the analytical form of each IMF can be written as

<!-- formula-not-decoded -->

Flow chart

Fig. 2. Data-driven RUL prediction framework. This framework involves three stages: 1) a feature extraction stage aims to extract the degradation indicator. The DEI is extracted using EMD, HT, and MHS. The value of DEI is also related to the natural frequencies of different bearing components; 2) a model training stage aims to train a CNN model based on the training data that discovers the hidden pattern between the DEI and the raw vibration data; and 3) a prediction stage predicts the RUL according to the trained CNN model and an --SVR forecasting model.

<!-- image -->

where i is the imaginary part of the IMF A i,j (t p ). IMF H i,j (t p ) is the HT of IMF i,j (t p ) by convolution with function 1 πt , given as

<!-- formula-not-decoded -->

By this means, we can calculate the instantaneous amplitude h i,j (t p ) and phase φi,j (t p )

<!-- formula-not-decoded -->

Accordingly, it is easy to derive the instantaneous frequency

<!-- formula-not-decoded -->

for p = 1,...,P .

Then, the Hilbert spectrum of Si is obtained by

<!-- formula-not-decoded -->

The marginal Hilbert spectrum (MHS) Mi(fi) can be written as

<!-- formula-not-decoded -->

Natural frequencies of bearing components depend on the geometry of the bearing and its rotation speed. Expression of these frequencies are given in Table I, where η is the number of balls, fω fω is the rotation frequency, Φ is the contact angle, and 	ball and pitch are the ball diameter and pitch diameter, respectively.

With the bearing frequencies of different components, the value of the DEI L i at historical unit i is defined as the maximum

TABLE I BEARING FREQUENCIES OF INNER RACE, OUTER RACE , AND BALL [28]

| Symbol              | Description                                              | Expression                                                |
|---------------------|----------------------------------------------------------|-----------------------------------------------------------|
| finner fouter fball | Inner race frequency Outer race frequency Ball frequency | 2 fw (1 + cOS epitch 2 fw cOS Ppitch Qpitch 22 pitch cos2 |

value of the MHS by substituting finner , fouter, and fball into Mi Mi (fi), given that

<!-- formula-not-decoded -->

The extracted DEI L = [L1, . . ., LN ] is normalized before training

<!-- formula-not-decoded -->

for i = 1,...,N, and  is an infinitesimal that is used to avoid the value of label equal to zero or one. Thus, the normalized DEI is

<!-- formula-not-decoded -->

## B. DEI Pattern Learning

In this article, layers with repeated components are stacked in a CNN architecture, including convolutional layers, pooling layers, fully connected layers, and a regression layer [21].

Convolutional layer contains organized patches in convolutional layers, each patch is calculated by composing the features of the previous layer through a filter bank with the following equation:

<!-- formula-not-decoded -->

Bar chart

Fig. 3. Diagram of extracting features for --SVR prediction, where l is the sampling window size, s is the sliding size, and x g stores the mean value μ g and variance σ 2 g of each sampling window.

<!-- image -->

where u k m ∈ R denotes the output of the mth unit in the kth layer. u (k−1) e ∈ R 1×o k is the input data of the eth subvector in the previous layer k − 1, where o k is kernel size in layer k . W k ∈ R 1×o k and B k ∈ R denote the connecting weights and bias in the kth layer, respectively. "∗" means the convolution operation. It is noted that when k = 1, u (k−1) e is a subvector of the raw vibration data S i . Here, we define all neurons in each layer as u k = [u k k
1 ,...,u k m ,...,u k Gk ] for k ∈ Dk = {1,2,...,K}, where G k ∈ Z is the number of neurons in the kth layer and K is the total number of layers. For convolutional layer, G k = (G k − 1 − o k )/I
c k I
cv + 1 and I
c k I
cv ∈ Z is the stride in convolutional layer.

Activation function is introduced after convolutional layer. Among various activation functions, rectified linear unit (ReLU) r k m = max(0, u k m )is chosen as the nonlinear activation function to prevent the issue of vanishing gradient which may significantly increase the training time or even lead to nonconvergence.

Pooling layeris then used as a nonlinear down-sampling layer to extract the maximum feature values in each patch of the input data. Its function is to save computation time and downsize the number of parameters of the model as well as control overfitting. More specifically, pooling transforms small windows into single values by maxing or averaging. Consequently, the features extracted within the small window are similar and therefore, illustrate the shift invariance property of CNN. Max-pooling layer is selected in this article as it is an algorithmic choice to ensure the generalization of neural networks [29], which is given by

<!-- formula-not-decoded -->

where λ k ∈ Z is the pooling size and I
p k I
pl is the stride in max pooling layer.

Fully connected layer and regression layer, like a classic ANN network, take the results of the convolution and max-pooling processes and use them to generate a predicted label. Since we use a normalized DEI L norm ∈ R 1×N as the label for learning, the sigmoid function sigm(u K − 1 ) with an output value between (0,1) is applied to the last layer for normalized output. Hereby, mean-square error (MSE) function is used to compute the loss

with the expression

<!-- formula-not-decoded -->

where the proposed CNN model is minimizing the loss function z between ground label DEI L norm and predicted label L ˜ norm. Algorithm 1 outlines the proposed CNN modeling procedure.

## C. RUL Prediction

With the obtained CNN model, for a new test bearing with Q ∈ Z historical units, the estimated DEI Ltest = [L1 , test ,...,LQ,test] ∈ R 1×Q can be automatically generated by the trained CNN with the new vibration signal Si, where
i ∈ D test = {1,...,Q}. Then, to predict the RUL T
f -T
failure ∈ R + , an -SVR forecasting model [31] is formalized to predict the

Bar chart

Fig. 4. Overview of the PRONOSTIA platform [32].

<!-- image -->

upcoming degradation L -Q+1 , test, L -Q+2 , test, ... based on the estimated DEI L test by sliding window method. The forecasting process contains the following three substeps.

- 1) Extract training features from the estimated DEI Ltest over a sliding window. The schematic of this step is illustrated in Fig. 3. The estimated DEI Ltest is decomposed into overlapping windows associated with sampling window size l and sliding size s . x g = (μ g , σ 2 g ) for g ∈ {1 ,..., Q−l−1 s + 1} represents a training feature for -SVR, where μ denotes the mean value and σ 2 denotes the variance of each sampling window. Thus, the training set for -SVR X = [(x1, Ll+1 , test ) ,..., (x g , L( g − 1)s+l+1 , test ) ,..., (x Q − l − 1 s +1, LQ,test)] is obtained, where L( g − 1)s+l+1 , test corresponds to the next value in L test of the gth sampling window.
- 2) -SVR modeling is described in Algorithm 2, while at the application level, two parameters (distance limit  ∈ R and penalty parameter C ∈ R) can be set manually when training the prediction model. A radial basis function (RBF) is necessary when we intend to train a nonlinear model.
- 3) The SVR model f(x) learned in Algorithm 2 is then used to predict the RUL by sliding window method (with the same l and s in step 1). Since the test bearing is undergoing the same operating condition as the training bearing, it is reasonable to define the FTs (denoted as Lf t) of the test bearing equal to the last feature of the DEI of the training bearing, such that Lf t=LN . Hence, the first prediction can be calculated as L -Q+1 , test = f(x Q − l s +1 ), and the predicted DEI L -U , test = [L -Q+1 , test,..., L -Q+U,test ] can be obtained by shifting the sampling window, with

## III. EXPERIMENTS

## A. Data Description

The validation of the proposed RUL prediction framework is conducted on an experimentation platform named PRONOSTIA (see Fig. 4). This platform is built as a combination of three

## Algorithm 2: Framework of -SVR.

## Require:

A training set X;

A distance limit  and a penalty parameter C;

A kernel function named RBF with the equation:

<!-- formula-not-decoded -->

## Ensure:

A regression model like f(x) = w T φ(x) + b, where w and b are optimized parameters and φ(x) is the x -mapped eigenvector that satisfies the equation:

κ (x g , xq) = 	φ(x g ), φ(x q ) .

Step 1: Establish optimization problems using C ,  , X , and two slack variables ξ g and ξ -g (slack degree for upper boundary and lower boundary, respectively) as following:

<!-- formula-not-decoded -->

Step 2: Add Lagrangian multipliers for each constraint: μg ≥ 0 , μ -g ≥ 0, α g ≥ 0 , α -g ≥ 0 and get the Lagrange function of formula (i):

<!-- formula-not-decoded -->

Step 3: Substitute f(X) into the formula (ii), we firstly compute the gradient of L(w, b, α, α, ξ, --ξ, μ, -μ -) with respect to w, b, ξ g and ξ -g and make it equal to zero. Then get the following results:

<!-- formula-not-decoded -->

Step 4: Substitute formula (iii) into formula (ii), and get a dual problem:

<!-- formula-not-decoded -->

Fig. 5. Evolutions of (a) original DEI and (b) normalized DEI.

| Physical parameter                   | Value      |
|--------------------------------------|------------|
| Number of balls of the bearing       | 13         |
| Ball diameter of the bearing         | 3.5 mm     |
| Pitch diameter of the bearing epitch | 25.6 mm    |
| Contact angle of the bearing         |            |
| Rotation frequency ( fw), bearing    | 1800 rlmin |
| Rotation frequency (fw), bearing2    | 1600 rhmin |
| Maximum dynamic load (F); bearing1   | 4000 N     |
| Maximum dynamic load (F); bearing2   | 4200 N     |

Photograph

TABLE II PHYSICAL CHARACTERISTICS AND OPERATION CONDITION

<!-- image -->

Step 5: The above process satisfies the Karush–Kuhn–Tucker (KKT) condition, which means at least one of α g and α--g is equal to 0. A sequential minimization algorithm is used to solve α g or α -g . Parameter b is calculated by substituting the α g to get the mean value. The final regression model is described as:

<!-- formula-not-decoded -->

parts—a rotating part, a loading part, and a measurement part. The rotation of the test bearing is driven by the low-speed shaft whose rotating torque is transmitted from an ac motor. A radial force generated by this loading part is applied on the external ring of the testing bearing. Since this external radial force exceeds the bearing's allowable dynamic load (4000 N), the degradation behavior is accelerated so that we can observe its degradation process within a relatively short time in few hours. During experimental tests, two high-frequency accelerometers (Type DYTRAN 3035B) are placed orthogonally on the external race of the test bearing to acquire the horizontal and vertical vibrations, respectively. The accelerometer bandwidth is 0.5– 10 kHz (±5%), and its natural/resonant frequency is 45 kHz. In this article, we extract our degradation labels by using the horizontal vibrations.

Line chart

Fig. 6. Time evolutions of the intermedia feature Mi(finner) , Mi(fouter), and Mi(fball) in (9).

<!-- image -->

Bearing1 is chosen to validate the proposed framework. More specifically, the training set bearing1\_2 is used for extracting the DEI and training the CNN model. Test sets bearing1\_4, bearing1\_5, and bearing1\_6 are then used for estimating their DEIs and predicting the RULs by applying -SVR forecasting model. Results of Bearing2 under different rotational frequency and external dynamic load are also provided and compared. The geometry parameters and the operation conditions of bearing1 and bearing2 are listed in Table II. Sampling frequency of the vibration sensor is 25.6 kHz. 0.1-s accelerometer vibration signals are recorded at a fixed time interval τ = 10 s. Therefore, each recording phase contains p = 2560 measurements. More detailed description of the dataset, bearings, and sensors is presented in the data description in [32].

## B. Degradation Indicator Extraction

The DEI of bearing1\_2 is extracted by substituting its outer ring frequency fouter = 168 Hz, inner ring frequency finner = 221 Hz, and ball frequency fball = 215.4 Hz into (9). The evolution of the final extracted DEI L of bearing1\_2 is shown in Fig. 5(a). We also show the time evolutions of intermediate features Mi Mi (finner) , Mi Mi (fouter), and Mi(fball) of (9) in Fig. 6 . It can be observed that the magnitude of each time point in Fig. 5(a) is the maximum of Mi(finner) , Mi Mi (fouter), and Mi Mi (fball) at that time.

## C. Degradation Indicator Estimation

We use the vibration signal in the horizontal direction of the bearing1\_2 as the input of the CNN, and the normalized DEI is used as the label which contains historical units N = 871. The normalized DEI, L norm , is shown in Fig. 5(b). A less complex architecture of the CNN model is designed to improve the robustness of the network. As shown in Fig. 7, our finalized CNN model consists of K = 6 layers—two convolutional layers (Conv1 and Conv2), two max-pooling layers (Maxpooling1 and Maxpooling2), one fully connected layer (FC1), and one regression layer for output. Before model training, Adam is set to be the optimizer with a small value 0.00001 as it guarantees a quick loss convergence compared with a larger or smaller learning rate for the CNN training process. The activation function in the output layer is the sigmoid function, while the ReLU function is used in the previous layers. Details of parameters in the proposed CNN model are concluded in Table III. The convolutional window sizes (kernel sizes) of convolutional layers are set to a large

Line chart

Fig. 7. Proposed CNN architecture. Normalized DEI L is used as the label for training. A six-layer CNN model is trained to map the raw vibration data S i (input) to the DEI L (output). For a new test bearing, the vibration data Si , test is directly input to the CNN model to obtain the estimated DEI L test.

<!-- image -->

Line chart

Fig. 8. Estimated DEIs as the output of the proposed CNN model: (a) bearing1\_2; (b) bearing1\_4; (c) bearing1\_5; and (d) bearing1\_6.

<!-- image -->

value 100 and a small value 2, respectively. The kernel size of Conv1 is relevantly large in order to extract more features from the raw vibration signal for more impressive power, meanwhile, small kernel size is selected for Conv2 to prevent overfitting. Hyperparameters are obtained after 1000 iterations of training.

The estimated DEIs L test as the output of the CNN model are shown in Fig. 8. In Fig. 8(a), estimated DEI of the training bearing1\_2 shows similar time evolution as the DEI label in Fig. 5(b). The final estimated DEI value of bearing1\_2, Lf t = 0 . 9756, is defined as the FT for the test bearings in Fig. 8(b)–(d) .

## D. RUL Prediction

As presented in Section III-C, the estimated DEIs have been obtained from the trained CNN model. However, the DEIs of the test sets shown in Fig. 8(b)–(d) do not reach their fault limit, which need a regression algorithm to predict the estimated DEI till the end-of-life of each test bearing. An -SVR method is proposed to predict the upcoming degradation process of the test bearings. We conduct a training on the predicted DEI of bearing1\_2. The sampling window size l and the moving size s in Fig. 3 is set to 50 and 1, respectively. The kernel function used in the prediction case is an RBF and penalty parameter C of the error term is chosen as 5.09 by grid search

Fig. 9. (a) Estimated DEI of the bearing1\_2. Predicted RULs of the test bearings are (b) 340 s for bearing1\_4, (c) 1500 s for bearing1\_5, and (d) 1480 s for bearing1\_6. Green dotted lines in (b)–(d) represent the FTs.

| Layer       |   Filters | Kernel  size/Stride   | Output size   |
|-------------|-----------|-----------------------|---------------|
| Input       |           |                       |               |
| Conv ]      |        64 | Ix100/50              | Ix50x64       |
| Maxpooling1 |           | 1x2/2                 | Ix25x64       |
| Conv2       |           | Ix2/1                 | Ix24x64       |
| Maxpooling2 |           | 1x2/2                 | Ix12x64       |
| Flatten     |           |                       | Ix768         |
| FCI         |           |                       | Ix100         |
| Output      |           |                       |               |

Line chart

TABLE III PARAMETERS IN THE CNN MODEL

<!-- image -->

and cross validation using the method of GridSearchCV [33].We estimate the DEI after 1000 steps based on the existing DEI, and using the maximum value of predicted DEI of bearing1\_2 (i.e., Ltf = 0 . 9756) to limit the termination time of the test sets.

-

Fig. 9(b)–(d) shows the predicted RULs T
f T
failure of the test bearings till the failures occur. Red lines represent the predicted evolution of the bearings' degradation behavior using the -SVR method. The RULs are calculated as the difference between the final time when DEI reaches the FT and the time of the last known point of the test bearings. For bearing1\_4, the predicted RUL is 340 s, while 1500 and 1480 s are the predicted RULs for bearing1\_5 and bearing1\_6, respectively.

TABLE IV PROPOSED METHOD AND TESTED METHODS FOR COMPARISON

## E. Comparison and Discussion

To assess the accuracy of the proposed method and compare with other existing approaches, two metrics are commonly adopted: 1) the relative percentage error (Er%) which is given by (15) and 2) the exponential transformed accuracy (ETA) proposed in IEEE PHM 2012 [32]. ETA is an assessment index to distinguish the seriousness of the underestimate and overestimate of RUL prediction. It is clear that underestimate (i.e., early warning) is preferred over overestimation (i.e., warning after damage) to prevent more severe damage to the bearing. The formulas are expressed in (16) in the following:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where Tf Tfailure is the real RUL for the test bearing. A higher | Er% | means a worse RUL prediction result. On the other hand, ETA value varies from 0 to 1, and a higher score means a better RUL prediction result. In this article, Er is the common choice of most previous literature, thus it will be used for comparison.

In addition to the two metrics that evaluate the prediction performance for a specific bearing, three more assessment metrics, namely, average score Sm Smean , mean average error MAE, and the normalized root-mean-square error NRMSE, to make a comprehensive comparison of different methods, which are with the expression forms

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where β is the number of test bearings.

To verify the benefits of the DEI and CNN techniques on RUL prediction, here, we also develop two other tested methods for comparison purpose (see Table IV). The proposed method and the tested methods are denoted and explained as follows.

- 1) C1: CNN and -SVR: This tested method uses a conventional linear time degradation label for CNN training rather than the nonlinear DEI. By this means, we can illustrate the impact of the DEI on RUL prediction.
- 2) C2: DEI and -SVR: Without training the CNN model for feature extraction, DEI in this tested method is extracted

Line chart

Fig. 10. Comparison of the estimated DEIs extracted using HHT (DEI:C2) and calculated by trained CNN model (DEI:Proposed method) of (a) test bearing1\_4, (b) test bearing1\_5, and (c) test bearing1\_6.

<!-- image -->

manually and the -SVR is followed for the RUL prediction. Computing a DEI for a new bearing requires high computational power and longer time. In the meantime, the FT of each test bearing has to be predefined artificially, which increases the uncertainties of the RUL prediction affected by different working conditions. By this means, we can illustrate the impact of CNN modeling on the prediction of the final RUL.

- 3) Proposed method: DEI-based CNN and -SVR: This is the proposed framework which integrates DEI extraction, CNN, and -SVR into one framework.

The predicted numerical errors of the test bearings with the proposed approach and the tested methods are listed in Table V . Our approach achieves Er% of −0.29%, 6.83%, and −1.37% for bearing1\_4, bearing1\_5, and bearing1\_6, respectively, which are much more smaller than the C1 and C2 methods. C1 uses a linear time degradation label for the training of the CNN model. The results show more than 19% prediction errors for test bearings and even 91.15% prediction error is obtained from bearing1\_4, indicating that time degradation label is less effective than the DEI for the CNN training process. C2 is the method extracting the degradation indicator of the test bearings and defines the FTs manually. As testing bearing1\_4, 1\_5, and 1\_6 operate under the same working conditions as bearing1\_2, we employ the maximum and minimum value of bearing1\_2 to normalize the extracted DEIs of C2 method. With the same working condition and same normalization parameters, FT of the proposed method could be reasonably used in C2 as well. To evaluate the impact of CNN modeling on the estimation of the final RUL, for the test bearings, the estimated DEIs extracted using HHT and calculated by trained CNN model are compared in Fig. 10. Without the CNN modeling procedure, one of the main drawback of the C2 method is that it requires a long time to calculate (∼2 s of each sampling period). This limits the practical application of this method in the industry. Moreover, Fig. 10(a) shows that due to the uncertainties and huge mount of noise, the DEI extracted by C2 method has already exceeded the FT before the exact failure time, resulting in a 100% E r . Similarly, in Fig. 10(c), the extracted DEI of C2 method is much noised

TABLE V COMPARISON OF PREDICTED RESULTS BETWEEN PROPOSED APPROACH AND OTHER TESTED METHODS

TABLE VI COMPARISON OF PREDICTED RESULTS BETWEEN PROPOSED APPROACH AND EXISTING MAJOR APPROACHES

than that of the proposed method. At 16 470 s, the magnitude of DEI:C2 is almost close to the FT, this phenomenon might lead to a waste of sources due to excessive underestimation of RUL. The comparison results in Table V demonstrate the benefits of using DEI and CNN in estimating RULs of REBs.

To demonstrate the generality of our proposed RUL estimation framework, other test bearings (i.e, bearing2\_4 and bear-ing2\_6) that operate undergoing a different external load and rotational speed are analyzed. Bearing2\_2 is used for training, and the trained CNN model is obtained without changing any hyperparameters and the architecture of the model in Table III . We just fine-tune the -SVR forecasting model by changing the penalty parameter C from 5.09 to 7.09, resulting in 5.75% and 1.55% E r for bearing2\_4 and bearing2\_6, respectively. Good RUL prediction results of bearings with different operating conditions indicate the repeatability and robustness of our proposed method, with respect to the hyperparameters and the architecture of the CNN.

To further verify the proposed approach, the predicted numerical errors of RULs generated by the proposed method and eight published methods are compared and listed in Table VI . Other published approaches include a recurrent neural network method-based health indicator [35], the method proposed by the winner of the IEEE PHM 2012 prognostic [37], a convolutional long-short-term memory network (LSTM) method [38], and a self-organization mapping (SOM) method, etc.. In addition, recent CNN-based approaches, including the frozen convolution and activated memory network (FCAMN) [23], multiscale CNN [24], and continuous wavelet transform CNN (CWT-CNN) [34] are also compared. The results of the comparison shown in Table VI confirm that our approach significantly outperforms the referenced methods with the highest average score Sm Smean = 0 . 87, and the smallest MAE = 46.2 and NRMSE = 0.05. In particular, a −0.29% E r of bearing1\_4 is achieved, owing to a 1-s absolute time error. This result benefits from a good nonlinear degradation indicator extracted using the HHT method. In addition, the CNN is a powerful tool for discovering the hidden pattern of the extracted degradation indicator and the underlying bearing system, further increasing the accuracy of the predicted RUL.

It can be concluded from the experimental results that the proposed data-driven RUL estimation approach has much smaller prediction errors, compared with both the tested methods in this article and the other methods published in previous studies.

## IV. CONCLUSION

In this article, a data-driven framework for RUL prediction of REB is presented using the HHT method, a CNN model, and an -SVR forecasting model. A nonlinear degradation indicator DEI is first extracted from the raw vibration signals using the HHT method, which is defined as the label for the training. A CNN model is trained to discover the hidden pattern between the extracted DEI and the raw vibration data of the training bearing. In this way, predicted DEIs are automatically obtained when applying the trained CNN model to the test bearings. Finally, the RULs of the testing bearings are obtained using an -SVR forecasting model. An experimentation platform that allows to observe the accelerated degradation process of bearings is employed to validate the proposed framework. The proposed framework achieves much smaller prediction errors for RUL predictions than previous published approaches.

Future work includes the application of the proposed framework to a wider range of case studies on experimental data in other applications [40] and the investigation of other potential degradation labels to achieve even higher accuracy in estimating RUL.

## REFERENCES

- [1] T. A. Harris, Rolling Bearing Analysis. Hoboken, NJ, USA: Wiley, 2001.
- [2] Y. Yuan, H.-T. Zhang, Y. Wu, T. Zhu, and H. Ding, "Bayesian learningbased model-predictive vibration control for thin-walled workpiece machining processes," IEEE/ASME Trans. Mechatronics, vol. 22, no. 1, pp. 509–520, Feb. 2017.

- [3] A. Heng, S. Zhang, A. C. Tan, and J. Mathew, "Rotating machinery prognostics: State of the art, challenges and opportunities," Mech. Syst. Signal Process., vol. 23, no. 3, pp. 724–739, 2009.
- [4] L. R. Rodrigues, "Remaining useful life prediction for multiple-component systems based on a system-level performance indicator," IEEE/ASME Trans. Mechatronics, vol. 23, no. 1, pp. 141–150, Feb. 2018.
- [5] I. V. de Bessa, R. M. Palhares, M. F. S. V. D'Angelo, and J. E. C. Filho, "Data-driven fault detection and isolation scheme for a wind turbine benchmark," Renewable Energy, vol. 87, pp. 634–645, 2016.
- [6] Z.-Q. Wang, C.-H. Hu, and H.-D. Fan, "Real-time remaining useful life prediction for a nonlinear degrading system in service: Application to bearing data," IEEE/ASME Trans. Mechatronics, vol. 23, no. 1, pp. 211– 222, Feb. 2018.
- [7] Y. Li, T. Kurfess, and S. Liang, "Stochastic prognostics for rolling element bearings," Mech. Syst. Signal Process., vol. 14, no. 5, pp. 747–762, 2000.
- [8] X.-S. Si, W. Wang, C.-H. Hu, and D.-H. Zhou, "Remaining useful life estimation—A review on the statistical data driven approaches," Eur. J. Oper. Res., vol. 213, no. 1, pp. 1–14, 2011.
- [9] J. Wang, P. Fu, L. Zhang, R. X. Gao, and R. Zhao, "Multi-level information fusion for induction motor fault diagnosis," IEEE/ASME Trans. Mechatronics, vol. 24, no. 5, pp. 2139–2150, Oct. 2019.
- [10] H. Gao, L. Liang, X. Chen, and G. Xu, "Feature extraction and recognition for rolling element bearing fault utilizing short-time fourier transform and non-negative matrix factorization," Chin. J. Mech. Eng., vol. 28, no. 1, pp. 96–105, 2015.
- [11] R. Yan, R. X. Gao, and X. Chen, "Wavelets for fault diagnosis of rotary machines: A review with applications," Signal Process., vol. 96, pp. 1–15, 2014.
- [12] Q. Meng and L. Qu, "Rotating machinery fault diagnosis using Wigner distribution," Mech. Syst. Signal Process., vol. 5, no. 3, pp. 155–166, 1991.
- [13] A. Soualhi, K. Medjaher, and N. Zerhouni, "Bearing health monitoring based on Hilbert–Huang transform, support vector machine, and regression," IEEE Trans. Instrum. Meas., vol. 64, no. 1, pp. 52–62, Jan. 2015.
- [14] J. Wu, C. Wu, S. Cao, S. W. Or, C. Deng, and X. Shao, "Degradation datadriven time-to-failure prognostics approach for rolling element bearings in electrical machines," IEEE Trans. Ind. Electron., vol. 66, no. 1, pp. 529– 539, Jan. 2019.
- [15] G. Cheng, Y.-L. Cheng, L.-H. Shen, J.-B. Qiu, and S. Zhang, "Gear fault identification based on Hilbert–Huang transform and SOM neural network," Measurement, vol. 46, no. 3, pp. 1137–1146, 2013.
- [16] Z. Tian, "An artificial neural network method for remaining useful life prediction of equipment subject to condition monitoring," J. Intell. Manuf. , vol. 23, no. 2, pp. 227–237, 2012.
- [17] W. Wang, "An adaptive predictor for dynamic system forecasting," Mech. Syst. Signal Process., vol. 21, no. 2, pp. 809–823, 2007.
- [18] J. Sikorska, M. Hodkiewicz, and L. Ma, "Prognostic modelling options for remaining useful life estimation by industry," Mech. Syst. Signal Process. , vol. 25, no. 5, pp. 1803–1836, 2011.
- [19] G. Hinton et al., "Deep neural networks for acoustic modeling in speech recognition: The shared views of four research groups," IEEE Signal Process. Mag., vol. 29, no. 6, pp. 82–97, Nov. 2012.
- [20] M. Sun, I. Konstantelos, and G. Strbac, "A deep learning-based feature extraction framework for system security assessment," IEEE Trans. Smart Grid, vol. 10, no. 5, pp. 5007–5020, Sep. 2019.
- [21] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, vol. 521, no. 7553, pp. 436–444, 2015.
- [22] Y. Yuan et al., "A general end-to-end diagnosis framework for manufacturing systems," Nat. Sci. Rev., Nov. 2019, doi: 10.1093/nsr/nwz190 .
- [23] Z. Chen, X. Tu, Y. Hu, and F. Li, "Real-time bearing remaining useful life estimation based on the frozen convolutional and activated memory neural network," IEEE Access, vol. 7, pp. 96583–96593, 2019.
- [24] J. Zhu, N. Chen, and W. Peng, "Estimation of bearing remaining useful life based on multiscale convolutional neural network," IEEE Trans. Ind. Electron., vol. 66, no. 4, pp. 3208–3216, Apr. 2019.
- [25] L. Ren, Y. Sun, H. Wang, and L. Zhang, "Prediction of bearing remaining useful life with deep convolution neural network," IEEE Access, vol. 6, pp. 13041–13049, 2018.
- [26] M. Sun, T. Zhang, Y. Wang, G. Strbac, and C. Kang, "Using Bayesian deep learning to capture uncertainty for residential net load forecasting," IEEE Trans. Power Syst., vol. 35, no. 1, pp. 188–201, Jan. 2020.
- [27] Y. Lei, N. Li, L. Guo, N. Li, T. Yan, and J. Lin, "Machinery health prognostics: A systematic review from data acquisition to RUL prediction," Mech. Syst. Signal Process., vol. 104, pp. 799–834, 2018.
- [28] N. Tandon and A. Choudhury, "A review of vibration and acoustic measurement methods for the detection of defects in rolling element bearings," Tribology Int., vol. 32, no. 8, pp. 469–480, 1999.
- [29] B. Neyshabur, S. Bhojanapalli, D. McAllester, and N. Srebro, "Exploring generalization in deep learning," in Proc. 31st Conf. Neural Inf. Process. Syst., 2017, pp. 5947–5956.
- [30] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," CORR, vol. abs/1412.6980, 2014.
- [31] T. Benkedjouh, K. Medjaher, N. Zerhouni, and S. Rechak, "Remaining useful life estimation based on nonlinear feature reduction and support vector regression," Eng. Appl. Artif. Intell., vol. 26, no. 7, pp. 1751–1760, 2013.
- [32] P. Nectoux et al., "PRONOSTIA: An experimental platform for bearings accelerated degradation tests," in Proc. IEEE Int. Conf. Prognostics Health Manage., 2012, pp. 1–8.
- [33] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," J. Mach. Learn. Res., vol. 12, pp. 2825–2830, 2011.
- [34] Y. Yoo and J.-G. Baek, "A novel image feature for the remaining useful lifetime prediction of bearings based on continuous wavelet transform and convolutional neural network," Appl. Sci., vol. 8, no. 7, 2018, Art. no. 1102.
- [35] L. Guo, N. Li, F. Jia, Y. Lei, and J. Lin, "A recurrent neural network based health indicator for remaining useful life prediction of bearings," Neurocomputing, vol. 240, pp. 98–109, 2017.
- [36] Y. Lei, N. Li, S. Gontarz, J. Lin, S. Radkowski, and J. Dybala, "A modelbased method for remaining useful life prediction of machinery," IEEE Trans. Rel., vol. 65, no. 3, pp. 1314–1326, Sep. 2016.
- [37] E. Sutrisno, H. Oh, A. S. S. Vasan, and M. Pecht, "Estimation of remaining useful life of ball bearings using data driven methodologies," in Proc. IEEE IEEE Conf. Prognostics Health Manage., 2012, pp. 1–7.
- [38] A. Z. Hinchi and M. Tkiouat, "Rolling element bearing remaining useful life estimation based on a convolutional long-short-term memory network," Procedia Comput. Sci., vol. 127, pp. 123–132, 2018.
- [39] S. Hong, Z. Zhou, E. Zio, and K. Hong, "Condition assessment for the performance degradation of bearing based on a combinatorial feature extraction method," Digit. Signal Process., vol. 27, pp. 159–166, 2014.
- [40] Y. Yuan et al., "Data driven discovery of cyber physical systems," Nature Commun., vol. 10, no. 1, pp. 1–9, 2019.

Photograph

<!-- image -->

Cheng Cheng received the B.Eng. degree in measurement, control technology, and instrument from Tianjin University, Tianjin, China, in 2012 and the M.Sc. and Ph.D. degrees in control systems from Imperial College London, London, U.K., in 2013 and 2018, respectively.

Since 2018, she has been a Postdoctoral Researcher with the Huazhong University of Science and Technology, Wuhan, China. Her research interests include robust control, mechatronic systems modeling and simulation, and deep learning applications.

Photograph

<!-- image -->

Guijun Ma received the bachelor's degree, in 2017, from the Huazhong University of Science and Technology, Wuhan, China, where he is currently working toward the Ph.D. degree in mechanical engineering with the School of Mechanical Science and Engineering.

His research interests include machine fault diagnosis and remaining useful life prediction.

Photograph

<!-- image -->

Yong Zhang received the B.Sc. degree in mathematics from Jiangsu Normal University, Xuzhou, China, in 2001, the M.Sc. degree in applied mathematics from Three Gorges University, Yichang, China, in 2007, and the Ph.D. degree in control theory and control engineering from the Huazhong University of Science and Technology, Wuhan, China, in 2010.

From 2014 to 2015, he was a Visiting Scholar with the Department of Information Systems and Computing, Brunel University London,

Uxbridge, U.K. He is currently an Associate Professor with the Department of Automation, School of Information Science and Engineering, Wuhan University of Science and Technology, Wuhan, China. He has authored more than ten papers in refereed international journals. He is a very active reviewer for many international journals. His current research interests include remaining useful life prediction of key equipment and fault diagnosis and fault tolerant control of networked systems.

Photograph

<!-- image -->

Mingyang Sun (Member, IEEE) received the Ph.D. degree in control science and engineering from the Department of Electrical and Electronic Engineering, Imperial College London, U.K., in 2017.

From 2017 to 2019, he was a Research Associate and a DSI Affiliate Fellow with Imperial College London. He is currently a Professor (Tenure-Track) under the Hundred Talents Program with the College of Control Science and Engineering, Zhejiang University, Hangzhou,

China. He is also a Visiting Researcher with Imperial College London. His research interests include artificial intelligence in energy systems and cyber-physical energy system security and control.

Photograph

<!-- image -->

Fei Teng (Member, IEEE) received the B.Eng. degree in electrical engineering from Beihang University, Beijing, China, in 2009, and the Ph.D. degree in electrical engineering from Imperial College London, London, U.K., in 2015.

He is currently a Lecturer with the Department of Electrical and Electronic Engineering, Imperial College London. His research interests include power system operation, cyber-physical system modeling, and data analytics.

Photograph

<!-- image -->

Han Ding (Senior Member, IEEE) received the Ph.D. degree in mechanical engineering from the Huazhong University of Science and Technology (HUST), Wuhan, China, in 1989.

From 1993 to 1994, he was with the University of Stuttgart, Stuttgart, Germany, supported by the Alexander von Humboldt Foundation. Since 1997, he has been a Professor with HUST, where he is currently the Director of the State Key Lab of Digital Manufacturing Equipment and Technology. From 2001 to 2006, he was a

Cheung Kong Chair Professor with Shanghai Jiao Tong University, Shanghai, China. His research interests include robotics, multiaxis machining, and control engineering.

Dr. Ding was an Associate Editor for the IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING from 2004 to 2007. He is an Editor for the IEEE TRANSACTIONS ON AUTOMATION SCIENCE AND ENGINEERING and a Senior Editor for the IEEE ROBOTICS AND AUTOMATION LETTERS . He was elected as a Member of the Chinese Academy of Sciences, in 2013.

Photograph

<!-- image -->

Ye Yuan (Member, IEEE) received the B.Eng. degree (Valedictorian) from the Department of Automation, Shanghai Jiao Tong University, Shanghai, China, in 2008, and the M.Phil. and Ph.D. degrees from the Department of Engineering, University of Cambridge, Cambridge, U.K., in 2009 and 2012, respectively, all in control theory.

Since 2016, he has been a Full Professor with the School of Artificial Intelligence and Automation, Huazhong University of Science and Tech- nology, Wuhan, China. Prior to this, he was a Postdoctoral Researcher with the University of California Berkeley, Berkeley, CA, USA and a Junior Research Fellow with Darwin College, University of Cambridge. His research interests include system identification and control with applications to cyber-physical systems.

Dr. Yuan was the recipient of the China National Recruitment Program of 1000 Talented Young Scholars, the Dorothy Hodgkin Postgraduate Awards, Microsoft Research Ph.D. Scholarship, and the Best of the Best Paper Award at the IEEE Power and Energy Society General Meeting.