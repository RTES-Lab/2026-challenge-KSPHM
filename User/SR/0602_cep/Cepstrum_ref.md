Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.

Digital Object Identifier 10.1109/ACCESS.2024.0429000

## A Physics-Informed Single-Source Domain Generalization Framework for Bearing Fault Diagnosis under Unseen Operating Conditions

## SUHYUN KIM 1 and TAEHYOUN KIM1, (Member, IEEE)

1 Department of Mechanical and Information Engineering/Smart Cities, University of Seoul, Seoul 02504, Korea

Corresponding author: Taehyoun Kim (e-mail: thkim@uos.ac.kr).

This work was supported by the National Research Foundation of Korea(NRF) grant funded by the Korea government(MSIT) (RS-2026-25468884).

ABSTRACT Unexpected breakdowns in rotating machinery can result in substantial economic losses and safety hazards, underscoring the critical need for reliable bearing fault diagnosis under variable operating conditions. Although deep learning has demonstrated a strong diagnostic capability, its practical deployment is often hindered by domain shifts caused by variations in speed or load, together with the difficulty of obtaining labeled data for all possible operating conditions. To address this challenge, a physics-informed single-source domain generalization (PI-SSDG) framework is proposed that learns from vibration signals collected from a single-source domain corresponding to a specific operating condition and generalizes effectively to unseen conditions without requiring target-domain data. The proposed method introduces a log-mean-removed, low-pass liftered cepstrum that explicitly suppresses global amplitude scaling induced by operating conditions and excitation-related periodicities while preserving fault-discriminative transferfunction-related features. In addition, a dual-branch architecture is designed that jointly exploits the preprocessed cepstrum and raw vibration signals, together with a joint training strategy using branch-specific losses that promotes complementary feature representations and mitigates residual domain variability. Extensive experiments conducted on three public bearing datasets demonstrate that the proposed method consistently achieves superior cross-domain accuracy under speed and load variations and outperforms recent domaingeneralization baselines. Comprehensive ablation studies and feature visualization further confirm the effectiveness of the proposed components in improving diagnostic robustness.

INDEX TERMS Bearing fault diagnosis, cepstrum-based preprocessing, dual-branch architecture, physicsinformed fault diagnosis, single-source domain generalization.

## I. INTRODUCTION

Rotating machinery is widely used in modern industries, including power plants, manufacturing facilities, and transportation. Its reliable operation must be ensured for productivity and operational safety. Rolling bearings are critical components of rotating machinery because they support loads and reduce friction between the shaft and housing. Bearing faults can lead to unplanned downtime, long production interruptions, and safety-critical accidents. Therefore, the early fault diagnosis of bearings and rotating machinery is essential for their safe and efficient operation.

Signal-based monitoring is crucial for early fault diagnosis under practical operating conditions because it enables the continuous observation of machine health. In particular,

vibration signals are widely used in bearing fault diagnosis, because they capture periodic impulses and characteristic energy redistribution induced by defects [1]. Early studies mainly employed frequency-domain analysis [2], [3] to identify fault types by extracting and interpreting characteristic fault-frequency components determined by defect geometry and rotational speed. However, these methods often require substantial expert involvement in feature design and interpretation, which limits their scalability in industrial settings.

Recently, deep-learning-based fault diagnosis methods using convolutional neural networks (CNNs) [4], [5] and recurrent neural networks [6] have become dominant. These methods automatically learn high-level representations from raw vibration signals and perform fault classification with less

Logo

<!-- image -->

Logo

<!-- image -->

reliance on handcrafted features, thereby enabling scalable and automated fault diagnosis.

Despite the considerable success of deep learning models, their deployment in real-world industrial environments remains challenging due to two fundamental challenges. The first challenge is domain shift. In this context, a domain refers to a data distribution induced by a specific set of operating conditions, and domain shift refers to changes in this distribution caused by variations in rotational speed, load, and temperature [7]. For instance, speed variations shift fault-related spectral components and their sidebands, whereas load variations alter vibration amplitudes and signalto-noise ratios [8]. Consequently, discrepancies arise between the source domain, where the model is trained, and the target domain, where the model is deployed. These discrepancies considerably degrade diagnostic performance. Moreover, under realistic operating conditions, domain factors generally vary conjointly rather than independently, and their interactions can produce more complex waveform changes. To address these issues, experimental scenarios involving varying rotational speeds and loads, which constitute major sources of variability in industrial operations, were designed and analyzed herein.

The second challenge is the limited availability of labeled fault data. In practical industrial environments, data corresponding to normal operating conditions are continuously generated, whereas fault occurrences are infrequent and difficult to replicate in a controlled manner [9]. Obtaining reliable labels often requires the physical disassembly of machinery or assessment by domain experts, both of which substantially increase annotation costs. Thus, most collected data are either unlabeled or exhibit class imbalance, with a strong predominance of samples representing normal conditions.

Existing efforts to address these challenges can be broadly categorized into adaptation-based and generalizationbased approaches. Domain adaptation (DA), a representative adaptation-based approach, can directly reduce the performance gap between source and target domains by leveraging data from both domains [10]. However, in industrial practice, target-domain data are often scarce and difficult to label, which limits the applicability of DA. Generalization-based approaches include physics-informed preprocessing (PIP), multi-source domain generalization (MSDG), and physicsinformed domain generalization (PIDG). Although preprocessing can improve robustness to domain shifts, it is often insufficient to ensure reliable performance under unseen operating conditions [11], [12]. In addition, most domaingeneralization methods assume the availability of multiple source domains, which are expensive and time-consuming to construct in practice [13], [14]. Furthermore, during model development, data are commonly collected only under a single operating condition. This limitation has motivated the use of single-source domain generalization (SSDG), which can maintain performance under unseen conditions using only a single-source domain [15].

To address these challenges, a physics-informed single- source domain generalization (PI-SSDG) method is proposed herein for bearing fault diagnosis that is robust to speed and load variations and reflects practical industrial constraints. It comprises an LL-cepstrum-based preprocessing scheme, a dual-branch model with cepstrum and time-domain pathways, and a joint training strategy based on branch-specific losses. The main contributions of this study are summarized as follows:

- 1) We propose a log-mean-removed, low-pass liftered (LL) cepstrum-based preprocessing method tailored for cross-domain bearing fault-type classification. The LLcepstrum suppresses two components that are sensitive to operating conditions, namely the zero-quefrency global offset and high-quefrency excitation periodicities, while preserving the low-quefrency band associated with the structural transfer function. This approach differs from prior cepstrum-based techniques such as cepstrum pre-whitening for envelope analysis and exponential liftering for modal extraction, which primarily focus on signal enhancement rather than on domainrobust feature representation.
- 2) Unlike prior PIDG methods that embed physicsmotivated layers within the network, the proposed approach pairs a shallow linear backbone for the cepstrum branch with rectified linear unit (ReLU) gating at the time-branch output to suppress domain-specific fluctuations. This design ensures that domain-robust spectral characteristics are not considerably distorted by deep nonlinear transformations and establishes the cepstrum branch as a stable anchor for single-source domain generalization.
- 3) We introduce a joint training scheme that optimizes branch-specific and fusion losses. This balanced optimization complements the cepstrum-anchored dualbranch architecture by reducing over-reliance on a single branch and enhancing generalization stability. Thus, the proposed framework achieves domain-robust generalization without requiring target-domain data, multiple source domains, or pseudo-domain synthesis, which are typically required in recent PI-SSDG methods.
- 4) These components are integrated into a PI-SSDG framework evaluated on three public bearing datasets under realistic single-source–multiple-target settings involving speed shifts, load shifts, and their combined variations. In addition to quantitative comparisons with recent domain-generalization methods, ablation studies and feature visualization analyses are conducted to quantify and interpret the contribution of each design choice.

The remainder of this paper is organized as follows. In Section II, approaches for handling domain shifts in machine fault diagnosis are reviewed and their limitations are discussed. Section III details the proposed preprocessing method, model architecture, and learning strategy. In Section IV, the datasets and experimental setup are described and the cross-domain performance is discussed. Section V

presents the ablation studies and feature analysis. Finally, Section VI concludes the paper and discusses future research directions.

## II. RELATED WORK

## A. ADAPTATION-BASED APPROACHES

Adaptation-based approaches address domain shifts after deployment by leveraging data collected from the target domain. A representative paradigm is DA, which can be categorized into supervised and unsupervised DA depending on the availability of target-domain labels. In machine fault diagnosis, unsupervised DA has been widely studied due to its labeling constraints. Unsupervised DA typically reduces discrepancies between source and target distributions by minimizing statistical measures, such as maximum mean discrepancy (MMD) [16] and correlation alignment (CORAL) [17], or by adopting adversarial learning to extract domain-invariant features [18]. Beyond these alignment objectives, recent studies have integrated physical knowledge into DA frameworks to improve alignment precision. Examples include the use of subsegment-guided impulse signatures [19] and adaptive filtering kernels based on frequency responses [20].

However, in industrial environments, target-domain faultcondition data are often scarce, and security policies and network isolation can prevent access to source-domain data after deployment. These constraints limit the practicality of adaptation-based approaches and motivate the development of generalization-based approaches that aim to achieve robustness using only source-domain data.

## B. GENERALIZATION-BASED APPROACHES

Generalization-based approaches aim to improve robustness to domain shifts without relying on target-domain data by learning exclusively from source-domain data. These approaches address domain shifts at different abstraction levels. PIP operates at the input-representation level by transforming signals before learning. In contrast, MSDG and PIDG operate at the training-paradigm level by structuring the learning process. PIDG further differs from MSDG because it combines physics-based preprocessing or physics-motivated network layers with the training paradigm.

## 1) Physics-Informed Preprocessing (PIP)

PIP mitigates disturbances induced by variations in operating conditions during the preprocessing stage by exploiting the prior knowledge of machine dynamics and operating mechanisms. Among these techniques, order tracking resamples time-domain signals into the angular (order) domain using a tachometer signal or an estimated rotational speed. Consequently, fault-related periodic components remain aligned even under speed variations [21], [22].

In contrast, cepstrum-based preprocessing reduces signal variability by selectively filtering or normalizing periodic components in the log-spectrum domain. Kim et al. [23] combined cepstrum analysis with a minimum-phase filter to map the signals collected under different measurement conditions

Logo

<!-- image -->

into a common pattern space. Wang et al. [24] constructed speed-invariant features under a variable-speed operation by applying mean filtering and threshold-based peak selection to the real cepstrum. In addition, signal-processing techniques such as the fast Fourier transform (FFT), short-time Fourier transform (STFT), and wavelet packet transform (WPT) are often used in combination with these approaches to enhance robustness against variations in operating conditions [25]– [29].

PIP approaches can reduce the effects of domain shifts at the input level without requiring target-domain data and provide physical interpretability. However, preprocessing alone does not necessarily guarantee strong generalization and can increase computational cost. In particular, order tracking may require additional sensors, whereas tachometer-free speed estimation can accumulate errors that degrade the performance.

Beyond fault detection, other cepstrum-based techniques focus on signal enhancement. In particular, cepstrum prewhitening [30] suppresses deterministic harmonic and sideband components by setting the real cepstrum to zero at all quefrencies except zero quefrency and then reconstructing the signal. This process exposes impulsive bearing-fault residuals for subsequent envelope analysis. Similarly, exponential liftering [31] applies an exponential window in the cepstrum domain to emphasize modal contributions for operational modal analysis. Both techniques are designed for signal enhancement rather than for constructing domain-robust feature representations, and their lifter designs are not explicitly intended to suppress variations in operating conditions.

## 2) Multi-Source Domain Generalization (MSDG)

MSDG learns domain-invariant representations from multiple source domains to maintain performance in unseen target domains. Representative methods include data augmentation [32], domain alignment [33], [34], and feature disentanglement [35], [36]. Data augmentation expands the source distribution via diverse transformations, exposing the model to a broad range of domains during training. For instance, Shao et al. [32] reported that augmenting limited thermal-image data improved the generalization performance of rotor–bearing multi-class classification. Domain alignment [33], [34] improves robustness to changes in operating conditions by mapping multiple source domains into a shared feature space using adversarial learning or discrepancybased measures. Feature-disentanglement separates domainspecific and domain-invariant components. Wang et al. [35] modeled domain-specific features using domain-wise classifiers and employed an invariant extractor to suppress them. In contrast, Zhao et al. [36] combined CORAL with triplet loss and proposed a prediction-similarity-based weighted fusion strategy. In a related extension, Huang et al. [37] integrated fault decoupling with adversarial training to address compound faults under varying operating conditions.

Although MSDG methods can mitigate performance degradation without using target-domain data, they are vulnerable to unseen domain shifts and may not fully exploit

Logo

<!-- image -->

the physical structure of measured signals. More importantly, these methods typically require access to multiple wellbalanced source domains, which limits their applicability in industrial settings where operating conditions and data collection are constrained.

## 3) Physics-Informed Domain Generalization (PIDG)

PIP mainly embeds physical knowledge at the preprocessing stage, whereas PIDG integrates such knowledge into both preprocessing and model design to mitigate performance degradation caused by domain shifts without requiring target-domain data. A common approach to achieve this involves compensating for predictable disturbances induced by changes in operating conditions via physics-based preprocessing and then reducing the remaining uncertainty via domain-generalization learning.

From an implementation perspective, PIDG can be realized via sequential coupling and embedded coupling. Sequential coupling applies domain-generalization learning after physics-based preprocessing, whereas embedded coupling embeds physics-motivated operations into network layers. For example, Zheng et al. [38] combined angle resampling with the Hilbert transform to obtain a consistent angledomain representation and used an instance-based discriminative loss to promote stable generalization. Similarly, Xie et al. [39] coupled a phase-centered representation with knowledge distillation to transfer invariant characteristics. Ni et al. [14] further integrated computed order tracking and cepstrum-filter-based physics layers into the network, enabling the direct learning of robust representations during training.

Despite their enhanced robustness and interpretability, PIDG methods depend on the validity of their underlying physical assumptions. Many existing PIDG methods rely on multiple source-domain settings, which limits their applicability in industrial scenarios where data are typically available from only a single operating condition.

## 4) Physics-Informed Single-Source Domain

## Generalization (PI-SSDG)

Recent studies have explored SSDG for machine fault diagnosis to achieve robustness under unseen operating conditions using only a single-source dataset [40]. For example, Huang et al. [41] expanded known samples via manifold mixup and established compact decision boundaries via classwise adversarial training to handle unpredictable category shifts in a single-source setting. However, the limited diversity of observable conditions in single-source settings makes it difficult to identify domain-robust features, and systematic SSDG methodologies remain limited. To mitigate these limitations, recent PI-SSDG methods typically synthesize pseudo-domain shifts for domain-invariant learning. Wu et al. [42] generated pseudo-target domains by randomly scaling spectral components because certain peak locations remained relatively stable under speed variations; they then combined this approach with unsupervised learning. Tang et al. [43]

investigated variations in amplitude distribution caused by speed variations and addressed them by integrating histogram matching with mixup augmentation. They also introduced a learnable linear dimension-boosting technique to encourage invariant feature learning.

Although these approaches extend the feasibility of SSDG in data-scarce conditions, they tend to rely on observationdriven augmentation rather than physics-based structural modeling. In particular, generalization strategies that can jointly handle compound domain shifts, in which speed and load vary simultaneously, have not been sufficiently explored in single-source settings. The proposed method addresses this gap by integrating three coordinated elements: domain-shift suppression during preprocessing, an anchored dual-branch architecture, and a joint training strategy with branch-specific and fusion losses. The integration of these components enables domain-robust generalization without requiring targetdomain data, multiple source domains, or pseudo-domain synthesis. Unlike prior approaches that assume multi-domain data availability, additional sensors, or post-deployment data access, the proposed method requires only lightweight preprocessing operations and a compact model structure. This design is aligned with practical industrial environments, where data access is often restricted by security policies and network isolation.

## III. METHODOLOGY

## A. PROBLEM FORMULATION

Herein, a domain refers to a data distribution induced by a specific operating-condition setting. All domains share the same bearing fault-type classification task and label space Y , but their input distributions differ across operating conditions. Each sample in a domain comprises an acceleration-based vibration segment x ∈ R L of length L and a corresponding bearing fault class label y ∈ Y. The operating conditions are represented by o = (rpm , load), where rpm and load denote rotational speed and load, respectively. Let the source and target domains be D s and D t , with joint distributions p s (x , y) and p t (x , y), respectively, induced by the source and target operating conditions o s and o t , respectively. The domain shift is characterized by p s (x , y) ̸= p t (x , y), where conditional shifts of the form p(x | y , o s ) ̸= p(x | y , ot ) arise due to variations in operating conditions. Consequently, a decision boundary learned from D s may lose optimality when applied to D t
.

MSDG methods address this issue by learning domaininvariant features from multiple source domains. In contrast, an SSDG setting is considered herein, in which the model has access to labeled data from only one source domain and no target-domain samples are available during training. Let fθ denote the diagnosis model with learnable parameters θ trained exclusively on a single-source domain D s = {(xi , yi)} N i=1 with N labeled samples. During deployment, the model may encounter target domains D ˜ t = {D (j) t }
j J }
j=1 induced by various unseen operating conditions. The objective is to learn a model fθ that consistently achieves high diagnostic performance

Logo

<!-- image -->

IEEE Access

Flow chart

FIGURE 1. Architecture of the proposed framework.

<!-- image -->

under these unseen operating conditions. This objective can be expressed as minimizing the expected risk over unseen target domains:

<!-- formula-not-decoded -->

where p D ˜ t denotes the unknown mixture distribution induced by D ˜ t , and ℓ(· , · ) denotes the cross-entropy loss for multiclass classification. In practice, the expected risk in (1) is approximated by evaluating the trained model using multiple held-out target domains, each of which is defined by a distinct unseen operating-condition tuple. Based on this formulation, a PI-SSDG framework is proposed herein for robust bearing fault-type classification under unseen operating conditions.

## B. PROPOSED FRAMEWORK

As shown in Fig. 1, the proposed framework addresses SSDG by integrating three components: physics-informed cepstrum preprocessing, a dual-branch cepstrum–raw-signal model, and a joint training strategy using branch-specific and fusion losses.

The preprocessing module reformulates the conventional real cepstrum pipeline by incorporating mean removal in the log-magnitude spectrum and low-pass liftering. This design mitigates operating-condition-induced variations at the input level and yields representations that are more robust to domain shifts.

The model adopts a dual-branch architecture in which the preprocessed cepstrum and raw vibration signal are processed through two complementary branches. Each branch extracts complementary features that are subsequently normalized and concatenated to form a fused embedding. A fusion classifier produces the final prediction based on this embedding. In Fig. 1, Conv A×B denotes a convolutional layer with kernel size A and B output channels, and Ncls represents the number of classes in each dataset.

During training, branch-specific classification losses and a fusion loss are jointly optimized, which discourages overreliance on a single branch and encourages complementary feature integration. Consequently, the proposed method achieves improved robustness under unseen target domains even when trained using only a single-source domain.

## C. PHYSICS-INFORMED CEPSTRUM PREPROCESSING

In the preprocessing stage, feature representations that are robust to domain shifts are constructed by employing a physicsinformed cepstrum transform. This design is guided by three motivations. First, the real cepstrum converts the convolutional relationship between the excitation force and structural transfer function in vibration signals into an additive representation in the cepstrum domain. This transformation improves interpretability and enables component-wise reasoning. Second, mean removal in the log-magnitude spectrum mitigates global amplitude scaling induced by variations in operating conditions. Third, low-pass liftering emphasizes transfer-function-related information while suppressing excitation-related periodic structures that are sensitive to speed variations. Note that liftering is the cepstral-domain analog of filtering.

This preprocessing pipeline mitigates speed-dependent periodic shifts and scale variations at the input level without requiring additional sensors or order tracking. In the following section, the real cepstrum and the proposed mean removal and low-pass liftering operations are described.

## 1) Real Cepstrum Definition

Let x[n] (n = 0 , . . . , L − 1) denote a vibration segment of length L, and X[k] = F{x}[k] denote its L-point discrete Fourier transform (DFT). The log-magnitude spectrum S[k] and the real cepstrum c[n] are defined as follows:

<!-- formula-not-decoded -->

Logo

<!-- image -->

where ϵ &gt; 0 denotes a small constant introduced for numerical stability. Although c[n] is indexed on the same discrete grid as x[n], the index is interpreted as quefrency, τ = n fs fs , where fs fs denotes the sampling frequency. The quefrency axis corresponds to the inverse of the frequency spacing associated with periodic structures in the log spectrum. For example, a periodic structure with spacing ∆f in S[k] produces a peak near τ0 = 1 ∆f (i.e., n0 ≈ fs fs ∆f ).

A key property of the cepstrum is that convolution in the time domain becomes an addition in the cepstrum domain. The measured vibration signal can be modeled as the convolution of an excitation component f (e.g., repetitive defect– contact impacts) and a structural impulse response h:

<!-- formula-not-decoded -->

In the frequency domain, this becomes

<!-- formula-not-decoded -->

In terms of magnitude,

<!-- formula-not-decoded -->

Therefore, the log-magnitude spectrum satisfies

<!-- formula-not-decoded -->

Applying the inverse Fourier transform yields

<!-- formula-not-decoded -->

where cf cf [n] is associated with excitation-related periodic structures and c h [n] reflects the cepstral components associated with the structural transfer function, concentrated in the low-quefrency region near n = 0 .

The decomposition in (7) has direct implications for domain shifts induced by variations in operating conditions. Variations in o = (rpm , load) predominantly affect the excitation component f via amplitude scaling and changes in excitation-related periodicity, thereby concentrating their effects in cf cf [n]. In contrast, the structural transfer function h is largely determined by the bearing geometry and mounting rather than by o. Consequently, ch[n] remains robust to variations in operating conditions and provides a comparatively domain-invariant component of c x [n] .

Fig. 2 shows the relationships among the excitation force f , impulse response h, and measured signal x across the time, frequency, and cepstrum domains. In the time domain, the periodic excitation f [n] is convolved with h[n] to produce a decaying vibration response x[n]. In the frequency domain, this relationship becomes multiplicative, as expressed in (4) and (5). Therein, |F[k]| typically exhibits a comb-like structure with an interval ∆f related to rotational periodicity, whereas |H[k]| forms a smooth spectral envelope determined by structural dynamics. In the cepstrum domain, the additive property yields the decomposition expressed in (7).

Consequently, cf[n] produces impulsive peaks at quefrencies corresponding to 1 ∆f and its integer multiples (i.e., harmonics), whereas ch[n] is concentrated in the low-quefrency

Engineering drawing

FIGURE 2. Relationships among the excitation force, structural impulse response, and measured signal across different signal domains. (a) Time domain. (b) Frequency domain. (c) Cepstrum domain.

<!-- image -->

region near n = 0. Based on this property, the preprocessing design is described in subsequent sections; this design mitigates global amplitude scaling and emphasizes transferfunction-related information.

## 2) Log-Mean-Removed, Low-Pass Liftered (LL) Cepstrum

Variations in operating conditions often induce global amplitude scaling in the measured signal, which appears as an additive constant in the log-magnitude spectrum. To mitigate this effect, the mean of the log-magnitude spectrum S ¯ and the mean-removed spectrum S e [k] are defined as follows:

<!-- formula-not-decoded -->

where S[k] = log(|X[k]| + ϵ), with ϵ = 10 − 10 for numerical stability. Notably, the n = 0 coefficient of the real cepstrum is equal to the mean of the log-magnitude spectrum

<!-- formula-not-decoded -->

Accordingly, the cepstrum computed from the mean-removed spectrum ˜ c [n] = F − 1 {S e }[n] satisfies

<!-- formula-not-decoded -->

where δ[n] denotes the discrete unit impulse; thus, ˜ c [0] = 0 always holds. Consequently, mean removal in the logmagnitude spectrum eliminates the DC component at zero quefrency (n = 0) in the cepstrum, where global-scale variations are concentrated. This operation improves robustness to domain shifts.

Because the spectral envelope induced by the transfer function varies smoothly along the frequency axis, its contribution tends to be concentrated in the low-quefrency region of the cepstrum. In contrast, periodic excitation components and

Line chart

FIGURE 3. Comparison of quefrency-domain representations under different rotational speeds on the UOS dataset: conventional real cepstrum (left) and proposed LL-cepstrum (right) for the normal condition and three fault classes.

<!-- image -->

noise are more prominent at higher quefrencies. Therefore, retaining only the low-quefrency components of ˜ c [n] emphasizes structural transfer-function characteristics. Based on this observation, low-pass liftering is applied to ˜ c [n] and the resulting LL-cepstrum is defined as

<!-- formula-not-decoded -->

where κ denotes the low-pass liftering length. Herein, κ = L/8 is used as a fixed default value to simplify the preprocessing pipeline and avoid target domain-dependent tuning. The rationale for this selection is discussed in Section V-A.

The proposed preprocessing can be summarized as follows:

<!-- formula-not-decoded -->

The resulting cLL is fed into the cepstrum branch of the proposed dual-branch model, and its extracted features are fused with those obtained from the time branch.

The effectiveness of the proposed preprocessing method is evaluated using three public bearing vibration datasets: CWRU [44], UOS [45], and PU [46]. These datasets are

Logo

<!-- image -->

IEEE Access described in detail in Section IV-A. Fig. 3 compares the conventional real cepstrum (left) and the proposed LL-cepstrum preprocessing (right) on the UOS dataset under different rotational speeds for normal conditions and three fault types. The LL-cepstrum produces more consistent class-wise patterns across different speeds and improves separability among fault classes compared with the conventional cepstrum. This behavior is primarily attributed to the removal of the n = 0 cepstral component and the retention of the low-quefrency band, which suppress speed-dependent excitation effects.

Similar trends are observed for the CWRU and PU datasets under various operating conditions, including load variations. These results indicate that the proposed preprocessing method produces representations that are less sensitive to variations in operating conditions, thereby improving diagnostic robustness under domain shifts.

## D. DUAL-BRANCH MODEL ARCHITECTURE

As shown in Fig. 1, the proposed model adopts a dual-branch architecture. In this design, the cepstrum branch extracts domain-robust transfer-function characteristics, whereas the time branch captures transient impulses that may be attenuated by cepstrum-based preprocessing. Thus, these two branches provide complementary views of the same physical system.

The cepstrum branch feature extractor, Fc Fcep , applies only dropout operations to the input data without additional normalization. This design choice is motivated by two observations. First, mean removal in the proposed preprocessing method already reduces sample-wise scale variations. Second, applying instance normalization (IN) to cepstral inputs can excessively amplify peak contrast and degrade performance. For the backbone design, a shallow linear transformation is employed instead of deep convolutional layers to preserve the global envelope of cLL, which is designed to emphasize transfer-function-related information. The cepstrum branch produces a 32-dimensional embedding, defined as follows:

<!-- formula-not-decoded -->

The time-branch feature extractor, Ftime, is based on a one-dimensional (1D) CNN architecture designed to capture short-term nonstationary features, such as shock patterns and impulse intervals, which are not explicitly represented in the cepstrum domain. In this manner, the time branch compensates for fine-grained transient information that may be attenuated after cepstrum transformation. At the input stage of the time branch, IN and dropout at a rate p = 0 . 5 are applied to reduce sample-wise scale variations and suppress overfitting. In addition, an IN layer is placed after each convolutional layer to mitigate the accumulation of domain-specific modulations as network depth increases. The time branch outputs a 32-dimensional embedding as follows:

<!-- formula-not-decoded -->

Logo

<!-- image -->

TABLE 1. Architecture of the time branch.

| No.                        | Type  K                        | C S P                      | Output                     |                            |                            |                            |
|----------------------------|--------------------------------|----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| Feature Extractor (Ftime)  | Feature Extractor (Ftime)      | Feature Extractor (Ftime)  | Feature Extractor (Ftime)  | Feature Extractor (Ftime)  | Feature Extractor (Ftime)  | Feature Extractor (Ftime)  |
| 1                          | IN + Dropout(0.5)              | – – – –                    | 1 × L                      |                            |                            |                            |
| 2                          | 1DConv + IN + ReLU             | 15 16 1 7                  | 16 × L                     |                            |                            |                            |
|                            | 3 MaxPool1d 2 – 2 0            |                            | 16 × L/2                   |                            |                            |                            |
| 4                          | 1DConv + IN + ReLU             | 3 32 1 1                   | 32 × L/2                   |                            |                            |                            |
|                            | 5 MaxPool1d 2 – 2 0            |                            | 32 × L/4                   |                            |                            |                            |
| 6                          | 1DConv + IN + ReLU             | 3 64 1 1                   | 64 × L/4                   |                            |                            |                            |
| 7                          | 1DConv + IN                    | 3 32 1 1                   | 32 × L/4                   |                            |                            |                            |
|                            | 8 AdaptiveMaxPool1d(4) – – – – |                            | 32 × 4                     |                            |                            |                            |
| 9                          | Linear + ReLU                  | – 32 – –                   | 32                         |                            |                            |                            |
| Branch-specific Classifier | Branch-specific Classifier     | Branch-specific Classifier | Branch-specific Classifier | Branch-specific Classifier | Branch-specific Classifier | Branch-specific Classifier |
|                            | 10 Dropout(0.2) – – – –        |                            | 32                         |                            |                            |                            |
| 11                         | Linear + ReLU                  | – 32 – –                   | 32                         |                            |                            |                            |
|                            | 12 Linear –                    | Ncls  – –                  | Ncls                       |                            |                            |                            |

1DConv: 1D convolution; IN: Instance normalization;

K: Kernel size; C: Output channels; S: Stride; P: Padding;

Output: Channel×Input data length

TABLE 2. Architecture of the cepstrum branch.

|                            |                               | No. Type C Output          |                            |
|----------------------------|-------------------------------|----------------------------|----------------------------|
| Feature Extractor (F cep ) | Feature Extractor (F cep )    | Feature Extractor (F cep ) | Feature Extractor (F cep ) |
|                            | 1 Dropout(0.5) –  2 Linear 32 | L/8 32                     |                            |
| Branch-specific Classifier | Branch-specific Classifier    | Branch-specific Classifier | Branch-specific Classifier |
|                            | 3 Linear                      | Ncls  Ncls                 |                            |

TABLE 3. Architecture of the fusion classifier.

|                                  | No. Type C Output   |
|----------------------------------|---------------------|
| 1 Dropout(0.5) –  2 Linear  Ncls | 64 Ncls             |

Before fusion, the output of the time branch is passed through a ReLU activation function to suppress noninformative components and prevent domain-specific fluctuations from propagating into the cepstrum embedding.

To minimize scale mismatch between the two embeddings, ℓ 2 -normalization, denoted by norm(·), is applied to each branch output. The normalized embeddings are concatenated to form a fused representation:

<!-- formula-not-decoded -->

This fusion approach prevents either branch from dominating the representation due to scale differences and encourages the classifier to exploit complementary information from both branches. Finally, the fusion classifier maps zfuse to an Nclsdimensional output to predict the fault class. The layer-bylayer specifications of the time branch, cepstrum branch, and fusion classifier are summarized in Tables 1, 2, and 3, respectively.

## E. JOINT TRAINING STRATEGY AND OPTIMIZATION

The proposed model is trained using only source-domain data D s = {(xi , yi)} N i=1 . The training objective jointly minimizes the classification losses associated with the time branch, cepstrum branch, and fusion classifier. Let Ctime , C cep, and Cfuse denote the classifiers corresponding to the time and cepstrum branch and fusion classifier, with parameters {Wt Wt, b t } , {Wc Wc, b c }, and {Wf , bf bf }, respectively. Given ˆ ztime = norm(ztime) , ˆ z cep = norm(z cep ), and zfuse = [ ˆ ztime; ˆ z cep ] , the three classifiers produce softmax predictions:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

For a mini-batch of size B, the time-branch loss ℓtime , cepstrum-branch loss ℓ cep , and fusion loss ℓfuse are defined using the cross-entropy loss LCE:

<!-- formula-not-decoded -->

<!-- formula-not-decoded -->

where yi ∈ {1 , . . . , Ncls} denotes the ground-truth label of the ith sample in the mini-batch. The final training objective is defined as the weighted sum of these three losses.

<!-- formula-not-decoded -->

Herein, the loss weights are set to λtime = λ cep = λ fuse = 1. This choice is supported by the ℓ2-normalization applied to the branch embeddings, which keeps the feature scales of the two branches comparable and stabilizes the joint training without additional weight tuning.

All parameters are optimized end-to-end using the Adam optimizer. The fusion loss backpropagates gradients through both branches, whereas the branch-specific losses ensure that each branch maintains discriminative representations. This joint training scheme reduces over-reliance on a single branch and promotes complementary feature learning, enabling consistent generalization under unseen operating conditions. The effects of the dual-branch design and joint training strategy are further analyzed in Section V-B.

## IV. PERFORMANCE EVALUATION

## A. DATASETS

To evaluate the proposed method and its generalization capability under diverse operating conditions, experiments were conducted using three publicly available datasets: CWRU [44], UOS [45], and PU [46]. These datasets encompass different types of variations in operating conditions, including simultaneous speed–load variations (CWRU), speed variations (UOS), and load variations (PU). In the experiments, a domain was defined by a specific operating condition

TABLE 4. Domain information for the three datasets.

|      | Dataset Operating Condition                                                                   |         |
|------|-----------------------------------------------------------------------------------------------|---------|
|      | Dataset Operating Condition                                                                   | A B C D |
| CWRU | Speed (rpm) 1,797 1,772 1,750 1,720 Load (HP) 0 1 2 3                                         |         |
|      | UOS Speed (rpm) 600 1,000 1,400 –                                                             |         |
| PU   | Speed (rpm) 1,500 1,500 1,500 – Torque (N·m) 0.7 0.1 0.7 – Radial Force (N) 1,000 1,000 400 – |         |

TABLE 5. Information on bearing fault classes in the CWRU dataset.

|                                | Class N OR1 OR2 OR3 IR1 IR2 IR3 B1 B2 B3   |
|--------------------------------|--------------------------------------------|
| Crack size – S M L S M L S M L |                                            |

S: 0.007 inch (small), M: 0.014 inch (medium), L: 0.021 inch (large).

TABLE 6. Information on bearing fault classes in the PU dataset.

|                                             | Class N OR1 OR2 OR3 OR4 OR5 IR1 IR2 IR3 IR4 IR5   |
|---------------------------------------------|---------------------------------------------------|
| Severity – 1 1 2 1 1 1 3 1 2 1              |                                                   |
| Damage – F PD F F PD F F F F F              |                                                   |
| Combination – SD SD RD SD RD CD SD RD SD SD |                                                   |

such as a speed–load pair in CWRU or a speed level in UOS, as summarized in Table 4.

For the CWRU dataset, acceleration signals sampled at 12 kHz were used. Four operating conditions with simultaneous variations in rotational speed and load were defined as Domains A–D. The classification task included 10 classes, comprising the normal condition (N), outer-race faults (OR1– OR3), inner-race faults (IR1–IR3), and ball faults (B1–B3). The fault-size information is summarized in Table 5. The segment length was set to 2,048 samples so that each segment contained information spanning at least one mechanical revolution.

For the UOS dataset, acceleration signals sampled at 16 kHz were used, and the operating conditions at 600, 1,000, and 1,400 rpm were defined as Domains A, B, and C, respectively. The dataset contained four classes: the normal condition, outer-race faults, inner-race faults, and ball faults. All fault defects in the UOS dataset were artificially introduced via grinding, and fault severity levels were not provided. The segment length was set to 2,048 samples.

For the PU dataset, acceleration signals sampled at 64 kHz were used. Three operating conditions with different combinations of torque and radial force were defined as Domains A–C. To better reflect practical operating conditions, only naturally occurring outer-race and inner-race faults were selected from the available fault scenarios. As summarized in Table 6, 11 classes were constructed, namely the normal class (N), outer-race faults (OR1–OR5), and inner-race faults (IR1– IR5). The segment length was set to 4,096 samples. Because the PU dataset combines different torque–force levels with naturally developed damage, it provides a realistic testbed for

Logo

<!-- image -->

IEEE Access

TABLE 7. Preprocessing methods evaluated for comparison in the cepstrum branch.

|                      | Method Description                                                                                                                   |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| Real Cepstrum (CEPS) | Real cepstrum obtained by applying the inverse DFT to the log-magnitude spectrum of the vibration signal (1D).                       |
| FFT                  | Magnitude of the DFT &#124;X[k]&#124; of the raw signal (1D).                                                                        |
| STFT                 | Spectrogram formed by stacking frame-wise FFT magni tudes from short-time windowed segments (2D) (hyperparameters follow [27]).     |
| WPT                  | Scalogram constructed from the magnitudes of wavelet packet coefficients in the time-scale plane (2D) (hyperparameters follow [29]). |

Flow chart

FIGURE 4. Cepstrum-branch backbone configurations used for preprocessing comparison. (a) Scenario 1. (b) Scenario 2.

<!-- image -->

assessing robustness to both operating conditions and fault severity variability.

For all datasets, the raw signals were segmented according to the specified segment lengths, and the overlap ratio was adjusted such that segment counts were comparable across classes. The resulting data were split into training and test sets in a consistent 6:4 ratio for all datasets. To prevent segment-level leakage, each raw recording was first split into training and test portions in a 6:4 ratio; then each portion was segmented independently. These design choices reduced the impact of class imbalance and inconsistent segmentation settings when evaluating the cross-domain generalization performance.

## B. COMPARISON METHODS AND EXPERIMENTAL SETUP

To comprehensively validate the proposed method, two sets of experiments were conducted. First, alternative preprocessing modules were compared within the proposed dual-branch architecture. Second, the proposed framework was evaluated against recent domain-generalization methods.

For comparing preprocessing performance, the overall dual-branch model was fixed and only the input representation and backbone of the cepstrum branch were varied. The candidate preprocessing methods are listed in Table 7. To reduce potential bias arising from the choice of backbone and input representation formats, we considered two scenarios in Fig. 4. In Scenario 1, the cepstrum-branch backbone was

Logo

<!-- image -->

restricted to a single linear layer occupying approximately 0.11 MB to evaluate the discriminative power of the preprocessing method under a small backbone. In Scenario 2, a CNN backbone following the baseline in [10] was used for the cepstrum branch to evaluate preprocessing performance under a commonly adopted feature-extraction setting. Specifically, a 1D CNN was used for 1D representations, whereas twodimensional (2D) CNNs were adopted for time–frequency maps generated by STFT and WPT to preserve their native input formats. This pairing avoids reshaping the input signals, which could otherwise distort their physical meaning. The 1D and 2D CNN backbones were designed with comparable parameter scales of approximately 0.74 and 0.99 MB, respectively, ensuring that performance differences were not driven by backbone capacity. This two-scenario evaluation separated the effects of preprocessing from those of backbone selection.

For a model-level comparison, the proposed framework was evaluated against recent domain-generalization methods, including MSDG, SSDG, and PI-SSDG, as listed in Table 8. All competing methods were reimplemented according to their original settings and evaluated under identical training configurations to ensure fair comparison.

To ensure fair comparability across both preprocessing and model-level comparisons, all experiments shared a common training configuration. As the proposed model applied IN at the input of the time branch, no additional input normalization was applied. For the baseline models and alternative preprocessing inputs, z-score normalization was applied for consistency. Training was performed for 100 epochs using the Adam optimizer with a learning rate of 10 − 3 and a batch size of 64. Kaiming initialization was used for convolutional layers, and Xavier initialization was applied to linear layers. For baseline models, the initialization methods specified in the original studies were applied whenever available. Domaingeneralization performance was evaluated by training on the source domain and testing on unseen target domains without fine-tuning. For statistical reliability, we conducted 10 independent trials for each configuration using independently generated random seeds, and reported the mean accuracy and standard deviation. All experiments were conducted on an RTX A4000 GPU using Ubuntu 22.04, Python 3.9, PyTorch 2.3.1, and CUDA 11.8.

## C. EXPERIMENTAL RESULTS

Table 9 presents the mean cross-domain accuracies and standard deviations obtained on the three datasets using different preprocessing methods applied to the cepstrum-branch input. The results are reported under two backbone configurations for the cepstrum branch: Scenario 1 using a linear backbone and Scenario 2 using a CNN backbone. For each scenario, the proposed LL-cepstrum preprocessing method is highlighted in bold, and the best result for each domain shift is underlined.

In Scenario 1, where the cepstrum branch was constrained to a simple linear layer, LL-cepstrum preprocessing outperformed all alternative methods on the CWRU and UOS datasets and achieved performance on the PU dataset com-

TABLE 8. Domain-generalization methods for model-level comparison under single-source domain training.

|         |               | Category Method Description                                                                                                                  |
|---------|---------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| MSDG    | IEDGNet [34]  | Combines triplet loss with adversarial training using multi-domain data and can be applied in a single-domain setting.                       |
| SSDG    | AMINet [40]   | Optimizes a generator and diagnostic network via a mutual-information-based min-max game to learn domain-invariant representations.          |
| PI-SSDG | SDGPI [15]    | Extracts domain-invariant features by applying rotational-speed normalization and order track ing before training a CNN.                    |
| PI-SSDG | HmmSeNet [43] | Combines histogram matching and mixup aug mentations tailored to speed-induced amplitude changes with a high-dimensional embedding network. |

parable to that of the best-performing method, i.e., CEPS. In Scenario 2, where CNN backbones tailored to each preprocessing output were employed, the LL-cepstrum preprocessing method continued to outperform other techniques on the CWRU and UOS datasets. However, on the PU dataset, WPT and STFT outperformed the LL-cepstrum preprocessing method. This behavior is attributed to the pronounced load variations in the PU dataset, for which 2D time–frequency representations better preserved load-dependent local patterns that can be effectively exploited by 2D CNNs.

Nevertheless, LL-cepstrum preprocessing paired with the linear backbone consistently achieved high average accuracy across all three datasets, indicating that it produced broadly transferable and highly discriminative features. In most cases, these features could be effectively utilized via simple linear mapping. CWRU constituted an exception, where the CNN backbone provided additional performance gains because residual speed-related periodic components remain as localized feature distortions. This effect can be mitigated by adjusting the low-pass liftering length, as discussed in Section V-A. Overall, these results indicate that the proposed dual-branch architecture based on LL-cepstrum preprocessing can achieve strong generalization performance and computational efficiency using a compact backbone, rather than relying on a high-capacity deep model.

Building on the comparison of preprocessing performance, the proposed method was compared with the linear cepstrumbranch backbone against five recent domain-generalization approaches listed in Table 8. Table 10 shows the mean crossdomain accuracies and standard deviations obtained on the three datasets. The proposed method achieved the highest average accuracy on all datasets, indicating strong generalization capability. Fig. 5 further illustrates that the proposed method maintains a more uniform accuracy distribution across diverse operating-condition variations compared

Logo

<!-- image -->

TABLE 9. Cross-domain accuracy (%) and standard deviation for different preprocessing methods and backbone architectures.

| Domain Shift   |                   |                                      |                                     | Scenario 1: Linear Backbone Scenario 2: CNN Backbone   | Scenario 1: Linear Backbone Scenario 2: CNN Backbone   | Scenario 1: Linear Backbone Scenario 2: CNN Backbone   | Scenario 1: Linear Backbone Scenario 2: CNN Backbone   | Scenario 1: Linear Backbone Scenario 2: CNN Backbone   |
|----------------|-------------------|--------------------------------------|-------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|--------------------------------------------------------|
| Domain Shift   | LL-Cepstrum + MLP | CEPS + MLP FFT + MLP STFT + MLP      | WPT + MLP                           | LL-Cepstrum + 1D CNN                                   | CEPS + 1D CNN FFT + 1D CNN                             | STFT + 2D CNN WPT + 2D CNN                             |                                                        |                                                        |
| A→B            | 97.9±1.0          | 79.2±2.8 94.3±3.3 93.2±2.8 52.2±2.1  |                                     | 99.0±1.3                                               | 99.2±0.7                                               | 83.1±2.6 94.6±2.7 84.9±4.2                             |                                                        |                                                        |
| A→C            | 87.9±2.7          | 83.6±2.5 84.1±3.2 88.4±1.4           | 52.8±3.3                            | 96.9±3.1                                               |                                                        | 92.6±7.7 72.5±3.7 88.7±2.6 74.9±5.7                    |                                                        |                                                        |
| A→D            | 96.1±3.8          | 82.0±2.2 77.8±5.0 80.0 *             | 51.6±2.9                            | 89.4±3.1                                               |                                                        | 89.2±3.6 57.3±3.3 84.8±4.1 76.9±4.9                    |                                                        |                                                        |
| B→A            | 99.7±0.3          | 90.5±1.5 94.8±2.6 95.7±1.4 65.6±5.0  |                                     | 98.2±2.5                                               | 98.0±2.1 85.5±4.6 98.3±1.7                             | 85.5±5.2                                               |                                                        |                                                        |
| B→C            | 96.2±1.4          | 94.0±1.4 96.2±3.6 99.9±0.1           | 72.5±3.0                            | 99.9±0.1                                               | 99.9±0.1  81.5±3.7 99.9±0.1                            | 90.6±13.4                                              |                                                        |                                                        |
| B→D            | 90.0±0.3          | 91.9±3.6                             | 80.9±4.0 87.6±3.1 65.7±3.5          | 99.4±0.9                                               |                                                        | 97.5±3.4 64.8±4.4 97.1±3.3 89.6±6.9                    |                                                        |                                                        |
| C→A            | 87.6±3.5          | 81.5±3.7 91.7±3.1                    | 89.3±0.4 66.6±8.0                   | 96.9±3.0                                               |                                                        | 94.7±3.4 61.6±5.6 91.9±4.6 81.1±7.8                    |                                                        |                                                        |
| C→B            | 97.1±2.0          |                                      | 88.7±1.9 90.6±2.0 90.1±0.2 77.0±3.0 | 99.7±0.5                                               |                                                        | 99.4±0.4 73.1±4.3 93.4±2.8 86.9±3.8                    |                                                        |                                                        |
| C→D            | 91.6±1.5          | 88.2±1.9 84.1±8.6 99.9±0.2           | 74.0±2.4                            | 99.8±0.2                                               | 100.0 *                                                | 62.9±4.2 97.6±2.9 92.2±4.9                             |                                                        |                                                        |
| D→A            | 81.7±2.9          | 75.0±4.2 75.6±6.1 87.4±2.3           | 61.7±5.2                            | 92.0±3.5                                               |                                                        | 91.0±1.3 50.1±4.9 86.7±4.8 78.2±1.9                    |                                                        |                                                        |
| D→B            | 87.7±1.6          | 89.0±3.2                             | 74.5±4.1 80.5±0.5 65.2±6.3          | 98.2±1.4                                               |                                                        | 96.6±2.3 53.0±4.7 88.4±2.8 80.2±3.2                    |                                                        |                                                        |
| D→C            | 99.0±0.8          |                                      | 90.4±1.2 86.8±5.0 97.7±1.9 68.1±4.8 | 99.9±0.1                                               | 99.9±0.1                                               | 62.5±7.0 92.8±5.2 81.1±5.2                             |                                                        |                                                        |
| Average        | 92.7              |                                      | 86.2 86.0 90.8 64.4                 | 97.5                                                   |                                                        | 96.5 67.3 92.8 83.5                                    |                                                        |                                                        |
| A→B            | 99.2±0.4          | 98.9±0.2 71.0±11.7 91.9±5.6 71.5±3.1 |                                     | 98.7±2.8                                               | 72.6±15.0 75.9±13.0 75.0                               | * 77.6±3.2                                             |                                                        |                                                        |
| A→C            | 95.4±1.8          |                                      | 90.6±1.5 63.8±9.4 77.9±2.4 64.0±7.6 | 74.9±8.7                                               |                                                        | 25.5±1.0 63.9±10.8 73.9±3.0 63.7±5.7                   |                                                        |                                                        |
| B→A            | 91.7±1.9          | 91.7±3.1 61.3±3.5 92.5±3.2           | 68.2±7.2                            | 93.8±2.1                                               | 73.0±11.3 63.5±5.5 60.0±1.7 68.1±2.9                   |                                                        |                                                        |                                                        |
| B→C            | 94.5±1.6          |                                      | 86.9±3.1 75.4±1.7 78.6±4.1 74.3±0.8 | 89.7±6.7                                               |                                                        | 85.5±8.2 67.6±5.7 75.1±0.2 71.5±7.7                    |                                                        |                                                        |
| C→A            | 90.8±2.4          |                                      | 86.1±1.4 57.6±7.0 70.9±6.8 59.8±4.5 | 65.7±8.0                                               |                                                        | 29.3±3.0 31.3±6.3 57.7±4.5 61.7±6.7                    |                                                        |                                                        |
| C→B            | 95.9±1.2          |                                      | 93.8±1.3 67.1±6.2 75.6±1.5 69.3±6.4 | 92.9±4.2                                               |                                                        | 85.4±7.6 53.2±9.5 71.7±10.6 61.0±7.9                   |                                                        |                                                        |
| Average        | 94.6              |                                      | 91.3 66.0 81.2 67.9                 | 86.0                                                   |                                                        | 61.9 59.2 68.9 67.3                                    |                                                        |                                                        |
| A→B            | 99.9 *            |                                      | 99.8±0.1 99.7±0.2 97.5±0.6 59.9±2.6 | 99.5±0.2                                               |                                                        | 99.4±0.6 98.1±0.5 99.4±0.5 99.1±1.1                    |                                                        |                                                        |
| A→C            | 90.1±2.0          | 94.3±1.6                             | 72.0±6.4 53.6±4.2 49.8±2.5          | 62.1±5.7                                               |                                                        | 62.0±5.6 58.7±4.1 73.5±6.3 78.2±4.6                    |                                                        |                                                        |
| B→A            | 99.9±0.1          | 93.9±1.6                             | 99.8±0.1 97.8±1.9 97.1±1.4 57.4±4.3 | 98.1±0.8                                               | 96.5±1.3 97.4±0.9 98.5±1.0                             | 98.2±0.5                                               |                                                        |                                                        |
| B→C            | 91.7±1.8          |                                      | 67.0±3.5 62.8±2.6 44.5±4.1          | 69.7±7.9                                               |                                                        | 76.3±6.5 59.4±2.7 76.7±4.4 82.6±5.5                    |                                                        |                                                        |
| C→A            | 82.5±4.2          | 90.9±2.1                             | 72.0±1.6 50.6±3.6 42.4±3.4          | 53.4±4.5                                               | 60.7±2.6 69.6±3.7                                      | 60.0±4.0 64.4±6.3                                      |                                                        |                                                        |
| C→B            | 94.9±2.1          |                                      | 94.7±1.9 72.6±1.3 55.0±1.5 44.6±3.7 | 59.6±5.9                                               |                                                        | 63.2±3.9 67.1±2.5 69.2±3.3 70.9±5.3                    |                                                        |                                                        |
| Average        | 93.2              | 95.6                                 | 80.2 69.4 49.8                      | 73.7                                                   |                                                        | 76.4 75.0 79.6 82.2                                    |                                                        |                                                        |

TABLE 10. Cross-domain accuracy (%) and standard deviation of the proposed method compared across recent domain-generalization methods.

| Dataset   | Domain Shift   | Ours     | IEDGNet [34] AMINet [40]                                | SDGPI [15] UCL-SDG [42] HmmSeNet [43]          |
|-----------|----------------|----------|---------------------------------------------------------|------------------------------------------------|
|           | A→B            | 97.9±1.0 | 86.8±3.2 99.1±1.5                                       | 89.4±0.8 95.5±1.2 91.5±0.9                     |
|           | A→C            | 87.9±2.7 | 83.4±8.7 84.9±5.2 90.7±4.0                              | 90.0±2.3 89.9±0.1                              |
|           | A→D            | 96.1±3.8 | 80.1±5.3 74.2±6.6 76.8±7.8 68.9±6.3 80.0                | *                                              |
|           | B→A            | 99.7±0.3 |                                                         | 87.3±3.9 92.1±5.7 81.8±4.5 90.6±3.8 93.6±2.5   |
|           | B→C            | 96.2±1.4 |                                                         | 94.6±6.4 96.8±4.1 92.6±2.0 94.3±3.3 99.9±0.1   |
| CWRU      | B→D            | 90.0±0.3 | 90.3±6.8                                                | 85.2±4.6 66.5±7.3 73.5±7.1 86.8±3.2            |
|           | C→A            |          | 87.6±3.5 82.8±12.0 79.4±5.5 79.9±4.2 81.6±4.6 89.7±0.3  |                                                |
|           | C→B            | 97.1±2.0 |                                                         | 90.0±0.5 89.9±3.8 90.1±0.5 95.3±1.0 90.0±0.0   |
|           | C→D            | 91.6±1.5 | 97.5±3.5 99.8±0.5 96.2±4.6 88.8±3.6 100.0               | *                                              |
|           | D→A            | 81.7±2.9 | 67.2±8.7 77.2±4.4 79.7±12.0 82.6±4.4                    | 79.5±0.2                                       |
|           | D→B            | 87.7±1.6 | 72.8±6.9 80.1±4.2 78.7±6.6 90.6±2.0                     | 80.0 *                                         |
|           | D→C            | 99.0±0.8 |                                                         | 84.4±5.8 86.1±5.3 88.1±5.6 98.6±1.3 93.6±2.5   |
|           | Average        | 92.7     |                                                         | 84.8 87.1 84.2 87.5 89.5                       |
| UOS       | A→B            | 99.2±0.4 |                                                         | 72.5±2.6 86.4±10.5 91.4±6.1 80.5±6.7 73.2±1.9  |
| UOS       | A→C            |          | 95.4±1.8 50.0±11.0 73.3±9.9 83.7±3.9 61.1±6.7 48.6±11.1 |                                                |
|           | B→A            | 91.7±1.9 |                                                         | 74.2±0.2 62.3±12.3 90.8±9.5 47.9±6.0 73.9±0.3  |
|           | B→C            | 94.5±1.6 |                                                         | 74.4±1.2 77.0±12.5 93.6±2.3 48.0±10.2 74.8±0.2 |
|           | C→A            | 90.8±2.4 |                                                         | 69.9±6.4 63.0±10.5 77.6±9.7 36.3±6.6 70.2±3.9  |
|           | C→B            | 95.9±1.2 | 58.6±9.2 77.1±12.4 96.1±6.2                             | 29.3±4.2 50.4±0.9                              |
|           | Average        | 94.6     |                                                         | 66.6 73.2 88.9 50.5 65.2                       |
|           | A→B            | 99.9 *   |                                                         | 99.6±0.7 99.7±0.4 97.8±7.0 78.1±3.1 99.9±0.1   |
| PU        | A→C            | 90.1±2.0 |                                                         | 69.9±6.3 73.6±6.7 74.7±4.6 47.1±3.3 56.7±2.4   |
| PU        | B→A            |          | 99.9±0.1 90.5±14.6 97.4±3.0 94.0±3.5 76.4±3.3 99.7±0.2  |                                                |
| PU        | B→C            | 91.7±1.8 |                                                         | 69.5±3.5 84.9±3.8 66.2±3.8 49.0±4.3 60.6±2.0   |
| PU        | C→A            | 82.5±4.2 | 53.2±5.4 83.1±4.5                                       | 60.9±5.4 58.2±2.8 46.4±1.9                     |
| PU        | C→B            | 94.9±2.1 |                                                         | 54.3±4.4 80.5±9.2 63.8±8.3 54.0±3.0 48.1±3.7   |
| PU        | Average        | 93.2     |                                                         | 72.8 86.5 76.2 60.5 68.6                       |

*

The reported standard deviation is below 0 . 1 after rounding to one decimal place.

with competing approaches, suggesting reduced bias toward specific domain scenarios.

IEDGNet [34] and AMINet [40] aim to improve generalization by expanding the source-domain distribution via transformations, augmentations, or virtual sample generation. Although these approaches achieved competitive performance under relatively limited domain shifts, such as those in CWRU and several PU scenarios, their performance deteriorated on the UOS dataset, where the domain shift was larger and more irregular. These results indicate that distribution expansion alone may be insufficient to capture the complexity of real variations in operating conditions.

Physics-informed methods enhance robustness by modeling variations in operating conditions under specific physical assumptions. For instance, SDGPI [15], which explicitly normalizes the rotational speed, remained relatively stable on the UOS dataset, where the speed variation was dominant, but exhibited limited effectiveness on the CWRU and PU datasets, where the load variation also played a significant role. Similarly, UCL-SDG [42] and HmmSeNet [43], which rely on empirical assumptions, such as peak stability and histogram invariance, exhibited degraded performance on the PU and UOS datasets. These results suggest that generalization performance may become constrained when actual domain-shift patterns deviate from the underlying assumptions. In contrast, the proposed method leveraged a physicsguided perspective on signal generation, demonstrating stable performance across all three datasets despite differences in domain-shift type and severity. This robustness makes the proposed method a practical SSDG framework for reliable fault diagnosis under unseen target domain conditions.

Logo

<!-- image -->

IEEE Access

Other

FIGURE 5. Cross-domain accuracy distributions of different domain-generalization methods across various domain shifts. (a) CWRU. (b) UOS. (c) PU.

<!-- image -->

## V. ANALYSIS OF PROPOSED METHOD

## A. PREPROCESSING EFFECTIVENESS

To isolate and validate the effects of the proposed LLcepstrum preprocessing method, we conducted a study using a single-branch cepstrum model. As baselines, the conventional real cepstrum (Cep) and the proposed log-meanremoved cepstrum (Log-mean) were considered. For fair comparison, the same low-pass liftering operation was applied to both representations while progressively reducing the low-pass liftering length κ from L to L/32. Subsequently, the mean cross-domain accuracy was measured for each dataset. Fig. 6 shows the accuracy trends with respect to κ for the three datasets.

For a fixed κ, Log-mean achieved an accuracy comparable to or higher than that of Cep in most settings, except within a limited range on the CWRU dataset. In particular, consistent improvements were observed on the UOS dataset, where speed variation was substantial, and stable performance was obtained across most κ values on the CWRU and PU datasets. These results indicate that removing the mean of the logmagnitude spectrum mitigates speed-induced amplitude scaling across domains, thereby enabling more domain-robust representations. Based on these observations, Log-mean was adopted as the default preprocessing method in subsequent experiments.

From the κ sweep in Figs. 6(a)–(c), the highest accuracy on the variable-speed CWRU and UOS datasets was obtained when κ ≤ L/8, whereas the fixed-speed PU dataset exhibited a gradual degradation as κ decreased. To better understand this behavior, Fig. 7 presents the log-mean cepstrum extracted with κ = L/8 for the highest-speed operating condition of each dataset in the IR fault case, which produces the lowest-quefrency peak. The shaded bands indicate quefrency locations corresponding to the integer multiples of the inverse of the characteristic fault frequencies.

In variable-speed datasets, including CWRU and UOS, excitation-induced periodic structures can shift across domains and thus become domain-specific factors. In the

Bar chart

FIGURE 6. Mean cross-domain accuracy of the real cepstrum (Cep) and log-mean-removed cepstrum (Log-mean) under different liftering lengths (κ). (a) CWRU. (b) UOS. (c) PU. For clarity, the y-axis range is adjusted separately for each dataset.

<!-- image -->

CWRU dataset, these structures remained visible even when κ = L/8, appearing at the inverses of the fault frequencies and their harmonics, as shown in Fig. 7(a). As κ was further reduced below L/8, the periodic components became increasingly attenuated, and the cross-domain accuracy improved, as shown in Fig. 6(a). However, in the UOS dataset, reducing κ below L/8 suppressed the excitation-induced periodic structures and discriminative components necessary for fault separation, as shown in Fig. 7(b). Consequently, the performance degraded, as shown in Fig. 6(b). These observations

Line chart

FIGURE 7. Log-mean cepstrum extracted with κ = L/8 under the highest-speed domain for each dataset, shown for the IR fault case. Locations corresponding to the integer multiples of the inverse characteristic fault frequencies are indicated. (a) CWRU. (b) UOS. (c) PU.

<!-- image -->

Line chart

FIGURE 8. IG attribution maps for an IR fault sample from the CWRU dataset, based on the log-mean cepstrum extracted with κ = L/8. (a) Correctly classified case. (b) Misclassified case.

<!-- image -->

suggest that κ = L/8 provides a tradeoff between suppressing domain-specific periodic structures and preserving essential fault-related information.

In contrast, for the PU dataset, where rotational speed is fixed, the periodic structure is preserved across domains and thus provides fault-discriminative features rather than domain-specific factors. Therefore, excessively reducing κ removes informative components for fault identification and

Logo

<!-- image -->

IEEE Access

TABLE 11. Model variants used in the ablation study.

|   Variant | Model name        | Time branch   | Cepstrum branch   | ReLU gating   | Branch-spec. losses   |
|-----------|-------------------|---------------|-------------------|---------------|-----------------------|
|         1 | Time Only         | ✓             |                   | – – –         |                       |
|         2 | Cep Only –        |               | ✓                 |               | – –                   |
|         3 | BS-, ReLU-        |               | ✓ ✓               |               | – –                   |
|         4 | BS-, ReLU+        |               | ✓ ✓ ✓             |               | –                     |
|         5 | BS+, ReLU+ (Ours) |               | ✓ ✓ ✓ ✓           |               |                       |

degrades performance, as shown in Fig. 6(c). Under the SSDG setting, we fixed κ = L/8 for all subsequent experiments because it provided balanced performance across the three datasets.

To verify that the proposed κ selection facilitated the learning of domain-robust features, we analyzed input attributes using integrated gradients (IGs) [47]. Fig. 8 compares the attribution maps for a CWRU IR fault sample using the logmean input with κ = L/8 under correctly classified and misclassified cases. For the correct classification, high attribution values are concentrated on the low-quefrency transferfunction-related components, as shown in Fig. 8(a). In contrast, for the misclassification shown in Fig. 8(b), strong positive attribution appears around quefrency regions corresponding to excitation-induced periodic structures. These observations suggest that periodic structures removed by controlling κ can act as confounding factors that trigger incorrect decisions, thereby supporting the effectiveness of the proposed κ selection for robust diagnosis under speed-varying conditions.

## B. ABLATION STUDY OF MODEL AND TRAINING STRATEGY

To verify the effectiveness of the proposed dual-branch architecture and joint optimization strategy with branch-specific losses, an ablation study was conducted using a set of model variants that progressively validated each design choice. The model variants and their configurations are presented in Table 11. Variant 1 uses only the time branch, Variant 2 uses only the cepstrum branch, and Variant 3 concatenates features from both branches without ReLU gating or branch-specific losses. Variant 4 introduces ReLU gating at the end of the time branch while maintaining simple feature concatenation, and Variant 5 further incorporates branch-specific losses in addition to ReLU gating. Fig. 9 shows the mean cross-domain accuracy and standard deviation of each variant across the three datasets.

Variant 1 consistently yielded low accuracy, confirming that the raw time-domain representation alone was insufficient for robust generalization under complex domain shifts. In contrast, Variant 2 achieved the best performance on the UOS dataset but performed the worst on the CWRU dataset, showing substantial performance discrepancies between datasets. These results indicate that although the preprocessed cepstrum representation provides a strong basis for domain-robust diagnosis, reliance on a single representation does not reliably cover diverse operating conditions.

Logo

<!-- image -->

IEEE Access

Bar chart

FIGURE 9. Mean cross-domain accuracy and standard deviation for each model variant listed in Table 11 on the three datasets.

<!-- image -->

Variant 3 improved the performance over Variant 2 on the CWRU dataset but exhibited a substantial performance drop on the UOS dataset, indicating limited stability across datasets. This finding suggests that naive feature concatenation can introduce noninformative or domain-sensitive components from the time branch, thereby undermining the robustness of the cepstrum representation. In contrast, Variant 4 alleviated this issue by applying ReLU gating at the end of the time branch. This operation suppressed nonessential components and improved the cross-dataset performance relative to simple concatenation while avoiding notable degradation on any specific dataset. In this context, ReLU functioned as a lightweight gating mechanism that regulated the contribution of the time branch.

Finally, Variant 5, i.e., the proposed dual-branch model, further improved the mean accuracy and reduced the standard deviation by introducing joint optimization with branchspecific losses in addition to ReLU gating. This training strategy prevented over-reliance on a single branch during optimization and encouraged both branches to learn complementary and discriminative representations. Notably, Variant 5 preserved the strong performance of Variant 2 on the UOS dataset while substantially improving the accuracy on datasets where Variant 2 was less reliable. As a result, it achieved both a high average accuracy and low variability across all three datasets. These results demonstrate that the proposed method delivers strong performance across diverse domain-shift scenarios without being overly tuned to a particular dataset.

## C. FEATURE VISUALIZATION

In addition to the ablation study, we performed feature visualization and confusion matrix analysis to provide an intuitive assessment of the generalization behavior of the proposed method compared with single-branch baselines. For a consistent comparison across models, each model was trained on a single-source domain (Domain A) and tested on a pooled set comprising samples from all remaining target domains.

Fig. 10 presents representative visualizations of the CWRU dataset, including t-distributed stochastic neighbor embedding (t-SNE) projections colored according to class, domain, and prediction correctness, and the corresponding confusion matrices. These visualizations are shown for the time-only configuration, the cepstrum-only configuration, and the proposed dual-branch model. The time-only configuration exhibited limited separability in the t-SNE projection. In particular, the B1, B2, and B3 clusters overlapped substantially, and misclassifications were concentrated in this region. This behavior was consistent with the confusion matrix, which showed reduced diagonal entries and increased confusion among the ball fault classes. When only the cepstrum branch was used, the model formed relatively compact clusters for many classes. However, persistent confusion was observed for certain class pairs, such as IR1–OR3 and B2–B3. These results suggest that the target-domain feature shifts may move some samples across the decision boundaries learned from the source domain. In contrast, the proposed dual-branch model improved separability in the fused feature space. The B2 and B3 classes, which were intermixed in the time-only configuration, became more clearly separated. In addition, the prominent IR1–OR3 confusion observed in the cepstrumonly configuration was alleviated, indicating more stable classification under unseen domains.

For the UOS dataset, the confusion matrices shown in Fig. 11 indicate that the cepstrum-only configuration and the proposed dual-branch model achieved comparably high accuracy across most classes. These observations suggest that, under the domain-shift characteristics of the UOS dataset, cepstrum-based representations alone can provide sufficiently domain-robust features, and the proposed model maintains the same performance level. Importantly, the proposed model preserves this strong performance while also maintaining robustness across datasets with different domainshift characteristics.

For the PU dataset, the confusion matrices shown in Fig. 12 indicate that the time-only configuration yielded limited discriminability for several outer-race fault classes, whereas the cepstrum-only configuration tended to produce class-specific prediction bias, frequently misclassifying samples as a particular class (IR4). In contrast, the proposed dual-branch model substantially mitigated this bias. For example, the misclassification rate of IR2 as IR4 decreased from 24% to 0%. In addition, the diagonal entries corresponding to the outer-race fault classes increased relative to those of the time-only configuration, indicating reduced class-specific prediction bias and improved discriminability.

In summary, across all three datasets, using only the time branch often resulted in reduced between-class margins under domain shifts, leading to class overlap and classification errors. The cepstrum-only configuration generally provided robust representation; however, it could still struggle with specific class pairs or exhibit biased predictions under certain conditions. In contrast, the proposed model improved domain generalization by integrating transient time-domain features with a domain-robust cepstrum representation. Consequently, the model enhanced separability in previously overlapping regions and alleviated class-specific prediction bias. In particular, misclassifications for confusing class pairs were sub-

Logo

<!-- image -->

IEEE Access

Scatter plot

FIGURE 10. t-SNE visualizations and confusion matrices for three models on the CWRU dataset, showing class/domain information and prediction correctness. (a) Time branch only. (b) Cepstrum branch only. (c) Proposed dual-branch model.

<!-- image -->

stantially reduced on the CWRU and PU datasets while maintaining the high performance achieved on the UOS dataset.

To quantitatively support these visual observations, the feature space was analyzed using two standard cluster-quality metrics that complement the t-SNE projections: the Silhouette Score ( ¯ s ) [48], i.e., the mean silhouette coefficient that contrasts within-cluster cohesion with between-cluster separation, and the Davies–Bouldin Index (DBI) [49], which evaluates the average ratio of within-class spread to betweenclass distance across cluster pairs. Higher ¯ s and lower DBI values both indicate stronger cluster separation. Across all three datasets, the proposed dual-branch model achieved the highest ¯ s and the lowest DBI values, as summarized in Ta- ble 12. This consistent ordering across both metrics indicates that the dual-branch feature space yields tighter within-class clusters and larger between-class margins compared with the single-branch baselines. These findings are consistent with the visual cluster separation observed in the t-SNE projections and the reduced misclassifications observed in the confusion matrices.

## VI. CONCLUSIONS AND FUTURE WORK

This study proposed a PI-SSDG framework for bearing fault diagnosis, motivated by two practical challenges in industrial environments: operating-condition variations, such as changes in rotational speed and load, and the limited avail-

Logo

<!-- image -->

IEEE Access

Bar chart

FIGURE 11. Confusion matrices for the UOS dataset. (a) Time branch only. (b) Cepstrum branch only. (c) Proposed dual-branch model.

<!-- image -->

Bar chart

FIGURE 12. Confusion matrices for the PU dataset. (a) Time branch only. (b) Cepstrum branch only. (c) Proposed dual-branch model.

<!-- image -->

TABLE 12. Comparison of feature-space class separability across the three datasets using the Silhouette Score and Davies–Bouldin Index.

| Dataset Model   | ¯ s  DBI                               |
|-----------------|----------------------------------------|
| CWRU            | Time branch only 0.444 0.988           |
| CWRU            | Cepstrum branch only 0.460 0.911       |
| CWRU            | Proposed dual-branch model 0.511 0.844 |
| UOS             | Time branch only 0.264 1.772           |
| UOS             | Cepstrum branch only 0.635 0.535       |
| UOS             | Proposed dual-branch model 0.707 0.456 |
| PU              | Time branch only 0.316 1.277           |
| PU              | Cepstrum branch only 0.384 1.014       |
| PU              | Proposed dual-branch model 0.492 0.796 |

ability of reliably labeled fault data. Contrary to previous approaches that assume the availability of multiple source domains, access to target-domain data, or complex resampling procedures, the proposed method was designed to remain robust to unseen operating-condition variations using only a single-source domain.

To extract domain-robust features, we developed a cepstrum-based preprocessing method that reformulates the conventional real cepstrum pipeline by incorporating logmagnitude spectral mean removal and low-pass liftering. This design reduced global amplitude scaling induced by operating conditions and suppressed excitation-related pe- riodic components at the input stage, thereby emphasizing transfer-function-related information. In addition, a dualbranch model was introduced and optimized using a joint training strategy with branch-specific losses and fusion loss, leveraging the preprocessed cepstrum and raw signal as complementary inputs. This design promoted the integration of complementary features between the two representations, each of which captured distinct physical characteristics of the system.

Extensive experiments on three public bearing datasets demonstrated that the proposed model achieved consistently high cross-domain accuracy and outperformed recent domain-generalization methods across a wide range of speed and load variations. Ablation studies and visualization analyses further confirmed that the proposed preprocessing method provided a key foundation for robust generalization. The results also showed that the dual-branch architecture with joint optimization using branch-specific losses alleviated over-reliance on a single representation and reduced classspecific prediction bias, thereby improving generalization performance.

In future work, we plan to reduce the computational cost of the proposed framework for edge deployment. We also aim to extend the framework to incorporate post-deployment unsupervised calibration of preprocessing hyperparameters

such as κ using only normal-condition data collected in the deployment setting.

## REFERENCES

- [1] P. Zhou, Y. Yang, H. Wang, M. Du, Z. Peng, and W. Zhang, ''The relationship between fault-induced impulses and harmonic-cluster with applications to rotating machinery fault diagnosis,'' Mech. Syst. Signal Process., vol. 144, p. 106896, Oct. 2020.
- [2] S. Nandi and H. A. Toliyat, ''Fault diagnosis of electrical machines-a review,'' in Proc. IEEE Int. Electr. Mach. Drives Conf., May 1999, pp. 219– 221.
- [3] S. A. McInerny and Y. Dai, ''Basic vibration signal processing for bearing fault detection,'' IEEE Trans. Educ., vol. 46, no. 1, pp. 149–156, Feb. 2003.
- [4] X. Chen, B. Zhang, and D. Gao, ''Bearing fault diagnosis base on multiscale CNN and LSTM model,'' J. Intell. Manuf., vol. 32, no. 4, pp. 971–987, Apr. 2021.
- [5] S. Lee and T. Kim, ''FRFconv-TDSNet: Lightweight, noise-robust convolutional neural network leveraging full-receptive-field convolution and time-domain statistics for intelligent machine fault diagnosis,'' IEEE Trans. Instrum. Meas., vol. 73, pp. 1–13, Aug. 2024.
- [6] A. Shenfield and M. Howarth, ''A novel deep learning model for the detection and identification of rolling element-bearing faults,'' Sensors , vol. 20, no. 18, p. 5112, Sep. 2020.
- [7] Y. Shi, X. Ying, and J. Yang, ''Deep unsupervised domain adaptation with time series sensor data: A survey,'' Sensors, vol. 22, no. 15, p. 5507, Jul. 2022.
- [8] D. Latil, R. H. Ngouna, K. Medjaher, and S. Lhuisset, ''Vibration-based data-driven fault diagnosis of rotating machines operating under varying working conditions: A review and bibliometric analysis,'' Int. J. Progn. Health. Manag., vol. 16, no. 2, Jul. 2025.
- [9] D. Neupane, M. R. Bouadjenek, R. Dazeley, and S. Aryal, ''Data-driven machinery fault diagnosis: A comprehensive review,'' Neurocomputing , vol. 627, p. 129588, Apr. 2025.
- [10] Z. Zhao, Q. Zhang, X. Yu, C. Sun, S. Wang, R. Yan, and X. Chen, ''Applications of unsupervised deep transfer learning to intelligent fault diagnosis: A survey and comparative study,'' IEEE Trans. Instrum. Meas. , vol. 70, pp. 1–28, Sep. 2021.
- [11] K. R. Fyfe and E. D. S. Munck, ''Analysis of computed order tracking,'' Mech. Syst. Signal Process., vol. 11, no. 2, pp. 187–205, Mar. 1997.
- [12] R. B. Randall and W. Smith, ''New cepstral methods for the diagnosis of gear and bearing faults under variable speed conditions,'' in Proc. 23rd Int. Congress Sound Vib., Jul. 2016.
- [13] Y. Xiao, H. Shao, S. Yan, J. Wang, Y. Peng, and B. Liu, ''Domain generalization for rotating machinery fault diagnosis: A survey,'' Adv. Eng. Inform., vol. 64, p. 103063, Mar. 2025.
- [14] Q. Ni, J. C. Ji, B. Halkon, K. Feng, and A. K. Nandi, ''Physics-informed residual network (PIResNet) for rolling element bearing fault diagnostics,'' Mech. Syst. Signal Process., vol. 200, p. 110544, Oct. 2023.
- [15] I. Kim, S. W. Kim, J. Kim, H. Huh, I. Jeong, T. Choi, J. Kim, and S. Lee, ''Single domain generalizable and physically interpretable bearing fault diagnosis for unseen working conditions,'' Expert Syst. Appl., vol. 241, p. 122455, May 2024.
- [16] Q. Qian, Y. Wang, T. Zhang, and Y. Qin, ''Maximum mean square discrepancy: A new discrepancy representation metric for mechanical fault transfer diagnosis,'' Knowl. Based Syst., vol. 276, Sep. 2023.
- [17] X. Shao and C. S. Kim, ''Unsupervised domain adaptive 1D-CNN for fault diagnosis of bearing,'' Sensors, vol. 22, no. 11, p. 110748, May 2022.
- [18] B. Xia, K. Wang, A. Xu, P. Zeng, N. Yang, and B. Li, ''Intelligent fault diagnosis for bearings of industrial robot joints under varying working conditions based on deep adversarial domain adaptation,'' IEEE Trans. Instrum. Meas., vol. 71, pp. 1–13, Mar. 2022.
- [19] B. Liu, C. Yan, Y. Liu, M. Lv, Y. Huang, and L. Wu, ''ISEANet: An interpretable subdomain enhanced adaptive network for unsupervised crossdomain fault diagnosis of rolling bearing,'' Adv. Eng. Inform., vol. 62, p. 102610, 2024.
- [20] B. Liu, C. Yan, C. He, M. Lv, J. Wei, and L. Wu, ''An interpretable physics-informed subdomain moment-enhanced adaptation network for unsupervised transfer fault diagnosis of rolling bearing,'' Adv. Eng. Inform. , vol. 67, p. 103491, 2025.
- [21] Y. Guo, T. W. Liu, J. Na, and R. F. Fung, ''Envelope order tracking for fault detection in rolling element bearings,'' J. Sound Vib., vol. 331, no. 25, pp. 5644–5654, Dec. 2012.
- [22] M. Ji, G. Peng, J. He, S. Liu, Z. Chen, and S. Li, ''A two-stage, intelligent bearing-fault-diagnosis method using order-tracking and a onedimensional convolutional neural network with variable speeds,'' Sensors , vol. 21, no. 3, p. 675, Jan. 2021.
- [23] T. Kim and J. Chai, ''Pre-processing method to improve cross-domain fault diagnosis for bearing,'' Sensors, vol. 21, no. 15, p. 4970, Jul. 2021.
- [24] J. Wang, Y. Sun, and W. Wang, ''Bearing fault diagnosis based on improved cepstrum under variable speed condition,'' Eng. Res. Express, vol. 5, p. 025051, May 2023.
- [25] T. Hu, T. Tang, R. Lin, M. Chen, S. Han, and J. Wu, ''A simple data augmentation algorithm and a self-adaptive convolutional architecture for fewshot fault diagnosis under different working conditions,'' Measurement , vol. 156, p. 107539, May 2020.
- [26] M. T. Pham, L. Kuzniar, M. Pieczynski, D. Plotzka, and D. Andruszkiewicz, ''Accurate bearing fault diagnosis under variable shaft speed using convolutional neural networks and vibration spectrogram,'' Appl. Sci., vol. 10, no. 18, p. 6385, Sep. 2020.
- [27] Q. Zhang and L. Deng, ''An intelligent fault diagnosis method of rolling bearings based on short-time fourier transform and convolutional neural network,'' J. Fail. Anal. Prev., vol. 23, no. 2, pp. 795–811, Feb. 2023.
- [28] X. Yu, H. Chen, L. Qiu, C. Wei, L. Liu, X. Li, and Q. Zhao, ''A wavelet packet transform-based deep feature transfer learning method for bearing fault diagnosis under different working conditions,'' Measurement, vol. 201, p. 111597, Sep. 2022.
- [29] M. Zhao, M. Kang, B. Tang, and M. Pecht, ''Multiple wavelet coefficients fusion in deep residual networks for fault diagnosis,'' IEEE Trans. Ind. Electron., vol. 66, no. 6, pp. 4696–4706, Jun. 2019.
- [30] P. Borghesani, P. Pennacchi, R. B. Randall, N. Sawalhi, and R. Ricci, ''Application of cepstrum pre-whitening for the diagnosis of bearing faults under variable speed conditions,'' Mech. Syst. Signal Process., vol. 36, no. 2, pp. 370–384, 2013.
- [31] R. B. Randall, ''A history of cepstrum analysis and its application to mechanical problems,'' Mech. Syst. Signal Process., vol. 97, pp. 3–19, 2017.
- [32] H. Shao et al., ''Dual-threshold attention-guided GAN and limited infrared thermal images for rotating machinery fault diagnosis under speed fluctuation,'' IEEE Trans. Ind. Informat., vol. 19, no. 9, pp. 9933–9942, Sep. 2023.
- [33] Z. Fan, Q. Xu, C. Jiang, and S. X. Ding, ''Deep mixed domain generalization network for intelligent fault diagnosis under unseen conditions,'' IEEE Trans. Ind. Electron., vol. 71, no. 1, pp. 965–974, Jan. 2024.
- [34] T. Han, Y.-F. Li, and M. Qian, ''A hybrid generalization network for intelligent fault diagnosis of rotating machinery under unseen working conditions,'' IEEE Trans. Instrum. Meas., vol. 70, pp. 1–11, Jun. 2021.
- [35] R. Wang, W. Huang, Y. Lu, X. Zhang, J. Wang, C. Ding, and C. Shen, ''A novel domain generalization network with multidomain specific auxiliary classifiers for machinery fault diagnosis under unseen working conditions,'' Reliab. Eng. Syst. Saf., vol. 238, p. 109463, Oct. 2023.
- [36] C. Zhao and W. Shen, ''A domain generalization network combing invariance and specificity towards real-time intelligent fault diagnosis,'' Mech. Syst. Signal Process., vol. 173, p. 108990, Jul. 2022.
- [37] R. Huang, J. Li, Y. Liao, J. Chen, Z. Wang, and W. Li, ''Deep adversarial capsule network for compound fault diagnosis of machinery toward multidomain generalization task,'' IEEE Trans. Instrum. Meas., vol. 70, pp. 1–11, 2021.
- [38] H. Zheng, Y. Yang, J. Yin, Y. Li, R. Wang, and M. Xu, ''Deep domain generalization combining a priori diagnosis knowledge toward cross-domain fault diagnosis of rolling bearing,'' IEEE Trans. Instrum. Meas., vol. 70, pp. 1–11, Aug. 2021.
- [39] Y. Xie, J. Shi, C. Gao, G. Yang, Z. Zhao, G. Guan, and D. Chen, ''Rolling bearing fault diagnosis method based on dual invariant feature domain generalization,'' IEEE Trans. Instrum. Meas., vol. 73, pp. 1–11, Feb. 2024.
- [40] C. Zhao and W. Shen, ''Adversarial mutual information-guided single domain generalization network for intelligent fault diagnosis,'' IEEE Trans. Ind. Informat., vol. 19, no. 3, pp. 2909–2918, May 2022.
- [41] Y. Huang, W. Huang, X. Hu, Z. Liu, and J. Huo, ''UDDGN: Domainindependent compact boundary learning method for universal diagnosis domain generation,'' IEEE Trans. Instrum. Meas., vol. 74, pp. 1–20, 2025.
- [42] Q. Wu, Y. Ma, Z. Feng, S. Yang, and H. Hu, ''Unsupervised contrastive learning based single domain generalization method for intelligent bearing fault diagnosis,'' IEEE Sens. J., vol. 25, no. 2, pp. 3923–3934, Dec. 2024.
- [43] J. Tang, X. Ding, C. Wei, J. Xiao, R. Liu, L. Wang, and W. Huang, ''HmmSeNet: A novel single domain generalization equipment fault diagnosis

Logo

<!-- image -->

Logo

<!-- image -->

IEEE Access

- under unknown working speed using histogram matching mixup,'' IEEE Trans. Ind. Informat., vol. 20, no. 5, pp. 7162–7172, May 2024.
- [44] D. Neupane and J. Seok, ''Bearing fault detection and diagnosis using case western reserve university dataset with deep learning approaches: A review,'' IEEE Access, vol. 8, pp. 93 155–93 178, Apr. 2020.
- [45] S. Lee, T. Kim, and T. Kim, ''Multi-domain vibration dataset with various bearing types under compound machine fault scenarios,'' Data Br., vol. 57, p. 110940, Dec. 2024.
- [46] C. Lessmeier, J. K. Kimotho, D. Zimmer, and W. Sextro, ''Condition monitoring of bearing damage in electromechanical drive systems by using motor current signals of electric motors: A benchmark data set for datadriven classification,'' in Proc. Eur. Conf. Prognostics Health Manage. Soc., vol. 3, no. 1, Jul. 2016, pp. 5–8.
- [47] M. Sundararajan, A. Taly, and Q. Yan, ''Axiomatic attribution for deep networks,'' in Proc. 34th Int. Conf. Mach. Learn., vol. 70, Aug. 2017, pp. 3319–3328.
- [48] P. J. Rousseeuw, ''Silhouettes: A graphical aid to the interpretation and validation of cluster analysis,'' J. Comput. Appl. Math., vol. 20, pp. 53–65, 1987.
- [49] D. L. Davies and D. W. Bouldin, ''A cluster separation measure,'' IEEE Trans. Pattern Anal. Mach. Intell., vol. PAMI-1, no. 2, pp. 224–227, 1979.

Photograph

<!-- image -->

SUHYUN KIM received B.S. and M.S. degrees in Mechanical and Information Engineering from the University of Seoul, Korea, in 2024 and 2026, respectively. He is currently a research engineer with the tractor research institute, LS Mtron, Ltd., Korea. His research interests include on-device AI and real-time embedded systems.

Photograph

<!-- image -->

TAEHYOUN KIM (M'06) received B.S., M.S., and Ph.D. degrees in Computer Engineering from Seoul National University, Korea, in 1994, 1996, and 2001, respectively. From 2001 to 2005, he was an R&amp;D manager at the SoC Division, GCT Research, Inc. Since 2005, he has been a professor in the Department of Mechanical and Information Engineering at the University of Seoul, Korea. His current research interests include real-time embedded on-device AI intelligence and IIoT systems.