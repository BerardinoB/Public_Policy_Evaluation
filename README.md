# Causal Impact Evaluation of Occupational Safety Policies on Firms’ default using Machine Learning Uplift Modelling
**Abstract**. It is often undermined that occupational safety policies do not only displace a direct effect on work well-being, but also an indirect effect on firms’ economic performances. In such context, econometric models dominated the scenes of causality until recently while Machine Learning models were seen with skepticism. With the rise of complex datasets, an ever-increasing need for automated algorithms capable to handle complex non-linear relationships between variables has brought to uncover the power of Machine Learning for causality. In this paper, we carry out an evaluation of a public aid-scheme implemented in Italy and oriented to support investment of small and medium enterprises (SMEs) in occupational safety and health (OSH) for assessing the impact on the survival of corporations. A comparison of thirteen models is performed and the Individual Treatment Effect (ITE) estimated and validated based on the AUUC and Qini score for which best values of 0.064 and 0.407, respectively, are obtained based on the Light Gradient Boosting Machine (LightGBM). An additional in-depth statistical analysis also revealed that the best beneficiaries of the policy intervention are those firms that experience performance issues in the period just before the interventions and for which the increased liquidity brought by the policy may have prevented default.

![alt text](https://github.com/BerardinoB/Public_Policy_Evaluation/blob/main/Images/Maps_Italy.png)

## Performance: Factual Results
in order to calculate the probability of corporate failure, a stratified 10-folds Cross-Validation (CV) procedure is implemented. We used CV aggregation or crogging to improve the generalization error estimate using our validation methods. The entire process is repeated 30 times and the obtained predictions averaged at the instance level in order to obtain reliable results. For all ML models, the same folds partitioning is used in order to guarantee fair comparisons. Cost sensitive learning is applied to deal with class imbalance by re-weighting the loss function toward the less represented. For the factual comparison of all the ML models, the ROC-AUC curve is reported below:

![alt text](https://github.com/BerardinoB/Public_Policy_Evaluation/blob/main/Images/ROC_Curve.png)

The highest classification score is obtained with RFSurvival which outperformed all other classification and survival-based algorithms. It is possible to notice that all tree-based models resulted very close to one another and provided the best predictive results. The worst performances are obtained with SVMSurvival and AAF with an AUC score of 0.69 in both cases.

## Policy Evaluation: Average Treatment Effect (ATE)
An evaluation of the public policy intervention designed by INAIL is proposed, which represents the primary aim of this work. In order to estimate the causal impact of the policy on the survival of Treated corporations, both a statistical and ML analysis is developed. Two different approaches for the calculation of the ATE are compared (see image below). A penalized CPH and a Non-Parametric Bootstrap method are implemented for the estimation of the ATE (vertical axis). For the CPH, different penalization terms in the range between 0.4 and 4 at steps of 0.1 are used (bottom horizontal axis). The black line represents the ATE estimated from the CPH at different penalization terms while in gray the respective 95% confidence intervals. The red line represents the expected ATE over all possible penalization terms. The green histogram depicts the distribution of the ATE obtained from the bootstrapping procedure. The frequency for each bin is reported in the upper horizontal axis. The blue line represents the expected ATE of the bootstrapping distribution.

![alt text](https://github.com/BerardinoB/Public_Policy_Evaluation/blob/main/Images/ATE_Stat.png)


## Individual Treatment Effect and Uplift Modelling

The uplift percentile distribution estimated from each model (horizontal axis) is plotted against the ATE estimated at each percentile threshold (vertical axis). Specifically, for each decile of the ITE distribution (horizontal axis), we compute the ATE (vertical axis). It is clear from the image that by sorting the dataset from the most responsive to the least responsive firms, a monotonic increase of the cumulative uplift is observed. Most of the models are able to accurately rank firms with respect to which the policy intervention is more effective, reducing the default probability up to 30% when the 10% of most responsive firms are selected. This is an interesting result since such a percentage represents a 10-folds increase in magnitude with respect to the overall bootstrapping ATE estimation.

![alt text](https://github.com/BerardinoB/Public_Policy_Evaluation/blob/main/Images/Gain_Chart.png)
