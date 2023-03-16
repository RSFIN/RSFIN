# RSFIN (Rule Search-based Fuzzy Inference Network)
Many modern software systems provide numerous configuration options that users can adjust for specific running environments. However, how to choose an appropriate configuration in different application scenarios brings trouble to users and engineers, because the complex impact of the configuration on the system performance and the lack of understanding of the system. To address this problem, various performance prediction methods have been proposed to utilize data-based machine learning models to learn configuration spaces. 



The essential difference between these methods is the way in which the underlying distribution of the configuration space is developed and explored. The core idea of RSFIN is to **adaptively explore and capture the underlying distribution hidden in the configuration space and perform performance inference based on the distribution**. Specifically, RSFIN consists of two main steps:

- Step 1: Construct a distribution model to capture the hidden underlying distribution.
- Step 2: Based on the distribution model, training the regression models to achieve prediction.

# Prerequisites
- Python 3.x
- numpy
- pandas
- matplotlib
- scipy
- scikit-learn
- tqdm

# Installation
RSFIN can be directly executed through source code:
1. Download and install [Python 3.x](https://www.python.org/downloads/).

2. Clone RSFIN.

   ``` $ git clone http://github.com/RSFIN/RSFIN.git```

3. Install required packages

   ``` $ pip3 install -r require.txt```

   ​

# Subject Systems
Our experiments are based on the [SPLConqueror open dataset](http://www.fosd.de/SPLConqueror/). RSFIN has been evaluated on 11 real-world configurable software systems:

<table>
    <thead>
        <tr>
            <th align="center">System Under Prediction</th>
            <th align="center">Domain</th>
            <th align="center">Language</th>
            <th align="center">Number of binary configuration options</th>
            <th align="center">Number of numeric configuration options</th>
            <th align="center">Number of measured configurations</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td align="center">x264</td>
            <td align="center">Video Encoder</td>
            <td align="center">C</td>
            <td align="center">16</td>
            <td align="center">0</td>
            <td align="center">1152</td>
        </tr>
        <tr>
            <td align="center">SQLite</td>
            <td align="center">Database</td>
            <td align="center">C</td>
            <td align="center">39</td>
            <td align="center">0</td>
            <td align="center">4653</td>
        </tr>
        <tr>
            <td align="center">LLVM</td>
            <td align="center">Compiler</td>
            <td align="center">C++</td>
            <td align="center">11</td>
            <td align="center">0</td>
            <td align="center">1024</td>
        </tr>
        <tr>
            <td align="center">Apache</td>
            <td align="center">Web Server</td>
            <td align="center">C</td>
            <td align="center">9</td>
            <td align="center">0</td>
            <td align="center">192</td>
        </tr>
        <tr>
            <td align="center">BDB-C</td>
            <td align="center">Database</td>
            <td align="center">C</td>
            <td align="center">18</td>
            <td align="center">0</td>
            <td align="center">2560</td>
        </tr>
        <tr>
            <td align="center">BDB-J</td>
            <td align="center">Database</td>
            <td align="center">Java</td>
            <td align="center">26</td>
            <td align="center">0</td>
            <td align="center">180</td>
        </tr>
        <tr>
            <td align="center">HIPAcc</td>
            <td align="center">Image Processing</td>
            <td align="center">C++</td>
            <td align="center">31</td>
            <td align="center">2</td>
            <td align="center">13485</td>
        </tr>
        <tr>
            <td align="center">HSMGP</td>
            <td align="center">Stencil-Grid Solver</td>
            <td align="center">C++</td>
            <td align="center">11</td>
            <td align="center">3</td>
            <td align="center">3456</td>
        </tr>
        <tr>
            <td align="center">DUNE MGS</td>
            <td align="center">Muiti-Grid Solver</td>
            <td align="center">C++</td>
            <td align="center">8</td>
            <td align="center">3</td>
            <td align="center">2304</td>
        </tr>
        <tr>
            <td align="center">JavaGC</td>
            <td align="center">Runtime Environment</td>
            <td align="center">Java</td>
            <td align="center">12</td>
            <td align="center">23</td>
            <td align="center">10^31</td>
        </tr>
        <tr>
            <td align="center">Sac</td>
            <td align="center">Compiler</td>
            <td align="center">C</td>
            <td align="center">53</td>
            <td align="center">7</td>
            <td align="center">10^23</td>
        </tr>
    </tbody>
</table>

# Usage

To run RSFIN, users need to prepare before the evaluation and then run the script `main.py`. For details, users can refer to the experimental setup of __x264__ (__SystemUnderPrediction__), including the following:

- Import data file to: /data/ __SystemUnderPrediction__.csv
  - The data should be $n+1$ columns, where $n$ is the number of configuration options, and the $n+1$-th column is the performance (label).
  - The first row in the table is the header, and each remaining row is a piece of measured data.

Specifically, for software systems under prediction, RSFIN will run with three different sample sizes and 30 experiments for each sample size. For example, if users want to evaluate RSFIN with with 5n pieces of data from the system Apache, the  modification of lines 137-139 in `main.py` will be:

```
SYSTEM = 'Apache'
k = 2.5 
PATH = 'data/' + SYSTEM + '.csv'
```

where k represents the size of training set/verification set (multiple of N). For example, when k=2.5, the size of training set and verification set are both 2.5N, so a total of 5N samples are required.

The time cost of tuning for each experiment ranges from 20-200 minutes depending on the software system, the sample size, and the user's CPU. Typically, the time cost will be smaller when the software system has a smaller number of configurations or when the sample size is small. 

# Experimental Results
To evaluate the performance improvement, we use the ![](http://latex.codecogs.com/svg.latex?%5Crm{Impro}), which is computed as,

![](http://latex.codecogs.com/svg.latex?{%5Crm{MRE}}(C_{test},P_{test})=%5Cfrac{1}{\vert C_{test} \vert}\sum_{\mathbf{c}_0 \in C_{test}} \frac{\vert M(\mathbf{c}_0)-P_0 \vert}{P_0}%5Ctimes{100\%})

where, ![](http://latex.codecogs.com/svg.latex?C_{test}) represents the test configuration, and ![](http://latex.codecogs.com/svg.latex?P_{test}) is the corresponding performance. ![](http://latex.codecogs.com/svg.latex?M(\mathbf{c}_0)) is the predicted result of ![](http://latex.codecogs.com/svg.latex?\mathbf{c}_0), and ![](http://latex.codecogs.com/svg.latex?P_0) is the actual measurement of ![](http://latex.codecogs.com/svg.latex?\mathbf{c}_0).

In the table below, we use three different measurement constraints to evaluate the impact of measurement effort. The results are obtained when evaluating RSFIN on a Windows 10 computer with Intel![](http://latex.codecogs.com/svg.latex?%5CcircledR) Core![](http://latex.codecogs.com/svg.latex?^%5Ctext{TM}) i7-8700 CPU @ 3.20GHz 16GB RAM.



#### - Prediction accuracy for software systems with binary options

<table>
    <tr>
        <td rowspan="2">SUP(n)</td>
        <td rowspan="2">Sample Size</td>
        <td colspan="2">DECART</td>
        <td colspan="2">DeePerf</td>
        <td colspan="2">RSFIN</td>
    </tr>
    <tr>
        <td>Mean</td>
        <td>Margin</td>
        <td>Mean</td>
        <td>Margin</td>
        <td>Mean</td>
        <td>Margin</td>
    </tr>
    <tr><td rowspan="3">x264(n = 13)</td><td>n</td><td>12.83</td><td>1.86</td><td>10.43</td><td>2.28</td><td>10.70</td><td>1.25</td></tr>
    <tr><td>3n</td><td>7.44</td><td>0.27</td><td>2.13</td><td>0.31</td><td>1.88</td><td>0.07</td></tr>
    <tr><td>5n</td><td>6.91</td><td>0.28</td><td>0.87</td><td>0.11</td><td>1.05</td><td>0.04</td></tr>
    <tr><td rowspan="3">SQLite(n = 39)</td><td>n</td><td>4.79</td><td>0.04</td><td>5.04</td><td>0.32</td><td>4.65</td><td>0.26</td></tr>
    <tr><td>3n</td><td>4.43</td><td>0.09</td><td>4.48</td><td>0.08</td><td>4.44</td><td>0.07</td></tr>
    <tr><td>5n</td><td>4.12</td><td>0.05</td><td>4.27</td><td>0.13</td><td>4.31</td><td>0.04</td></tr>
    <tr><td rowspan="3">LLVM(n = 11)</td><td>n</td><td>6.52</td><td>0.23</td><td>5.09</td><td>0.80</td><td>4.65</td><td>0.53</td></tr>
    <tr><td>3n</td><td>3.81</td><td>0.15</td><td>2.54</td><td>0.15</td><td>2.37</td><td>0.07</td></tr>
    <tr><td>5n</td><td>3.63</td><td>0.21</td><td>1.99</td><td>0.15</td><td>2.27</td><td>0.03</td></tr>
    <tr><td rowspan="3">Apache(n = 9)</td><td>n</td><td>19.60</td><td>2.13</td><td>17.87</td><td>1.85</td><td>16.93</td><td>1.25</td></tr>
    <tr><td>3n</td><td>10.46</td><td>1.17</td><td>8.25</td><td>0.75</td><td>6.92</td><td>0.30</td></tr>
    <tr><td>5n</td><td>6.91</td><td>0.28</td><td>0.87</td><td>0.11</td><td>1.05</td><td>0.04</td></tr>
    <tr><td rowspan="3">BDB-C(n = 18)</td><td>n</td><td>128.83</td><td>26.72</td><td>133.60</td><td>54.33</td><td>85.74</td><td>30.12</td></tr>
    <tr><td>3n</td><td>90.71</td><td>5.85</td><td>13.10</td><td>3.39</td><td>10.89</td><td>1.21</td></tr>
    <tr><td>5n</td><td>65.98</td><td>3.74</td><td>5.82</td><td>1.33</td><td>4.92</td><td>1.44</td></tr>
    <tr><td rowspan="3">BDB-J(n = 26)</td><td>n</td><td>14.84</td><td>4.96</td><td>7.25</td><td>4.21</td><td>5.31</td><td>1.01</td></tr>
    <tr><td>3n</td><td>4.95</td><td>0.09</td><td>1.73</td><td>0.12</td><td>3.41</td><td>0.08</td></tr>
    <tr><td>5n</td><td>3.98</td><td>0.12</td><td>1.61</td><td>0.01</td><td>2.14</td><td>0.03</td></tr>
</table>

#### - Prediction accuracy for software systems with binary-numeric options

<table>
    <tr>
        <td rowspan="2">SUP(n)</td>
        <td rowspan="2">Sample Size</td>
        <td colspan="2">DeePerf</td>
        <td colspan="2">RSFIN</td>
    </tr>
    <tr>
        <td>Mean</td>
        <td>Margin</td>
        <td>Mean</td>
        <td>Margin</td>
    </tr>
    <tr><td rowspan="3">HIPAcc(n = 33)</td><td>10n</td><td>10.45</td><td>0.76</td><td>10.56</td><td>0.51</td></tr>
    <tr><td>30n</td><td>5.06</td><td>0.33</td><td>10.37</td><td>0.35</td></tr>
    <tr><td>50n</td><td>3.97</td><td>0.12</td><td>10.12</td><td>0.12</td></tr>
    <tr><td rowspan="3">HSMGP(n = 14)</td><td>10n</td><td>3.87</td><td>0.27</td><td>3.41</td><td>0.56</td></tr>
    <tr><td>30n</td><td>2.53</td><td>0.12</td><td>2.34</td><td>0.71</td></tr>
    <tr><td>50n</td><td>2.21</td><td>0.11</td><td>1.93</td><td>0.66</td></tr>
    <tr><td rowspan="3">DUNE MGS(n = 11)</td><td>10n</td><td>13.31</td><td>0.87</td><td>11.21</td><td>1.13</td></tr>
    <tr><td>30n</td><td>8.01</td><td>0.31</td><td>7.92</td><td>0.22</td></tr>
    <tr><td>50n</td><td>7.13</td><td>0.12</td><td>7.63</td><td>0.11</td></tr>
    <tr><td rowspan="3">JavaGC(n = 35)</td><td>10n</td><td>22.51</td><td>2.73</td><td>26.53</td><td>5.87</td></tr>
    <tr><td>30n</td><td>20.54</td><td>5.84</td><td>24.93</td><td>3.42</td></tr>
    <tr><td>50n</td><td>15.98</td><td>7.13</td><td>20.11</td><td>3.17</td></tr>
    <tr><td rowspan="3">SaC(n = 60)</td><td>10n</td><td>19.54</td><td>1.74</td><td>35.71</td><td>5.21</td></tr>
    <tr><td>30n</td><td>17.42</td><td>6.51</td><td>32.21</td><td>3.54</td></tr>
    <tr><td>50n</td><td>15.64</td><td>3.29</td><td>31.95</td><td>3.67</td></tr>
</table>

