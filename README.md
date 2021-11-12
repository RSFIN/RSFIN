# RSFIN
Rule Search-based Fuzzy Inference Network

An end-to-end model

Debug in **main.m** (RSFIN) and **main2** (TwoLayer_RSFIN)

* matlab version: 2017a

* Statistics of specific experimental results (30 times) view in **prediction_result**. The header is for example as follows: 

| Number | MRE (in %) | Number of Sampling | Number of Rule | Time |
| ------ | ---------- | ------------------ | -------------- | ---- |
|        |            |                    |                |      |



# Result of Comparison Algorithm

To evaluate the prediction accuracy, we use the mean relative error (MRE), which is computed as,

```text
![1](http://latex.codecogs.com/svg.latex?MRE(C_{test},P_{test})=\frac{1}{|C_{test}|}\sum_{\textbf{c}_0\in C_{test}}\frac{|M(\textbf{c}_0)-P_0|}{P_0}\times 100)
```

where $C_{test}$ is the testing dataset, $M(\textbf{c}_0)$ is the predicted performance value of configuration $c$ generated using the model, $P_0$ is the actual performance value of configuration $c_0$. In the two tables below, Mean is the mean of the MREs seen in 30 experiments and Margin is the margin of the 95% confidence interval of the MREs in the 30 experiments. The results are obtained when evaluating RSFIN, DECART, SPLConqueror, DeepPerf on the Windows with Intel® CoreTMi7-8700 CPU @ 3.20GHz3.19GHz.

### Prediction accuracy for software systems (RSFIN, DECART, DeepPerf)

<table>
    <tr><td rowspan="2">SUP(EC)</td><td rowspan="2">Algorithm</td><td colspan="2">Effort</td><td colspan="2">MRE(in %)</td><td rowspan="2">Rank</td></tr>
    <tr><td>Sample Size</td><td>Time (in s)</td><td>Mean</td><td>Margin</td></tr>
<tr><td rowspan="5">x264(𝐸𝐶=80)</td></tr>
<tr><td>DECART</td><td>64</td><td>1.31</td><td>3.02</td><td>0.28</td><td>3rd</td></tr>
<tr><td>DeepPerf</td><td>64</td><td>221.77</td><td>1.49</td><td>0.38</td><td>1st</td></tr>
<tr><td>RSFIN</td><td>64</td><td>33.5</td><td>1.7</td><td>0.04</td><td>2nd</td></tr>
    <tr><td>Two-layer RSFIN</td><td>64</td><td>51.21</td><td>1.88</td><td>0.07</td><td>2nd</td></tr>
    <tr><td rowspan="5">SQLite(𝐸𝐶=195)</td></tr>
<tr><td>DECART</td><td>156</td><td>2.11</td><td>4.07</td><td>0.06</td><td>1st</td></tr>
<tr><td>DeepPerf</td><td>156</td><td>325.04</td><td>4.4</td><td>0.14</td><td>2nd</td></tr>
<tr><td>RSFIN</td><td>156</td><td>30.76</td><td>4.5</td><td>0.21</td><td>2nd</td></tr>
<tr><td>Two-layer RSFIN</td><td>156</td><td>45.37</td><td>4.51</td><td>0.11</td><td>2nd</td></tr>
<tr><td rowspan="5">LLVM(𝐸C=55)</td></tr>
<tr><td>DECART</td><td>44</td><td>1.75</td><td>2.67</td><td>0.17</td><td>2nd</td></tr>
<tr><td>DeepPerf</td><td>44</td><td>209.91</td><td>2.27</td><td>0.16</td><td>1st</td></tr>
<tr><td>RSFIN</td><td>44</td><td>2.14</td><td>2.58</td><td>0.06</td><td>1st</td></tr>
<tr><td>Two-layer RSFIN</td><td>44</td><td>11.37</td><td>2.53</td><td>0.09</td><td>1st</td></tr>
<tr><td rowspan="5">Apache(𝐸𝐶=45)</td></tr>
<tr><td>DECART</td><td>36</td><td>1.32</td><td>8.96</td><td>0.87</td><td>4th</td></tr>
<tr><td>DeepPerf</td><td>36</td><td>196.35</td><td>6.97</td><td>0.39</td><td>3rd</td></tr>
<tr><td>RSFIN</td><td>36</td><td>7.26</td><td>5.58</td><td>0.09</td><td>2nd</td></tr>
<tr><td>Two-layer RSFIN</td><td>36</td><td>5.17</td><td>4.86</td><td>0.19</td><td>1st</td></tr>
<tr><td rowspan="5">BDB-C(𝐸𝐶=90)</td></tr>
<tr><td>DECART</td><td>72</td><td>1.92</td><td>6.19</td><td>0.89</td><td>2nd</td></tr>
<tr><td>DeepPerf</td><td>72</td><td>236.59</td><td>6.95</td><td>1.11</td><td>2nd</td></tr>
<tr><td>RSFIN</td><td>72</td><td>36.945</td><td>11.59</td><td>1.04</td><td>3rd</td></tr>
<tr><td>Two-layer RSFIN</td><td>72</td><td>53.34</td><td>3.16</td><td>0.19</td><td>1st</td></tr>
<tr><td rowspan="5">BDB-J(𝐸𝐶=130)</td></tr>
<tr><td>DECART</td><td>104</td><td>1.87</td><td>1.62</td><td>0.07</td><td>2nd</td></tr>
<tr><td>DeepPerf</td><td>104</td><td>333.94</td><td>1.67</td><td>0.12</td><td>2nd</td></tr>
<tr><td>RSFIN</td><td>104</td><td>13.47</td><td>2.37</td><td>0.1</td><td>3rd</td></tr>
<tr><td>Two-layer RSFIN</td><td>104</td><td>14.49</td><td>1.45</td><td>0.05</td><td>1st</td></tr>
<tr><td rowspan="4">HIPA𝑐𝑐(𝐸𝐶=495)</td></tr>
<tr><td>DeepPerf</td><td>465</td><td>419.4</td><td>7.01</td><td>0.77</td><td>1st</td></tr>
<tr><td>RSFIN</td><td>465</td><td>47.48</td><td>13.04</td><td>1.16</td><td>3rd</td></tr>
<tr><td>Two-layer RSFIN</td><td>465</td><td>65.27</td><td>12.41</td><td>1.5</td><td>2nd</td></tr>
<tr><td rowspan="4">HSMGP(𝐸𝐶=210)</td></tr>
<tr><td>DeepPerf</td><td>144</td><td>176.78</td><td>3.87</td><td>0.18</td><td>2nd</td></tr>
<tr><td>RSFIN</td><td>144</td><td>17.01</td><td>4.95</td><td>0.91</td><td>3rd</td></tr>
<tr><td>Two-layer RSFIN</td><td>144</td><td>46.89</td><td>3.28</td><td>0.22</td><td>1st</td></tr>
<tr><td rowspan="4">Dune MGS(𝐸𝐶=165)</td></tr>
<tr><td>DeepPerf</td><td>72</td><td>162.89</td><td>14.31</td><td>0.93</td><td>2nd</td></tr>
<tr><td>RSFIN</td><td>72</td><td>6.347</td><td>11.5</td><td>0.33</td><td>1st</td></tr>
<tr><td>Two-layer RSFIN</td><td>72</td><td>24.62</td><td>11.43</td><td>0.07</td><td>1st</td></tr>
</table>

* Rank is derivedfrom the Kruskal-Wallison test on 30 MREs with significant level 0.05 (the same rank indicates that the hypothesis cannot be rejected by the statistical test).

### When the number of EC measurements is given, the execution effect of other methods

#### · Prediction accuracy for software systems with binary options

<table>
    <tr>
        <td rowspan="2">SUP</td>
        <td rowspan="2">Sample Size</td>
        <td colspan="2">DECART</td>
        <td colspan="2">DeePerf</td>
    </tr>
    <tr>
        <td>Mean</td>
        <td>Margin</td>
        <td>Mean</td>
        <td>Margin</td>
    </tr>
    <tr>
        <td rowspan="5">Apache(n = 9)</td>
        <td>n</td>
        <td>NA</td>
        <td>NA</td>
        <td>17.87</td>
        <td>1.85</td>
    </tr>
    <tr>
        <td>2n</td>
        <td>15.83</td>
        <td>2.89</td>
        <td>10.24</td>
        <td>1.15</td>
    </tr>
    <tr>
        <td>3n</td>
        <td>11.03</td>
        <td>1.46</td>
        <td>8.25</td>
        <td>0.75</td>
    </tr>
    <tr>
        <td>4n</td>
        <td>9.49</td>
        <td>1.00</td>
        <td>6.97</td>
        <td>0.39</td>
    </tr>
    <tr><td>5n</td><td>7.84</td><td>0.28</td><td>6.29</td><td>0.44</td></tr>
    <tr><td rowspan="5">x264(n = 16)</td><td>n</td><td>17.71</td><td>3.87</td><td>10.43</td><td>2.28</td></tr>
<tr><td>2n</td><td>9.31</td><td>1.30</td><td>3.61</td><td>0.54</td></tr>
<tr><td>3n</td><td>6.37</td><td>0.83</td><td>2.13</td><td>0.31</td></tr>
<tr><td>4n</td><td>4.26</td><td>0.47</td><td>1.49</td><td>0.38</td></tr>
<tr><td>5n</td><td>2.94</td><td>0.52</td><td>0.87</td><td>0.11</td></tr>
<tr><td rowspan="5">BDBJ(n = 26)</td><td>n</td><td>10.04</td><td>4.67</td><td>7.25</td><td>4.21</td></tr>
<tr><td>2n</td><td>2.23</td><td>0.16</td><td>2.07</td><td>0.32</td></tr>
<tr><td>3n</td><td>2.03</td><td>0.16</td><td>1.73</td><td>0.12</td></tr>
<tr><td>4n</td><td>1.72</td><td>0.09</td><td>1.67</td><td>0.12</td></tr>
<tr><td>5n</td><td>1.67</td><td>0.09</td><td>1.61</td><td>0.09</td></tr>
<tr><td rowspan="5">LLVM(n = 11)</td><td>n</td><td>6.00</td><td>0.34</td><td>5.09</td><td>0.80</td></tr>
<tr><td>2n</td><td>4.66</td><td>0.47</td><td>3.87</td><td>0.48</td></tr>
<tr><td>3n</td><td>3.96</td><td>0.39</td><td>2.54</td><td>0.15</td></tr>
<tr><td>4n</td><td>3.54</td><td>0.42</td><td>2.27</td><td>0.16</td></tr>
<tr><td>5n</td><td>2.84</td><td>0.33</td><td>1.99</td><td>0.15</td></tr>
<tr><td rowspan="5">BDBC(n = 18)</td><td>n</td><td>151.0</td><td>90.70</td><td>133.6</td><td>54.33</td></tr>
<tr><td>2n</td><td>43.8</td><td>26.72</td><td>16.77</td><td>2.25</td></tr>
<tr><td>3n</td><td>31.9</td><td>22.73</td><td>13.1</td><td>3.39</td></tr>
<tr><td>4n</td><td>6.93</td><td>1.39</td><td>6.95</td><td>1.11</td></tr>
<tr><td>5n</td><td>5.02</td><td>1.69</td><td>5.82</td><td>1.33</td></tr>
<tr><td rowspan="5">SQL(n = 39)</td><td>n</td><td>4.87</td><td>0.22</td><td>5.04</td><td>0.32</td></tr>
<tr><td>2n</td><td>4.67</td><td>0.17</td><td>4.63</td><td>0.13</td></tr>
<tr><td>3n</td><td>4.36</td><td>0.09</td><td>4.48</td><td>0.08</td></tr>
<tr><td>4n</td><td>4.21</td><td>0.1</td><td>4.40</td><td>0.14</td></tr>
<tr><td>5n</td><td>4.11</td><td>0.08</td><td>4.27</td><td>0.13</td></tr>
</table>
#### · Prediction accuracy for software systems with binary-numeric options

<table>
    <tr>
        <td rowspan="2">SUP</td>
        <td rowspan="2">Sample Size</td>
        <td colspan="2">SPLConqueror</td>
        <td colspan="3">DeePerf</td>
    </tr>
    <tr>
        <td>Sampling Heuristic</td>
        <td>Mean</td>
        <td>Sampling Heuristic</td>
        <td>Mean</td>
        <td>Margin</td>
    </tr>
<tr><td rowspan="4">Dune</td><td>49</td><td>OW RD</td><td>20.1</td><td>RD</td><td>15.73</td><td>0.90</td></tr>
<tr><td>78</td><td>PW RD</td><td>22.1</td><td>RD</td><td>13.67</td><td>0.82</td></tr>
<tr><td>240</td><td>OW PBD(49, 7)</td><td>10.6</td><td>RD</td><td>8.19</td><td>0.34</td></tr>
<tr><td>375</td><td>OW PBD(125, 5)</td><td>18.8</td><td>RD</td><td>7.20</td><td>0.17</td></tr>
<tr><td rowspan="4">hipacc</td><td>261</td><td>OW RD</td><td>14.2</td><td>RD</td><td>9.39</td><td>0.37</td></tr>
<tr><td>528</td><td>OW PBD(125, 5)</td><td>13.8</td><td>RD</td><td>6.38</td><td>0.44</td></tr>
<tr><td>736</td><td>OW PBD(49, 7)</td><td>13.9</td><td>RD</td><td>5.06</td><td>0.35</td></tr>
<tr><td>1281</td><td>PW RD</td><td>13.9</td><td>RD</td><td>3.75</td><td>0.26</td></tr>
<tr><td rowspan="4">hsmgp</td><td>77</td><td>OW RD</td><td>4.5</td><td>RD</td><td>6.76</td><td>0.87</td></tr>
<tr><td>173</td><td>PW RD</td><td>2.8</td><td>RD</td><td>3.60</td><td>0.2</td></tr>
<tr><td>384</td><td>OW PBD(49, 7)</td><td>2.2</td><td>RD</td><td>2.53</td><td>0.13</td></tr>
<tr><td>480</td><td>OW PBD(125, 5)</td><td>1.7</td><td>RD</td><td>2.24</td><td>0.11</td></tr>
<tr><td rowspan="4">javagc</td><td>423</td><td>OW PBD(49, 7)</td><td>37.4</td><td>RD</td><td>24.76</td><td>2.42</td></tr>
<tr><td>534</td><td>OW RD</td><td>31.3</td><td>RD</td><td>23.27</td><td>4.00</td></tr>
<tr><td>855</td><td>OW PBD(125, 5)</td><td>21.9</td><td>RD</td><td>21.83</td><td>7.07</td></tr>
<tr><td>2571</td><td>OW PBD(49, 7)</td><td>28.2</td><td>RD</td><td>17.32</td><td>7.89</td></tr>
<tr><td rowspan="4">sac</td><td>2060</td><td>OW RD</td><td>21.1</td><td>RD</td><td>15.83</td><td>1.25</td></tr>
<tr><td>2295</td><td>OW PBD(125, 5)</td><td>20.3</td><td>RD</td><td>17.95</td><td>5.63</td></tr>
<tr><td>2499</td><td>OW PBD(49, 7)</td><td>16</td><td>RD</td><td>17.13</td><td>2.22</td></tr>
<tr><td>3261</td><td>PW RD</td><td>30.7</td><td>RD</td><td>15.40</td><td>2.05</td></tr>
</table>
