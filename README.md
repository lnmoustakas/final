# Final Project PGE379 Code
![image](https://user-images.githubusercontent.com/41163936/204724617-465c45a0-4ef5-4039-a04b-cbd1945fbb61.png)
The code does two main things, firstly it can be used to calculate hyperbolic/exponential decline curves given well daily/monthly production data timeseries such as "21_01- 6.csv".
Secondly it can use a combination of well data and hyperbolic decline curves as a covariate in order to apply Darts deep learning models to improve the use of DCA as well production targets. </p>
![image](https://user-images.githubusercontent.com/41163936/204725605-42963c8a-7027-43aa-903d-e749db9b681e.png)
<br>
One line needs to be fixed (line 84) to use with multiple well data compiled in a single file where it can plot decline curves/deep learning predictions for a field of wells.
```
t = df.groupby(['well_id'], as_index=False).apply(lambda x: x['date'] - x['date'].iloc[0]).reset_index()
```
If you have any questions about it feel free to reach out to me at my email __lmoustakas@utexas.edu__
