## Model
- f<sub>true</sub>(x) = &alpha;<sub>mot</sub>·sin(&beta;<sub>mot</sub>x+&phi;<sub>mot</sub>)+a<sub>mot</sub>
- h<sub>true</sub>(x) = &alpha;<sub>obs</sub>·(&beta;<sub>obs</sub>x+a<sub>obs</sub>)<sup>2

## Model parameters
|Parameters             |Generative | Model 
|:-----------------:|:-------------:|:--------------:
|&alpha;<sub>mot</sub>  | 0.9       |1
|&beta;<sub>mot</sub>   | 1.1       |1
|&phi;<sub>mot</sub>    | &pi;/10   |0
|a<sub>mot</sub>        | 0.01      |0
|&alpha;<sub>obs</sub>  | 1         |1
|&beta;<sub>obs</sub>   | 1         |1
|a<sub>obs</sub>        | 0         |0
|Q(diagonal)            |0.01·eye   |0.01·eye          
|R(diagonal)            |0.01·eye   |0.01·eye

## Initial conditions
- x<sub>0</sub> = [0.1 0.1] (to avoid no evolution error)
- &Sigma;<sub>0</sub> = [[0,0],[0,0]]

## Time sequence length
- T<sub>train</sub> = 10
- T<sub>test</sub> = 100