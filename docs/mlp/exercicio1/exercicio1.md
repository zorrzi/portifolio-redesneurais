# **Exercício 1: MLP**

---

### Obejtivo: Calculo manual de um MLP
Nesta atividade, será implementado o cálculo manual completo de um MLP, abordando as seguintes etapas durante esse processo:

- **`Forward Pass`**: Compreender como os dados fluem através da rede, desde a entrada até a saída final
- **`Calculo da função de perda (MSE)`** 
- **`Backward Pass`**: Assimilar o algoritmo de backpropagation e como os gradientes são calculados e propagados
- **`Visualizar a Atualização de Parâmetros`**: Observar como os pesos e bias são ajustados através do **gradient descent**

## **Arquitetura da Rede:**
- **Entrada:** 2 features
- **Camada oculta:** 2 neurônios com ativação tanh
- **Saída:** 1 neurônio com ativação tanh
- **Função de perda:** Mean Squared Error (MSE)

---


## **1. Configuração Inicial e Dados**

Primeiro, vamos configurar todos os dados que serão utilizados no MLP:

**Entrada**  

\[
x = \begin{bmatrix} 0.5 & -0.2 \end{bmatrix}
\]

**Saída alvo** 

\[
y = 1.0
\]

---

**Pesos e Bias da 1ª camada (W^(1), b^(1))**  

\[
W^{(1)} =
\begin{bmatrix}
0.3 & -0.1 \\
0.2 & 0.4
\end{bmatrix}
\]

\[
b^{(1)} = 
\begin{bmatrix}
0.1 & -0.2
\end{bmatrix}
\]

---

**Pesos e Bias da 2ª camada (W^(2), b^(2))**  

\[
W^{(2)} = 
\begin{bmatrix}
0.5 & -0.3
\end{bmatrix}
\]

\[
b^{(2)} = 0.2
\]

---

**Taxa de aprendizado**  

\[
\eta = 0.1
\]


```python
import numpy as np

np.set_printoptions(precision=4, suppress=True)

x = np.array([0.5, -0.2])  # entrada
y = 1.0                    # saida alvo

# Pesos e bias da camada oculta
W1 = np.array([[0.3, -0.1],
               [0.2,  0.4]])
b1 = np.array([0.1, -0.2])

# Pesos e bias da camada de saida
W2 = np.array([0.5, -0.3])
b2 = 0.2

eta = 0.1
```


## **2. Forward Pass (Passe Adiante)**

Agora vamos implementar o passo a passo dos cálcuos matematicos para a execução do Forward Pass:




### **Passo 2.1: Calcular pré-ativações da camada oculta**
$$\mathbf{z}^{(1)} = \mathbf{W}^{(1)}\mathbf{x} + \mathbf{b}^{(1)}$$

Substituindo os valores:

\[
z^{(1)} =
\begin{bmatrix}
0.3 & -0.1 \\
0.2 & 0.4
\end{bmatrix}
\cdot
\begin{bmatrix}
0.5 \\
-0.2
\end{bmatrix}
+
\begin{bmatrix}
0.1 \\
-0.2
\end{bmatrix}
\]

Cálculo:

\[
z^{(1)} =
\begin{bmatrix}
(0.3 \cdot 0.5) + (-0.1 \cdot -0.2) + 0.1 \\
(0.2 \cdot 0.5) + (0.4 \cdot -0.2) - 0.2
\end{bmatrix}
=
\begin{bmatrix}
0.27 \\
-0.18
\end{bmatrix}
\]


```python
z1 = W1 @ x + b1

print(f"z^(1) = {z1}")
```

```
z^(1) = [ 0.27 -0.18]
```

### **Passo 2.2: Aplicar função de ativação tanh na camada oculta**
$$\mathbf{h}^{(1)} = \tanh(\mathbf{z}^{(1)})$$

Substituindo os valores:

\[
h^{(1)} = \tanh
\begin{bmatrix}
0.27 \\
-0.18
\end{bmatrix}
\]

Calculando elemento a elemento:

\[
h^{(1)} =
\begin{bmatrix}
\tanh(0.27) \\
\tanh(-0.18)
\end{bmatrix}
=
\begin{bmatrix}
0.2636 \\
-0.1781
\end{bmatrix}
\]

```python
h1 = np.tanh(z1)
print(f"h^(1) = {h1}")

```

```
h^(1) = [ 0.2636 -0.1781]
```


### **Passo 2.3: Calcular pré-ativação da camada de saída**
$$u^{(2)} = \mathbf{W}^{(2)}\mathbf{h}^{(1)} + b^{(2)}$$

Substituindo os valores:

\[
u^{(2)} =
\begin{bmatrix}
0.5 & -0.3
\end{bmatrix}
\cdot
\begin{bmatrix}
0.2636 \\
-0.1781
\end{bmatrix}
+ 0.2
\]

Cálculo:

\[
u^{(2)} = (0.5 \cdot 0.2636) + (-0.3 \cdot -0.1781) + 0.2
\]

\[
u^{(2)} \approx 0.3852
\]

```python
u2 = W2 @ h1 + b2
print(f"u^(2) = {u2}")
```

```
u^(2) = 0.38523667817130075
```

### **Passo 2.4: Calcular saída final**
$$\hat{y} = \tanh(u^{(2)})$$

Substituindo o valor calculado:

\[
\hat{y} = \tanh(0.3852)
\]

Resultado:

\[
\hat{y} \approx 0.3672
\]

```python
y_hat = np.tanh(u2)
print(f"ŷ = {y_hat}")

```

```
ŷ = 0.36724656264510797
```

## **3. Cálculo da Perda (Loss Calculation)**

Agora vamos calcular a função de perda Mean Squared Error (MSE):

$$L = \frac{1}{N}(y - \hat{y})^2$$

Como temos apenas uma amostra (N=1), a fórmula se simplifica para:

$$L = (y - \hat{y})^2$$

Substituindo os valores:

\[
L = (1.0 - 0.3672)^2
\]

\[
L = (0.6328)^2
\]

Resultado:

\[
L \approx 0.4004
\]

```python
L = (y - y_hat)**2
print(f"L = {L}")

```

```
L = 0.4003769124844312
```


## **4. Backward Pass (Backpropagation)**

Agora vamos implementar o backward pass para calcular todos os gradientes. Começamos pela derivada da perda em relação à saída:

### **Passo 4.1: Gradiente da perda em relação à saída**
$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$$

Note que usamos a derivada de $(y - \hat{y})^2 = (\hat{y} - y)^2$, que é $2(\hat{y} - y)$


Substituindo os valores:

\[
\frac{\partial L}{\partial \hat{y}} = 2(0.3672 - 1.0)
\]

\[
\frac{\partial L}{\partial \hat{y}} = 2(-0.6328)
\]

Resultado:

\[
\frac{\partial L}{\partial \hat{y}} \approx -1.2655
\]

```python
def tanh_dt(u):
    """Deriva de tanh: d/du tanh(u) = 1 - tanh²(u)"""
    return 1 - np.tanh(u)**2

dL_dy_hat = 2 * (y_hat - y)
print(f"∂L/∂ŷ = {dL_dy_hat}")

```

```
∂L/∂ŷ = -1.265506874709784
```


### **Passo 4.2: Gradiente em relação à pré-ativação da saída**
$$\frac{\partial L}{\partial u^{(2)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{d}{du^{(2)}}\tanh(u^{(2)}) = \frac{\partial L}{\partial \hat{y}} \cdot (1 - \tanh^2(u^{(2)}))$$


Sabemos que:

\[
\tanh'(z) = 1 - \tanh^2(z)
\]

Substituindo os valores:

\[
\frac{\partial L}{\partial u^{(2)}} =
-1.2655 \cdot \left( 1 - \tanh^2(0.3852) \right)
\]

\[
\frac{\partial L}{\partial u^{(2)}} =
-1.2655 \cdot (1 - 0.1349)
\]

\[
\frac{\partial L}{\partial u^{(2)}} =
-1.2655 \cdot 0.8651
\]

Resultado:

\[
\frac{\partial L}{\partial u^{(2)}} \approx -1.0948
\]

```python
dL_du2 = dL_dy_hat * tanh_dt(u2)
print(f"∂L/∂u^(2) = {dL_du2}")
```

```
∂L/∂u^(2) = -1.0948279147135995
```


### **Passo 4.3: Gradientes para a camada de saída**
Agora calculamos os gradientes para os pesos e bias da camada de saída:

### Gradiente em relação aos pesos \(W^{(2)}\)

\[
\frac{\partial L}{\partial W^{(2)}} =
\frac{\partial L}{\partial u^{(2)}} \cdot h^{(1)}
\]

Substituindo os valores:

\[
\frac{\partial L}{\partial W^{(2)}} =
-1.0948 \cdot 
\begin{bmatrix}
0.2636 & -0.1781
\end{bmatrix}
\]

\[
\frac{\partial L}{\partial W^{(2)}} =
\begin{bmatrix}
-0.2886 & 0.1950
\end{bmatrix}
\]

---

### Gradiente em relação ao viés \(b^{(2)}\)

\[
\frac{\partial L}{\partial b^{(2)}} =
\frac{\partial L}{\partial u^{(2)}}
\]

Substituindo o valor:

\[
\frac{\partial L}{\partial b^{(2)}} = -1.0948
\]

```python
dL_dW2 = dL_du2 * h1  # Gradiente para W^(2)
dL_db2 = dL_du2       # Gradiente para b^(2)

print(f"∂L/∂W^(2) = {dL_dW2}")
print(f"∂L/∂b^(2) = {dL_db2}")

```

```
∂L/∂W^(2) = [-0.2886  0.195 ]
∂L/∂b^(2) = -1.0948279147135995
```


### **Passo 4.4: Propagação para a camada oculta**
Agora precisamos propagar o erro de volta para a camada oculta:

$$\frac{\partial L}{\partial \mathbf{h}^{(1)}} = (\mathbf{W}^{(2)})^T \cdot \frac{\partial L}{\partial u^{(2)}}$$


Substituindo os valores:

\[
\frac{\partial L}{\partial h^{(1)}} =
\begin{bmatrix}
0.5 \\
-0.3
\end{bmatrix}
\cdot (-1.0948)
\]

\[
\frac{\partial L}{\partial h^{(1)}} =
\begin{bmatrix}
-0.5474 \\
0.3284
\end{bmatrix}
\]

```python
dL_dh1 = W2 * dL_du2
print(f"∂L/∂h^(1) = {dL_dh1}")
```

```
∂L/∂h^(1) = [-0.5474  0.3284]
```


### **Passo 4.5: Gradientes para a camada oculta**
Agora calculamos os gradientes em relação às pré-ativações da camada oculta:

$$\frac{\partial L}{\partial \mathbf{z}^{(1)}} = \frac{\partial L}{\partial \mathbf{h}^{(1)}} \odot \tanh'(\mathbf{z}^{(1)})$$

onde $\odot$ representa o produto elemento a elemento (Hadamard product).

Sabemos que:

\[
\tanh'(z) = 1 - \tanh^2(z)
\]

Substituindo os valores:

\[
\frac{\partial L}{\partial z^{(1)}} =
\begin{bmatrix}
-0.5474 & 0.3284
\end{bmatrix}
\odot
\left( 1 - \tanh^2
\begin{bmatrix}
0.27 & -0.18
\end{bmatrix} \right)
\]

\[
\frac{\partial L}{\partial z^{(1)}} =
\begin{bmatrix}
-0.5474 & 0.3284
\end{bmatrix}
\odot
\begin{bmatrix}
0.9305 & 0.9683
\end{bmatrix}
\]

\[
\frac{\partial L}{\partial z^{(1)}} =
\begin{bmatrix}
-0.5094 & 0.3180
\end{bmatrix}
\]


```python
dL_dz1 = dL_dh1 * tanh_dt(z1)
print(f"∂L/∂z^(1) = {dL_dz1}")
```

```
∂L/∂z^(1) = [-0.5094  0.318 ]
```


### **Passo 4.6: Gradientes finais para pesos e bias da camada oculta**
Finalmente, calculamos os gradientes para os pesos e bias da camada oculta:

### Gradiente em relação aos pesos \(W^{(1)}\)

\[
\frac{\partial L}{\partial W^{(1)}} =
\frac{\partial L}{\partial z^{(1)}} \otimes x^T
\]

Substituindo os valores:

\[
\frac{\partial L}{\partial W^{(1)}} =
\begin{bmatrix}
-0.5094 \\
0.3180
\end{bmatrix}
\otimes
\begin{bmatrix}
0.5 & -0.2
\end{bmatrix}
\]

\[
\frac{\partial L}{\partial W^{(1)}} =
\begin{bmatrix}
-0.2547 & 0.1019 \\
0.1590 & -0.0636
\end{bmatrix}
\]

---

### Gradiente em relação ao viés \(b^{(1)}\)

\[
\frac{\partial L}{\partial b^{(1)}} =
\frac{\partial L}{\partial z^{(1)}}
\]

\[
\frac{\partial L}{\partial b^{(1)}} =
\begin{bmatrix}
-0.5094 & 0.3180
\end{bmatrix}
\]


```python
dL_dW1 = np.outer(dL_dz1, x)  
dL_db1 = dL_dz1               

print(f"∂L/∂W^(1) = \n{dL_dW1}")
print(f"∂L/∂b^(1) = {dL_db1}")

```


```
∂L/∂W^(1) = 
[[-0.2547  0.1019]
 [ 0.159  -0.0636]]

 ∂L/∂b^(1) = [-0.5094  0.318 ]
```


## **5. Atualização dos Parâmetros**

Agora aplicamos o algoritmo de **gradient descent** para atualizar todos os parâmetros usando a taxa de aprendizagem η = 0.1:

$$\theta_{novo} = \theta_{antigo} - \eta \cdot \frac{\partial L}{\partial \theta}$$


**Parâmetros antes da atualização**

\[
W^{(2)} = 
\begin{bmatrix}
0.5 & -0.3
\end{bmatrix},
\quad
b^{(2)} = 0.2
\]

\[
W^{(1)} =
\begin{bmatrix}
0.3 & -0.1 \\
0.2 & \phantom{-}0.4
\end{bmatrix},
\quad
b^{(1)} =
\begin{bmatrix}
0.1 & -0.2
\end{bmatrix}
\]


---

### Atualização de \(W^{(2)}\)

\[
W^{(2)}_{\text{novo}} = W^{(2)} - \eta \,\frac{\partial L}{\partial W^{(2)}}
\]

\[
W^{(2)}_{\text{novo}} =
\begin{bmatrix}
0.5 & -0.3
\end{bmatrix}
- 0.1 \cdot
\begin{bmatrix}
-0.2886 & 0.1950
\end{bmatrix}
=
\begin{bmatrix}
0.5289 & -0.3195
\end{bmatrix}
\]

---

### Atualização de \(b^{(2)}\)

\[
b^{(2)}_{\text{novo}} = b^{(2)} - \eta \,\frac{\partial L}{\partial b^{(2)}}
= 0.2 - 0.1 \cdot (-1.0948279147135995)
= 0.30948279147136
\]

---

### Atualização de \(W^{(1)}\)

\[
W^{(1)}_{\text{novo}} = W^{(1)} - \eta \,\frac{\partial L}{\partial W^{(1)}}
\]

\[
W^{(1)}_{\text{novo}} =
\begin{bmatrix}
0.3 & -0.1 \\
0.2 & 0.4
\end{bmatrix}
- 0.1 \cdot
\begin{bmatrix}
-0.2547 & 0.1019 \\
\phantom{-}0.1590 & -0.0636
\end{bmatrix}
=
\begin{bmatrix}
0.3255 & -0.1102 \\
0.1841 & \phantom{-}0.4064
\end{bmatrix}
\]

---

### Atualização de \(b^{(1)}\)

\[
b^{(1)}_{\text{novo}} = b^{(1)} - \eta \,\frac{\partial L}{\partial b^{(1)}}
\]

\[
b^{(1)}_{\text{novo}} =
\begin{bmatrix}
0.1 & -0.2
\end{bmatrix}
- 0.1 \cdot
\begin{bmatrix}
-0.5094 & 0.3180
\end{bmatrix}
=
\begin{bmatrix}
0.1509 & -0.2318
\end{bmatrix}
\]

```python
W2_new = W2 - eta * dL_dW2
b2_new = b2 - eta * dL_db2
W1_new = W1 - eta * dL_dW1
b1_new = b1 - eta * dL_db1


print(f"W^(2)_novo = {W2_new}")
print(f"b^(2)_novo = {b2_new}")
print(f"W^(1)_novo = \n{W1_new}")
print(f"b^(1)_novo = {b1_new}")

```


```
W^(2)_novo = [ 0.5289 -0.3195]
b^(2)_novo = 0.30948279147136

W^(1)_novo = 
[[ 0.3255 -0.1102]
 [ 0.1841  0.4064]]

b^(1)_novo = [ 0.1509 -0.2318]



```


## **6. Verificação dos Resultados**

Vamos verificar se a atualização dos parâmetros realmente melhorou o desempenho da rede calculando um novo forward pass com os parâmetros atualizados:

Após atualizar os parâmetros, realizamos um novo **forward pass**:

\[
z^{(1)}_{\text{novo}} =
\begin{bmatrix}
0.3357 & -0.221
\end{bmatrix}
\]

\[
h^{(1)}_{\text{novo}} =
\begin{bmatrix}
0.3236 & -0.2175
\end{bmatrix}
\]

\[
u^{(2)}_{\text{novo}} = 0.5501
\]

\[
\hat{y}_{\text{novo}} = 0.5006
\]

\[
L_{\text{novo}} = 0.2494
\]

---


| Métrica        | Antes    | Depois   | Melhoria |
|----------------|----------|----------|----------|
| Saída \(\hat{y}\)   | 0.3672   | 0.5006   | Sim |
| Perda \(L\)        | 0.4004   | 0.2494   | Sim |
| Erro absoluto  | 0.6328   | 0.4994   | Sim |

---


Houve uma **redução da perda de 37.71%**, confirmando que a atualização dos parâmetros aproximou a saída predita do valor alvo.
