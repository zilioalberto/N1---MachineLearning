# N1 - Machine Learning - Comitê de Classificadores

## Alunos
- Alberto Zilio  
- Lucas Carvalho Esteffens  
- Roni Pereira  

---

## 1. Introdução
O reconhecimento de padrões é um dos principais objetivos da área de *Machine Learning*. A tarefa de **classificação supervisionada** consiste em duas etapas principais:  
1. **Treinamento** — o modelo aprende padrões a partir de exemplos rotulados.  
2. **Teste** — o modelo aplica o conhecimento aprendido para prever classes de novas instâncias.  

Nesta atividade (N1), o objetivo é criar um **comitê de classificadores** aplicando **ao menos dois algoritmos de ML** sobre um dataset real, comparar os resultados e avaliar o desempenho por métricas como **acurácia, erro, matriz de confusão, precisão, recall, F1-score e curva ROC**.

---

## 2. Dataset Escolhido: *Traffic Flow Prediction Dataset*
O dataset selecionado é o **Traffic Flow Prediction Dataset**, que tem como objetivo prever o volume de tráfego em rodovias utilizando dados históricos, temporais e contextuais.

### Estrutura geral
- **36 sensores** distribuídos em duas rodovias na região de Washington D.C./Virgínia.  
- Dados coletados a cada **15 minutos**.  
- O objetivo é prever o **volume de tráfego 15 minutos à frente**.  

### Conjuntos disponíveis
- **Treino:**  
  - `tra_X_tr`: 1261 intervalos de 15 min, cada um contendo uma matriz (36 × 48).  
  - `tra_Y_tr`: matriz (36 × 1261) com o volume real de tráfego.  
- **Teste:**  
  - `tra_X_te`: 840 intervalos de 15 min (36 × 48).  
  - `tra_Y_te`: matriz (36 × 840) com o volume real de tráfego.  
- **Matriz de adjacência (`tra_adj_mat`):** 36 × 36, representando a conectividade espacial entre os sensores.

### Estrutura das 48 features
Cada sensor possui 48 atributos de entrada:
- **f0–f9:** Últimos 10 volumes de tráfego (lags temporais).  
- **f10–f16:** Codificação one-hot do dia da semana.  
- **f17–f40:** Codificação one-hot da hora do dia (0h–23h).  
- **f41–f44:** Codificação one-hot da direção da estrada.  
- **f45:** Número de faixas da estrada.  
- **f46:** Identificador/nome da estrada.  

### Exemplo
Para um sensor em um instante `t`:
```
[120, 135, 98, 210, 87, 65, 145, 175, 90, 130, 0, 0, 1, ...]
Target (Y): 142 veículos no próximo intervalo de 15 min
```

---

## 3. Metodologia
1. **Pré-processamento dos dados**  
   - Conversão do alvo contínuo (volume de tráfego) em **classes categóricas** (ex.: baixo, médio, alto tráfego).  
   - Normalização dos atributos numéricos.  

2. **Seleção de algoritmos**  
   - **KNN (K-Nearest Neighbors)** — classificador baseado na proximidade entre amostras.  
   - **Árvore de Decisão** — classificador simbólico que cria regras interpretáveis.  
   - (Opcional) Outros algoritmos como SVM ou Naive Bayes podem ser adicionados para comparação.  

3. **Treinamento e teste**  
   - Aplicação dos algoritmos nos conjuntos `tra_X_tr` e `tra_X_te`.  
   - Previsão das classes no conjunto de teste.  

4. **Métricas de avaliação**  
   - Acurácia.  
   - Taxa de erro.  
   - Matriz de confusão.  
   - Precisão, Recall e F1-score.  
   - Curva ROC comparando classificadores.

---

## 4. Resultados (a preencher)
- **Acurácia dos classificadores:**  
  - KNN: XX%  
  - Árvore de Decisão: YY%  

- **Matriz de Confusão:**  
  *(inserir tabelas geradas no Colab)*  

- **Curvas ROC:**  
  *(inserir gráfico comparativo dos classificadores)*  

---

## 5. Conclusão
- O dataset de tráfego se mostrou **rico em variáveis temporais e contextuais**.  
- A transformação do problema de regressão em **classificação** permitiu aplicar algoritmos de *Machine Learning* conforme solicitado na N1.  
- O comparativo entre classificadores mostrou que:  
  - O **KNN** é eficiente para capturar padrões locais de tráfego.  
  - A **Árvore de Decisão** oferece melhor interpretabilidade das regras de classificação.  
- O comitê de classificadores fornece uma visão mais robusta, permitindo escolher o algoritmo mais adequado em cenários reais de previsão de tráfego.
