
---

## 3. Metodologia
1. **Pré-processamento dos dados**  
   - Conversão do alvo contínuo (volume de tráfego) em **classes categóricas** (baixo, médio, alto tráfego) a partir de **tercis**.  
   - Normalização dos atributos numéricos.  

2. **Seleção de algoritmos**  
   - **KNN (K-Nearest Neighbors)** — classificador baseado na proximidade entre amostras.  
   - **Árvore de Decisão** — classificador simbólico que cria regras interpretáveis.  
   - (Opcional) Outros algoritmos como SVM ou Naive Bayes podem ser adicionados para comparação.  

3. **Treinamento e teste**  
   - Divisão 70/30 entre treino e teste após unificação do dataset.  
   - Aplicação dos algoritmos e comparação de desempenho.  

4. **Métricas de avaliação**  
   - Acurácia.  
   - Taxa de erro.  
   - Matriz de confusão.  
   - Precisão, Recall e F1-score.  
   - Curva ROC comparando classificadores.

---

## 4. Resultados

### 4.1 KNN – Baseline
- Configuração: **k=5, distância Euclidiana (p=2), pesos uniformes**.  
- **Acurácia:** 92,3%  
- Classe 0 (baixo tráfego): precisão/recall ≈ 95%.  
- Classe 1 (médio tráfego): desempenho inferior (≈ 88% recall).  
- Classe 2 (alto tráfego): bom desempenho (≈ 93%).  

**Matriz de Confusão – Baseline**  
![baseline_cm](baseline_cm.png)

➡️ O modelo acerta muito bem as classes 0 e 2, mas confunde a classe 1 (tráfego médio) com as vizinhas.

---

### 4.2 KNN – Melhor Configuração
- Configuração escolhida após teste de hiperparâmetros:  
  **k=7, distância Manhattan (p=1), pesos = distance**.  
- **Acurácia:** 92,7%  
- Classe 1 apresentou **melhoria no F1-score** (de 0.886 para 0.892).  

**Matriz de Confusão – Melhor KNN**  
![best_cm](best_cm.png)

➡️ Houve redução nos erros da classe intermediária, tornando o modelo mais equilibrado.

---

### 4.3 Comparações Visuais

**Curva ROC (macro-average)**  
![roc_comparison](roc_comparison.png)  
- Ambas as versões apresentam AUC elevado (~0.97–0.98).  
- O modelo otimizado (k=7, Manhattan) apresenta leve ganho na classe 1.  

**Curva Precision–Recall (macro-average)**  
![pr_comparison](pr_comparison.png)  
- O modelo otimizado mostra maior equilíbrio entre precisão e recall, especialmente na classe 1.  

**Métricas por Classe (Precision, Recall, F1)**  
![bars_metrics](bars_metrics.png)  
- Classe 0 e 2: mantêm desempenho muito alto.  
- Classe 1: melhoria perceptível no modelo otimizado.  

**Resumo: Acurácia e F1-macro**  
![acc_f1](acc_f1.png)  
- Baseline: Accuracy = 0.923, F1_macro = 0.923.  
- Melhor KNN: Accuracy = 0.927, F1_macro = 0.927.  

---

## 5. Conclusão
- O dataset de tráfego se mostrou **rico em variáveis temporais e contextuais**.  
- A transformação do problema de regressão em **classificação** permitiu aplicar algoritmos de *Machine Learning* conforme solicitado.  
- O modelo **KNN baseline** já apresentava excelente desempenho (92,3%).  
- A busca de hiperparâmetros resultou em um **modelo otimizado (k=7, Manhattan, distance)** com leve ganho global e melhoria importante na **classe intermediária (tráfego médio)**.  
- Os gráficos confirmam que a maior dificuldade é separar o **tráfego médio** das outras categorias, o que faz sentido do ponto de vista prático.  
- A próxima etapa será comparar os resultados do KNN com a **Árvore de Decisão**, compondo o **comitê de classificadores**.

---
