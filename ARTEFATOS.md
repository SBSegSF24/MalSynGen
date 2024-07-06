## Ferramenta MalSynGen
A MalSynGen  é uma ferramenta que utiliza redes neurais artificiais para gerar dados sintéticos tabulares para o domínio de malware Android.
Para avaliar sua performance foram aumentados os dados de dois datasets, considerando métricas de fidelidade estatística e utilidade. 
Os resultados indicam que MalSynGen é capaz de capturar padrões representativos para o aumento de dados tabulares.



## Download do código fonte 
**Opção 1**: Para realizar a instalação do códgio fonte utilize o comando:
   ```bash
git clone https://github.com/MalwareDataLab/SynTabData.git
  ```
**Opção 2**: Acesse o repositório diretamente através do link https://github.com/SBSegSF24/MalSynGen e faça o download.
Acesse o diretorio da ferramenta:
  ```bash
cd MalSynGen/
 ```

## Dependências
O código da MalSynGen possui dependências com diversos pacotes e bibliotecas Python.
No entanto, as principais são:
numpy 1.21.5, Keras 2.9.0, Tensorflow 2.9.1, pandas 1.4.4, scikit-learn 1.1.1. e mlflow 2.12.1.
Ademais, a lista extensa das dependências encontra-se no arquivo requirements.txt.


## Pré-configuração
Antes de executar a ferramenta é necessário a execução dos seguintes comandos para resolução de dependências e instanciamento do ambiente.
### Configurar pipenv
```
pip install pipenv
```
```
pipenv install -r requirements.txt

```


## Fluxo de execução

![Screenshot from 2024-07-05 17-00-39](https://github.com/SBSegSF24/MalSynGen/assets/174879323/71ed6b27-8273-4e4f-a57f-fe717d9b26fd)

O fluxo de execução da ferramenta consiste de três etapas:

**Seleção de dataset**: Nesta etapa,  realizamos o balanceamento pela classe minoritária, atravẽs do uso de técnicas de subamostragem. Os datasets balanceados e o código utilizado nesse processo se encontram em: https://github.com/MalwareDataLab/SynTabData/tree/87f5018d6acdbe79eb91563c34eb428f36c19a7a/datasets

 O dataset balanceado é então processado nas etapas de treinamento e avaliação, através validação cruzada por meio de k-dobras (do inglês k-folds) onde são criados dois subconjuntos: subconjunto de avaliação (Dataset r) e subconjunto de treino (Dataset R)

  **Treinamento**: Nesta etapa ocorre o treinamento da cGAN e geração de dois conjuntos de dados sintéticos: Dataset S (gerado a partir de R) e Dataset s (gerado a paritr de r). Após a geração dos dados são treinados dois classificadores TR-Ar(Treinado com dataset R, avaliado com r) e TS-Ar(Treinado com S, avaliado com r).

   **Avaliação**: Consiste da coleta de métricas de fidelidade e utilidade dos classificadores e dados sintéticos. E subsequente, aplicação  de uma análise estática utilizando o teste wilcoxon sobre as métricas após o término da execução em dobras.


## Datasets
O diretório datasets contem  os datasets balanceados KronoDroid_emulator e KronoDroid_real_device[^1] utilizados no artigo, assim
como o código utilizado para balancear estes datasets. Além dos arquivos de validação de cada dataset e código de validação utilizado no subdiretório validation.
[^1]: https://github.com/aleguma/kronodroid


## Ambiente de testes
Utilizamos um servidor AMD Ryzen 7 5800x como processador de 8 cores e 64 GB de memória RAM para execução dos experimentos. 
O sistema operacional do servidor é o Ubuntu Server versão 22.04.

## Parâmetros da Ferramenta
|       Flag/ parametro       |                                  Descrição                                 | Obrigatório |
|:---------------------------:|:--------------------------------------------------------------------------:|:-----------:|
|     -i , --input_dataset    |              Caminho para o arquivo do dataset real de entrada             |     Sim     |
|       -o, --output_dir      |               Diretório para gravação dos arquivos de saída.               |     Não     |
|         --data_type         |       Tipo de dado para representar as características das amostras.       |     Não     |
| --num_samples_class_malware |                  Número de amostras da Classe 1 (maligno).                 |     Sim     |
|  --num_samples_class_benign |                  Número de amostras da Classe 0 (benigno).                 |     Sim     |
|       --number_epochs       |            Número de épocas (iterações de treinamento) da cGAN.            |     Não     |
|           --k_fold          |                 Número de subdivisões da validação cruzada                 |     Não     |
|      --initializer_mean     |          Valor central da distribuição gaussiana do inicializador.         |     Não     |
|   --initializer_deviation   |          Desvio padrão da distribuição gaussiana do inicializador.         |     Não     |
|      --latent_dimension     |              Dimensão do espaço latente para treinamento cGAN.             |     Não     |
|     --training_algorithm    | Algoritmo de treinamento para cGAN. Opções: 'Adam', 'RMSprop', 'Adadelta'. |     Não     |
|    --activation_function    |      Função de ativação da cGAN. Opções: 'LeakyReLU', 'ReLU', 'PReLU'      |     Não     |
|    --dropout_decay_rate_g   |              Taxa de decaimento do dropout do gerador da cGAN.             |     Não     |
|    --dropout_decay_rate_d   |           Taxa de decaimento do dropout do discriminador da cGAN.          |     Não     |
|    --dense_layer_sizes_g    |                   Valores das camadas densas do gerador.                   |     Não     |
|    --dense_layer_sizes_d    |                Valores das camadas densas do discriminador.                |     Não     |
|         --batch_size        |                          Tamanho do lote da cGAN.                          |     Não     |
|         --verbosity         |                            Nível de verbosidade.                           |     Não     |
|        --save_models        |                    Opção para salvar modelos treinados.                    |     Não     |
|   --path_confusion_matrix   |                Diretório de saída das matrizes de confusão.                |     Não     |
|      --path_curve_loss      |          Diretório de saída dos gráficos de curva de treinamento.          |     Não     |
|        -a, --use_aim        |         Opção para utilizar a ferramenta de rastreamento Aimstack.         |     Não     |
|      -ml, --use_mlflow      |          Opção para utilizar a ferramenta de rastreamento mlflow.          |     Não     |
|        -rid, --run_id       |  Opção ligado ao mlflow, utilizada para resumir uma execução não terminada |     Não     |
|    -tb, --use_tensorboard   |          Opção para utilizar a ferramenta de rastreamento Tensorb          |     Não     |
## Reprodução de experimentos 
Para a reprodução dos experimentos executados no artigo utilize os seguintes comandos:
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples\_class_malware 10000 --batch\_size 256 --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500--k_fold 10
pipenv run python3 main.py -i datasets/kronodroid_real_emulador-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500--k_fold 10
```
Alternativamente executar:
```bash
pipenv run python3 run_campaign.py -c SF24_4096_2048_1    
```
### Selos Reivindicados
Os autores julgam como considerados no processo de avaliação os selos:

**Selo D - Artefatos Disponíveis** 
Justificativa: Repositório anônimo disponível publicamente.

**Selo F - Artefatos Funcionais**
Justificativa: Artefatos completamente funcionais.

**Selo R - Artefatos Reprodutíveis**
Justificativa: Scripts disponibilizados reproduzem as configurações exatas do experimento

**Selo S- Artefatos Sustentáveis**
Justificativa: Código com boa documentação.
