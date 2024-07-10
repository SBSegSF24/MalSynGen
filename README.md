## 1. Preparação e instalação

1. Clonar o repositório e execute os seguintes comandos.
   ```bash
    git clone https://github.com/SBSegSF24/MalSynGen.git
    cd MalSynGen
   ```
2. Instalação das dependências.
   
   **Opção 1**: Construir uma imagem Docker localmente a partir do Dockerfile.
      
      ```bash
      ./scripts/docker_build.sh
      ```
   **Opção 2**: Utilizar o script **pip_env_install.sh**.
      
      ```bash
   ./pip_env_install.sh
      ```

   **Opção 3**: Configurar o venv.
   ```
   python3 -m venv .venv
   ```
   ```
   source .venv/bin/activate
   ```
   ```
   pip3 install -r requirements.txt
   ```

   **Opção 4**: Configurar o pipenv.
   ```
   pip install pipenv
   ```
   ```
   pipenv install -r requirements.txt
   ```
## 2. Execução
1. Executar a demonstração de funcionamento da ferramenta: 

   **Opção 1**: instalar as dependências e executar a aplicação em um ambiente Linux.
   ```bash
   ./run_demo_app.sh
   ```
     

   **Opção 2**: Executar o script de execução da imagen Docker.
   
   ```bash
   sudo ./scripts/docker_run_solo.sh
   ```
   
   **Opção 3**: Executar o docker demo que instancia uma versão reduzida do experimento.
   ```bash
   ./run_demo_docker.sh
   ```

    
2. Executar os mesmos experimentos (campanhas) do paper.

   ```bash
    ./run_reproduce_sf24.sh
   ```
   
## 3. Outras opções de execução
   O script **run_balanced_datasets.sh** é responsável pela execução dos datasets balanceaddos dos experimentos com base na entrada especificada pelo usuário.
   Executar o script: 


   ```bash
   ./run_balanced_datasets.sh
   ```


#### 3.1. Executando outros experimentos

A ferramenta conta com o **run_campaign.py** para automatizar o treinamento e a avaliação da cGAN. O **run_campaign.py** permite executar várias campanhas de avaliação com diferentes parâmetros, registrando os resultados em arquivos de saída para análise posterior. O usuário poderá visualmente realizar uma análise comparativa das diferentes configurações em relação aos conjuntos de dados utilizados.

Exemplo de execução de uma campanha pré-configurada com base na execução do Kronodroid R do artigo:

```
pipenv run python3 run_campaign.py -c Kronodroid_r
```


Mesma campanha (Kronodroid_r) sendo executada diretamente na aplicação (**main.py**):
```
pipenv run python main.py --verbosity 20 --input_dataset datasets/kronodroid_real_device-balanced.csv --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500 --k_fold 10 --num_samples_class_benign 10000 --num_samples_class_malware 10000 --training_algorithm Adam
```
#### 3.2. Parâmetros dos testes automatizados:

      --------------------------------------------------------------

    --campaign ou -c:    Especifica a campanha de avaliação que você deseja executar. 
                         Você pode fornecer o nome de uma campanha específica ou uma  
                         lista de campanhas separadas por vírgula. 
                         Por exemplo: --campaign SF24_4096_2048_10 ou --campaign 
                          Kronodroid_e,kronodroid_r.

    --demo ou -d:
                         Ativa o modo demo. Quando presente, o script será executado 
                         no modo demo, o que pode ter comportamento reduzido 
                         ou exibir informações de teste.
                         --verbosity ou -v: Especifica o nível de verbosidade do log.
                         Pode ser INFO (1) ou DEBUG (2). 
                         Por padrão, o nível de verbosidade é definido como INFO.


     Outros parâmetros de entrada são definidos dentro das campanhas de avaliação em 
     campaigns_available. Cada campanha tem suas próprias configurações específicas, 
     como input_dataset, number_epochs, training_algorithm, dense_layer_sizes_g, 
     dense_layer_sizes_d, classifier, activation_function, dropout_decay_rate_g, 
     dropout_decay_rate_d, e data_type. As configurações podem variar dependendo do 
     objetivo e das configurações específicas de cada campanha.  


     Em campaigns_available o script irá iterar sobre as combinações de configurações 
     especificadas e executar os experimentos correspondentes.

    --------------------------------------------------------------


#### 3.3. Executando a ferramenta no Google Colab
Google collab é uma ferramenta cloud que permite a execução de códigos Python no seu navegador.

1. Acesse o seguinte link para utilizar a ferramenta Google colab: https://colab.google/
   
2. Crie um novo notebook, clicando no botão **New notebook** no topo direito da tela.
   
<td><img src= https://github.com/SBSegSF24/MalSynGen/assets/174879323/628010a5-f2e9-48dc-8044-178f7e9c2c37 style="max-width:100%;"></td>

3. Faça o upload da pasta do MalSynGen no seu Google Drive.

4. Adicione uma nova célula ao clicar no botão **+code** no topo esquerda da tela, contendo o seguinte  trecho de código para acessar a pasta do Google Drive.
```
from google.colab import drive
drive.mount('/content/drive')
```
5. Crie uma célula para acessar a pasta do MalSynGen (Exemplo):
```
cd /content/drive/MyDrive/MalSynGen-main
```
6. Instale as dependências da ferramenta, criando uma célula com o seguinte código:
```
!pip install -r requirements.txt
```
7. Crie uma célula para a execução da ferramenta (Exemplo):
```
!python main.py --verbosity 20 --input_dataset datasets/kronodroid_real_device-balanced.csv --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500 --k_fold 10 --num_samples_class_benign 10000 --num_samples_class_malware 10000 --training_algorithm Adam
```


## 4. Fluxo de execução 
![Screenshot from 2024-07-05 17-00-39](https://github.com/SBSegSF24/MalSynGen/assets/174879323/4d55117e-4203-4930-a0f5-2ed19c8857e3)

O fluxo de execução da ferramenta consiste de três etapas:

   **Seleção de dataset**: Nesta etapa,  realizamos o balanceamento pela classe minoritária, atravẽs do uso de técnicas de subamostragem. Os datasets balanceados e o código utilizado nesse processo se encontram em: https://github.com/SBSegSF24/MalSynGen/tree/accbe69f12bbf02d5d7f9c75291a60a5738bbb67/datasets

 O dataset balanceado é então processado nas etapas de treinamento e avaliação, através validação cruzada por meio de k-dobras (do inglês k-folds) onde são criados dois subconjuntos: subconjunto de avaliação (Dataset r) e subconjunto de treino (Dataset R).

  **Treinamento**: Nessa etapa, a cGANs é treinada  e utilizada cGANs para gerar dados sintéticos, precisamos treinar classificadores para posteriormente verificarmos a utilidade dos dados sintéticos gerados: Dataset S (gerado a partir de R) e Dataset s (gerado a paritr de r).  Os classificadores utilizados são denominados TR-As(Treinado com dataset R, avaliado com s) e TS-Ar(Treinado com S, avaliado com r).

   **Avaliação**: Esta etapa consiste da  execução e subsquente extração de métricas de utilidade dos classificadores, e fidelidade dos sintéticos atravês de uma comparação entre s e r. Por fim, verificamos se a utilidade dos dados sintéticos é fiel à utilidade dos dados reais através de testes como o de Wilcoxon no final da execução de dobras.





## 5. Parâmetros da Ferramenta
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

## 6. Ambientes de teste

A ferramenta foi executada e testada nos seguintes ambientes:


-**Hardware**:AMD Ryzen 7 5800x, 8 cores, 64 GB RAM. **Software**: Ubuntu Server 22.04.2 e 22.04.3, Python 3.8.10 e 3.10.12, Docker 24.07.

-**Hardware**:Intel Core i7-9700 CPU 3.00GHz, 8 cores, 16 GB RAM. **Software**: Debian GNU 11 e 12, Python 3.9.2 e 3.11.2, Docker 20.10.5 e 24.07.
Módulos Python conforme [requirements](requirements.txt).


## 7. Ferramentas de rastreamento
### 7.1. Aimstack

1. Instalar a ferramenta.

```bash
pip install aim
```

2. Executar MalSynGen com a opção -a ou --use_aim (Exemplo):
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --number_epochs 300 --k_fold 10 -a
```
3. Após o final da execução, utilize o comando aim up na pasta do MalSynGen.
```bash
aim up
```
Documentação Aimstack: https://aimstack.readthedocs.io/en/latest/


### 7.2. Mlflow

1. Instalar a ferramenta
   
```bash
pip install mlflow
```

2. Instanciar um servidor local na porta 6002

```bash
mlflow server --port 6002
```
3. Executar MalSynGen com a opção -ml ou --use_mlflow (Exemplo):
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --number_epochs 300 --k_fold 10 -ml
```

4. Após o fim da execução, acesse o endereço http://localhost:6002/ no seu navegador para visualizar os resultados.


Documentação Mlflow: https://mlflow.org/docs/latest/index.html


### 7.3. Tensorboard

1. Instalar a ferramenta

```bash
pip install tensorboard
```

2. Executar MalSynGen com a opção -tb ou --use_tensorboard (Exemplo):
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --number_epochs 300 --k_fold 10 -tb
```

3. Visualizar os resultados com o comando:
   
```bash
tensorboard --logdir=tensorboardfolder/ --port=6002
```

Documentação TensorBoard: https://www.tensorflow.org/tensorboard/get_started?hl=pt-br

