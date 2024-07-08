## Preparação e Execução
### Instalação
1. Clonar o repositório e execute os seguintes comandos
   ```bash
    git clone https://github.com/SBSegSF24/MalSynGen.git
    cd MalSynGen
   ./Install.sh
   ```

#### (ALTERNATIVA AO INSTALL.SH) Configurar o pipenv

```
pip install pipenv
```
```
pipenv install -r requirements.txt
```


Execução básica:
```
pipenv python3 run_campaign.py
```

Exemplo de execução de uma campanha pré-configurada:

```
pipenv run python3 run_campaign.py -c Kronodroid_r

```
#### (ALTERNATIVA AO INSTALL.SH)  Utilizar um virtual enviroment (venv) para a execução dos experimentos:
#### Configurar Venv
```
python3 -m venv .venv
```
```
source .venv/bin/activate
```
```
pip3 install -r requirements.txt
```
Exemplo de execução de uma campanha pré-configurada:

```
python3 run_campaign.py -c Kronodroid_e
```


2. Executar a demonstração de funcionamento da ferramenta: 

   **Opção 1**: instalar as dependências e executar a aplicação em um ambiente Linux.
   ```bash
   ./run_demo_app.sh
   ```
     
   **Opção 2**: construir uma imagem Docker localmente a partir do Dockerfile e instanciar um container.
   
   ```bash
   ./scripts/docker_build.sh
   ./scripts/docker_run_solo.sh
   ```
   
   **Opção 3**: Executar o docker demo que instancia uma versão reduzida do experimento.
   ```bash
   ./run_demo_docker.sh
   ```

    
4. Executar os mesmos experimentos (campanhas) do paper

   ```bash
    ./run_sf24_experiments.sh
    ```

## Fluxo de execução 
![Screenshot from 2024-07-05 17-00-39](https://github.com/SBSegSF24/MalSynGen/assets/174879323/4d55117e-4203-4930-a0f5-2ed19c8857e3)

O fluxo de execução da ferramenta consiste de três etapas:

   **Seleção de dataset**: Nesta etapa,  realizamos o balanceamento pela classe minoritária, atravẽs do uso de técnicas de subamostragem. Os datasets balanceados e o código utilizado nesse processo se encontram em: https://github.com/SBSegSF24/MalSynGen/tree/accbe69f12bbf02d5d7f9c75291a60a5738bbb67/datasets

 O dataset balanceado é então processado nas etapas de treinamento e avaliação, através validação cruzada por meio de k-dobras (do inglês k-folds) onde são criados dois subconjuntos: subconjunto de avaliação (Dataset r) e subconjunto de treino (Dataset R)

  **Treinamento**: Nessa etapa, a cGANs é treinada  e utilizada cGANs para gerar dados sintéticos, precisamos treinar classificadores para posteriormente verificarmos a utilidade dos dados sintéticos gerados: Dataset S (gerado a partir de R) e Dataset s (gerado a paritr de r).  Os classificadores utilizados são denominados TR-As(Treinado com dataset R, avaliado com s) e TS-Ar(Treinado com S, avaliado com r).

   **Avaliação**: Esta etapa consiste da  execução e subsquente extração de métricas de utilidade dos classificadores, e fidelidade dos sintéticos atravês de uma comparação entre s e r. Por fim, verificamos se a utilidade dos dados sintéticos é fiel à utilidade dos dados reais através de testes como o de Wilcoxon no final da execução de dobras.


## Executando os datasets balanceados
O script em bash Interactive_execution_datasets.sh é reponsavel pela execução de todos os datasets balanceados.

Executar o script: 


   ```bash
   bash Interactive_execution_datasets
   ```


### Executando outros experimentos

A ferramenta conta com o **run_campaign.py** para automatizar o treinamento e a avaliação da cGAN. O **run_campaign.py** permite executar várias campanhas de avaliação com diferentes parâmetros, registrando os resultados em arquivos de saída para análise posterior. O usuário poderá visualmente realizar uma análise comparativa das diferentes configurações em relação aos conjuntos de dados utilizados.



Mesma campanha (Kronodroid_r) sendo executada diretamente na aplicação (**main.py**):
```
pipenv run python main.py --verbosity 20 --input_dataset datasets/kronodroid_real_device-balanced.csv --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500 --k_fold 10 --num_samples_class_benign 10000 --num_samples_class_malware 10000 --training_algorithm Adam
```



###  Parâmetros dos testes automatizados:

      --------------------------------------------------------------

    --campaign ou -c:    Especifica a campanha de avaliação que você deseja executar. 
                         Você pode fornecer o nome de uma campanha específica ou uma  
                         lista de campanhas separadas por vírgula. 
                         Por exemplo: --campaign SF24_4096_2048_10 ou --campaign 
                          Kronodroid_e.

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


## Executando a ferramenta no Google Colab

```
from google.colab import drive
drive.mount('/content/drive')
```

```
!pip install -r requirements.txt
```
```
input_file_path = "/content/dataset.csv"
```

```
!python main.py -i "$input_file_path" 
```

Obs.: Lembre-se de ter Models, Tools e a main devidamente importada no seu drive.
 <td><img src=https://github.com/SBSegSF24/MalSynGen/assets/174879323/6d1ccd21-242d-4cbb-a66e-ecd4660bc4d7 style="max-width:100%;"></td>




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

## Ambientes de teste

A ferramenta foi executada e testada na prática nos seguinte ambiente:

   
. Linux Ubuntu 22.04.2 LTS<br/>
   Kernel Version = 5.15.109+<br/>
   Python = 3.8.10 <br/>
   Módulos Python conforme [requirements](requirements.txt).


## Feramentas de rastreamento
**Aimstack**

1. Instalar a ferramenta

   ```bash
   pip install aim
   ```

2. Executar MalSynGen com a opção -a ou --use_aim

3. Executar o comando aim up na pasta do MalSynGen

Documentação Aimstack: https://aimstack.readthedocs.io/en/latest/
      
**Mlflow**

1. Instalar a ferramenta
   
   ```bash
   pip install mlflow
   ```

2. Instanciar um servidor local na porta 6002
   
   ```bash
   mlflow server --port 6002
   ```
3. Executar MalSynGencom a opção -ml ou --use_mlflow   

4. Acessar o endereço http://localhost:6002/ no seu navegador para visualizar os resultados

Documentação Mlflow: https://mlflow.org/docs/latest/index.html

**Tensorboard**

1. Instalar a ferramenta

  ```bash
  pip install tensorboard
  ```

2. Executar MalSynGen com a opção -tb ou --use_tensorboard

3. Visualizar os resultados com o comando
   
  ```bash
  tensorboard --logdir=tensorboardfolder/ --port=6002
  ```

Documentação TensorBoard: https://www.tensorflow.org/tensorboard/get_started?hl=pt-br
