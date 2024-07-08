## Artefatos apêndice SBSEG 2024: #243359: MalSynGen: redes neurais artificiais na geração de dados tabulares sintéticos para detecção de malware
A MalSynGen  é uma ferramenta que utiliza redes neurais artificiais para gerar dados sintéticos tabulares para o domínio de malware Android.
Para avaliar sua performance foram aumentados os dados de dois datasets, considerando métricas de fidelidade estatística e utilidade. 
Os resultados indicam que MalSynGen é capaz de capturar padrões representativos para o aumento de dados tabulares.

### 1. Selos Considerados
Os autores julgam como considerados no processo de avaliação os selos:

**Selo D - Artefatos Disponíveis** 
Justificativa: Repositório  anônimo  GitHub público com documentação da ferramente e modulos

**Selo F - Artefatos Funcionais**
Justificativa: Artefatos funcionais e testados em Ubuntu 22.04 e 20.04 e Debian  11 e 12

**Selo R - Artefatos Reprodutíveis**
Justificativa: São disponbilizados scripts para reprodução dos experimentos detalhados no paper.

**Selo S- Artefatos Sustentáveis**
Justificativa: Código inteligível e acompanhado com boa documentação.


## 2. Informações básicas
Os códigos da utilizados para a execução ferramenta 
MalSynGen  estão disponibilizados no repositório GitHub https://github.com/MalwareDataLab/SynTabData. Neste encontram-se um README.md contendo sobre o fluxo de execução da ferramenta,configuração, parâmetros de entrada e instalação em ambeintes :
-*bare metal* (testado em  Ubuntu 22.04 com Python 3.10.12,3.8.10 e 3.8.2. E Debian 11 e 12 com python  3.9.2)
-*Google Collab* 
-*containers* Docker (testado em versões 24.0.7 e 20.10.5)


### 2.1. Dependências
Testamos o código da ferramentas com as seguintes versões python:
- Python 3.8.10
- Python 3.9.2
- Python 3.10.12
- Python 3.8.2 

O código da MalSynGen possui dependências com diversos pacotes e bibliotecas Python.
No entanto, as principais são:
numpy 1.21.5, Keras 2.9.0, Tensorflow 2.9.1, pandas 1.4.4, scikit-learn 1.1.1. e mlflow 2.12.1.
Ademais, a lista extensa das dependências encontra-se no arquivo [requirements.txt.](https://github.com/SBSegSF24/MalSynGen/blob/07c602b7a43a3cd2bf305a684759a45c4e7cc2f1/requirements.txt)


## 3. Instalação 
Para a instalação da ferramenta MalSynGEn siga os seguintes comandos
**1.** Para realizar a instalação do código fonte utilize o comando:
```bash
git clone https://github.com/SBSegSF24/MalSynGen.git
  ```
**2.** Acesse o diretório clonado
```bash
cd MalSynGen/
```
**3.** Execute o script de instalação
```bash
./Install.sh
```
Alternativamente
```bash
./scripts/docker_build.sh
./scripts/docker_run_solo.sh
```
A instalação manual e em outros ambientes está detalhada no README.md do repositório GitHub.

## 4. Datasets
O diretório datasets contem  os datasets balanceados KronoDroid_emulator e KronoDroid_real_device[^1] utilizados no artigo, assim
como o código utilizado para balancear estes datasets. Além dos arquivos de validação de cada dataset e código de validação utilizado no subdiretório validation.
[^1]: https://github.com/aleguma/kronodroid



## 5. Ambiente de testes
A ferramenta foi testada nos seguintes ambientes: 

- Hardware:AMD Ryzen 7 5800x,8 cores, 64 GB RAM. Software: Ubuntu Server versão 22.04, python 3.8.10, docker 24.07
- Hardware:Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz,8 cores, 75 GB RAM. Software: Debian GNU 11, python 3.9.2, docker 20.10.5
-  Hardware:Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz,8 cores, 75 GB RAM. Software: Debian GNU 12, python 3.11.2, docker 24.07

## 6.Teste mínimo
Teste funcional rápido utilizando o dataset kronodroid R, com 20000 amostras e 286 características, com 300 epocas em 10 folds. 
```bash
./run_demo_app.sh
```
## 7. Experimentos
Para a reprodução dos experimentos executados no artigo utilize os seguintes comandos:
```bash
./Reproduction.sh
```
Alternativamente executar:
```bash
pipenv run python3 run_campaign.py -c SF24_4096_2048_1    
```

