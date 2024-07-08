## Artefatos apêndice SBSEG 2024: #243359: MalSynGen: redes neurais artificiais na geração de dados tabulares sintéticos para detecção de malware
A MalSynGen  é uma ferramenta que utiliza redes neurais artificiais para gerar dados sintéticos tabulares para o domínio de malware Android.
Para avaliar o seu desempenho foram aumentados os dados de dois datasets, considerando métricas de fidelidade estatística e utilidade. 
Os resultados indicam que MalSynGen é capaz de capturar padrões representativos para o aumento de dados tabulares.

## 1. Selos Considerados
Os autores julgam como considerados no processo de avaliação os seguintes selos:

**Selo D - Artefatos Disponíveis**:
Justificativa: Repositório  anônimo disponível no GitHub público com documentação da ferramente e módulos.

**Selo F - Artefatos Funcionais**:
Justificativa: Artefatos funcionais e testados em Ubuntu 22.04 (*bare metal* e Docker) e 20.04 (Docker) e Debian  11 (*bare metal*) e 12 (*bare metal*).

**Selo R - Artefatos Reprodutíveis**:
Justificativa: São disponibilizados scripts para reprodução dos experimentos detalhados no artigo.[script1](https://github.com/SBSegSF24/MalSynGen/blob/07ccc905a5a48af6cb8d9d9b426e1d5abc65a718/reproduzir_sf24.sh);[script2](https://github.com/SBSegSF24/MalSynGen/blob/e71cec8b62a395ca282528912f21b279e64992c8/Reproduction.sh).

**Selo S- Artefatos Sustentáveis**:
Justificativa: Código inteligível e acompanhado com boa documentação.


## 2. Informações básicas
Os códigos utilizados para a execução ferramenta MalSynGen, estão disponibilizados no repositório GitHub https://github.com/SBSegSF24/MalSynGen.git. Nesse repositório encontram-se um README.md contendo informações sobre o fluxo de execução da ferramenta, configuração, parâmetros de entrada e instalação nos seguintes ambientes :

-*Bare metal* (testado em  Ubuntu 22.04 com Python 3.10.12,3.8.10 e 3.8.2. E Debian 11 e 12 com python  3.9.2)

-*Google Collab* 

-*Containers* Docker (testado em Docker versões 24.0.7 e 20.10.5 com imagem Ubuntu 20.04 e 22.04)


### 2.1. Dependências
Testamos o código da ferramenta com as seguintes versões Python:
- Python 3.8.10
- Python 3.9.2
- Python 3.10.12
- Python 3.8.2 

O código da MalSynGen possui dependências com diversos pacotes e bibliotecas Python.
Entre elas, as principais são:
numpy 1.21.5, Keras 2.9.0, Tensorflow 2.9.1, pandas 1.4.4, scikit-learn 1.1.1. e mlflow 2.12.1.
Ademais, a lista extensa das dependências encontra-se no arquivo [requirements.txt.](https://github.com/SBSegSF24/MalSynGen/blob/07c602b7a43a3cd2bf305a684759a45c4e7cc2f1/requirements.txt)


## 3. Instalação 
Para instalar a ferramenta MalSynGen siga os seguintes comandos

**1.** Para instalar as dependências na máquina local, utilize o comando:
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
Alternativamente, para construir uma imagem Docker localmente a partir do Dockerfile.
```bash
./scripts/docker_build.sh
```
A instalação manual e em outros ambientes está detalhada no README.md do repositório GitHub.

## 4. Datasets
O diretório [datasets](https://github.com/SBSegSF24/MalSynGen/tree/0669acd4855a7e268eba045346cf526def3acade/datasets) contém  os datasets balanceados KronoDroid_emulator e KronoDroid_real_device[^1] utilizados no artigo, assim como o código utilizado para balancear esses datasets. O diretório também contém os arquivos de validação de cada dataset e código de validação utilizado no subdiretório [validation](https://github.com/SBSegSF24/MalSynGen/tree/0669acd4855a7e268eba045346cf526def3acade/datasets/validation).
[^1]: https://github.com/aleguma/kronodroid



## 5. Ambiente de testes
A ferramenta foi testada nos seguintes ambientes: 

-**Hardware**:AMD Ryzen 7 5800x, 8 cores, 64 GB RAM. **Software**: Ubuntu Server 22.04.2 e 22.04.3, Python 3.8.10 e 3.10.12, Docker 24.07.

-**Hardware**:Intel Core i7-9700 CPU 3.00GHz, 8 cores, 16 GB RAM. **Software**: Debian GNU 11 e 12, Python 3.9.2 e 3.11.2, Docker 20.10.5 e 24.07.

## 6. Teste mínimo
Teste funcional rápido utilizando o dataset kronodroid R, com 20000 amostras e 286 características, com 100 épocas em 2 folds. Leva 5 minutos num computador AMD Ryzen 7 5800x, 8 cores, 64GB RAM.
```bash
./run_demo_app.sh
```
Para a execução do mesmo teste no ambiente docker execute o script docker_run_solo.sh com o comando.
```bash
 sudo ./scripts/docker_run_solo.sh
```
## 7. Experimentos
Para reproduzir os experimentos executados no artigo utilize o seguinte comando:
```bash
 ./reproduzir_sf24.sh
```
Os resultados serão salvos na pasta campanhas_SF24/. Para acessar essa pasta use o seguinte comando:
```bash
cd campanhas_SF24/
```

