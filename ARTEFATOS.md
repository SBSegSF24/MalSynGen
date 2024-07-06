## Ferramenta MalSynGen
A MalSynGen  é uma ferramenta que utiliza redes neurais artificiais para gerar dados sintéticos tabulares para o domínio de malware Android.
Para avaliar sua performance foram aumentados os dados de dois datasets, considerando métricas de fidelidade estatística e utilidade. 
Os resultados indicam que MalSynGen é capaz de capturar padrões representativos para o aumento de dados tabulares.





### Datasets
O diretório datasets contem  os datasets balanceados KronoDroid_emulator e KronoDroid_real_device[^1] utilizados no artigo, assim
como o código utilizado para balancear estes datasets. Além dos arquivos de validação de cada dataset e código de validação utilizado no subdiretório validation.
[^1]: https://github.com/aleguma/kronodroid


### Dependências
O código da MalSynGen possui dependências com diversos pacotes e bibliotecas Python.
No entanto, as principais são:
numpy 1.21.5, Keras 2.9.0, Tensorflow 2.9.1, pandas 1.4.4, scikit-learn 1.1.1. e mlflow 2.12.1.
Ademais, a lista extensa das dependências encontra-se no arquivo requirements.txt.

### Ambiente de testes
Utilizamos um servidor AMD Ryzen 7 5800x como processador de 8 cores e 64 GB de memória RAM para execução dos experimentos. 
O sistema operacional do servidor é o Ubuntu Server versão 22.04.

### Reprodutubalidade 
Para a reprodução dos experimentos executados no artigo utilize os seguintes comandos:
```bash
pipenv run python3 main.py -i datasets/kronodroid_real_device-balanced.csv  --num_samples_class_benign 10000 --num_samples\_class_malware 10000 --batch\_size 256 --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500--k_fold 10
pipenv run python3 main.py -i datasets/kronodroid_real_emulador-balanced.csv  --num_samples_class_benign 10000 --num_samples_class_malware 10000 --batch_size 256 --dense_layer_sizes_g 4096 --dense_layer_sizes_d 2048 --number_epochs 500--k_fold 10
```
Alternativamente executar:
```bash
pipenv run python3 run_campaign.py -c SF24_4096_2048_1    
```
