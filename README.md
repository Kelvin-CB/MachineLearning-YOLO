### Author: Kelvin Carvalho Bomfim

## Seleção Voxar: Projeto de Machine Learning utilizando Modelo YOLO 
* Revisão do funcionamento de modelos YOLO, hiperparâmetros, optimização, loss function, taxa de aprendizagem.
* Implementação do modelo YOLOv5 pré treinado com o dataset COCO128.
* Treinamento e teste de modelos customizados a partir de diferentes datasets. 
* Pré-Processamento dos dados com o RoboFlow. 
* Análise do treinamento em tempo real com WandB. 

## Código e recursos utilizados 
**Python Version:** 3.9  
**Libs:** pandas, numpy, matplotlib, pyTorch(CPU to train/GPU to predict), OpenCV                                                          
**COCO128:** https://www.kaggle.com/datasets/ultralytics/coco128                                                         
**BCCD:** https://github.com/Shenggan/BCCD_Dataset                                                         
**Snail Custom Dataset:** /SnailDataset                                                       
**LabelImg:** https://github.com/tzutalin/labelImg                                                       
**RoboFlow:** https://roboflow.com/                                                       
**WandB:** https://https://wandb.ai/                                                       

## Módulo Básico: 1
O modelo escolhido para o case, foi o YOLOv5 pré-treinado no dataset COCO128, uma versão reduzida do COCO Dataset, o mesmo mostrou-se de fácil manipulação e bem intuitivo, o teste foi feito nas imagens pré disponíveis do mesmo:

## Módulo Básico: 2
O backgroud por trás dos modelos YOLO se dá através da subdivisão de uma imagem em N pedaços, formando um grid SxS = N, cada um desses n pedaços é responsavel pela detecção e localização de objetos no seu conteúdo. Como a detecção e reconhecimento acontecem simultaneamente para cada N pedaço, o consumo de poder computacional é reduzido, mas em consequência, ocorre um aumento no numero de predições duplicadas (Ex: um objeto estar subdivido entre mais de um N pedaço), este problema é contornado através do uso de Non Maximal Suppression (IoU), dessa forma conseguindo reduzir o numero de predições duplicadas sem utilizar valores máximo globais como parâmetro.

## Módulo Básico: 3
Implementação do modelo:

## Módulo Básico: 4
Implementação do modelo para parâmetros de teste alterados:

## Módulo Especifico: TREINAMENTO DETECÇÃO 2D
* Snail Dataset:
Para a realização deste módulo, foi criado um dataset personalizado chamado de Snail Dataset, contendo 40 frames retirados de diferentes episódios do cartoon americano "Hora de Aventura", frames esses que contém pequenos "easter eggs" de uma lesma.
O dataset foi subdivido para 28 imagens de treinamento, 8 imagens de validação e 4 imagens de teste.
A rotulação de cada um dos frames foi feito através da ferramenta LabelImg, onde cada lesma foi selecionada e rotulada manualmente.

* BCCD YOLO Dataset:
Com o fracasso do treinamento do Snail Dataset, o foco da detecção foi alterado para o âmbito biológico, então o novo dataset a ser utilizado foi o bccD (Blood Cell Counting & Detection), contendo 360 imagens no total (240 de treino, 60 de validação e 60 de teste), compreendendo 3 classes de objetos: White Blood Cells, Red Blood Cells e Platelets

## Módulo Especifico: Pré-Processamento
Após a implementação e treinamento "vanilla" do modelo, foram utilizados diversos métodos para tentar aumentar acurácia das detecções, para tal utilizou-se da ferramenta Roboflow para gerar novos dados através de Data Augmentation.
O processo se dá quando criam-se novas imagens que podem ser utilizadas para treinamento e/ou validação, através de distorções das imagens já existentes, esse processo além de aumentar a quantidade de dados, aumenta também os graus de liberdade de posição no qual o modelo é capaz de reconhecer as classes.

* Snail Dataset:
Inicio: 28 de treino, 8 de validação e 4 de teste
Depois de aplicar Data Augmentation:  84 de treino, 8 de validação e 4 de teste
Pré Processamento: Auto-Orient + Stretch to 416x416
Data Augmentation: Flip: Horizontal, Vertical + 90° Rotate: Clockwise, Counter-Clockwise, Upside Down + Crop: 0% Minimum Zoom, 43% Maximum Zoom + Saturation: Between -67% and +67% + Mosaic + Bounding Box: Flip: Horizontal, Vertical

* BCCD Dataset:
Inicio: 240 de treino, 60 de validação e 60 de teste
Depois de aplicar Data Augmentation: 720 de treino, 60 de validação e 60 de teste
Pré Processamento: Auto-Orient
Data Augmentation: Flip: Horizontal, Vertical + 90° Rotate: Clockwise, Counter-Clockwise + Rotation: Between -30° and +30° + Shear: ±15° Horizontal, ±15° Vertical

## Módulo Especifico: Fine Tuning
Para o acompanhamento em tempo real do treinamento de cada um dos modelos, foi utilizada a ferramenta WandB, ferramenta que permite acompanhar a evolução de métricas enquanto o modelo ainda segue em treinamento, a partir dela podemos também comparar diferentes resultados de diferentes treinamentos e datasets.

## Performance do Modelo
BCCD Vanilla:
*	**Precision** : 0.56264
*	**box_loss**: 0.04872
*	**cls_loss**: 0.00785
*	**obj_loss**: 0.14802

BCCD Augmented:
*	**Precision** : 0.78559
*	**box_loss**: 0.03540
*	**cls_loss**: 0.00268
*	**obj_loss**: 0.14197

Snail Dataset Vanilla:
*	**Precision** : 0.00194
*	**box_loss**: 0.11223
*	**cls_loss**: 0
*	**obj_loss**: 0.02152

Snail Dataset Augmented:
*	**Precision** : 0.03351
*	**box_loss**: 0.08506
*	**cls_loss**: 0
*	**obj_loss**: 0.02645

## Conclusão 
O algoritmo YOLO se mostra ser de fácil implementação e manipulação, sendo capaz de reconhecer e localizar diferentes classes de objetos em uma imagem com ligeira facilidade e pouco poder de processamento, apesar de ter dificuldade em reconhecer objetos pequenos em imagens "desenhos 2d", apresenta sim uma boa precisão e confiabilidade em predições. 


