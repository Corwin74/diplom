![](https://i.ibb.co/4p6dK9c/asset-v1-Skillfactory-DST-WEBINARS-MAY2020-type-asset-block-Data-Science.png)
# Дипломная работа
Репозиторий с дипломной работой<br/> 
Требуется решить задачу кредитного скоринга только на основании карточных транзакций клиента.

Особенности датасета:
1. Огромный объем: 1.5m объектов, 450m строк данных, 6gb данных.
2. Максимальная детализация данных: 19 признаков на каждую транзакцию, пользовательская история глубиной в год (до 8к транзакций на клиента).

#### Структура репозитория:


eda.ipynb - Exploratory Data Analyses<br/>
preprocessing.ipynb - ноутбук с препроцессингом транзакционных данных для нейроной сети<br/>
model.ipynb - ноутбук с обучением модели и предскзанием результата<br/>

|-- prototypes - промежуточные и тестовые варианты модулей<br/>
|-- constants - папка с полезными константами для препроцессинга <br/>
utils<br/>
|-- data_generators.py - содержит функционал для генерации батчей <br/>
|-- dataset_preprocessing_utils.py - методы для препроцессинга транзакционных данных <br/>
|-- pytorch_training.py - методы обучения, валидации и инференса модели на torch <br/>
|-- tf_training.py - методы обучения, валидации и инференса модели на tensorflow <br/>
|-- training_aux.py - реализация early_stopping-а <br/>
|-- utils.py - методы для пакетного чтения и предобработки данных<br/> 
