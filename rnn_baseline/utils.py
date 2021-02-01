import os
import pandas as pd
import tqdm


def read_parquet_dataset_from_local(path_to_dataset, start_from=0,
                                     num_parts_to_read=2, columns=None, verbose=False):
    """
    читает num_parts_to_read партиций, преобразует их к pd.DataFrame и возвращает
    :param path_to_dataset: путь до директории с партициями
    :param start_from: номер партиции, с которой начать чтение
    :param num_parts_to_read: количество партиций, которые требуется прочитать
    :param columns: список колонок, которые нужно прочитать из партиции
    :return: pd.DataFrame
    """

    res = []
    dataset_paths = sorted([os.path.join(path_to_dataset, filename) for filename in os.listdir(path_to_dataset) 
                              if filename.startswith('part')])
    
    start_from = max(0, start_from)
    chunks = dataset_paths[start_from: start_from + num_parts_to_read]
    if verbose:
        print('Reading chunks:\n')
        for chunk in chunks:
            print(chunk)
    for chunk_path in chunks:
        chunk = pd.read_parquet(chunk_path,columns=columns)
        res.append(chunk)
    return pd.concat(res).reset_index(drop=True)


def eda_value_counts(path_to_dataset, partitions, column=None):
    """
    Читает столбец column из partitions партиций
    и возвращает суммарный результат функции
    value_counts над ним
    """
    summator = []
    for i in range(0, partitions):
        df_temp = read_parquet_dataset_from_local(path_to_dataset, start_from=i, num_parts_to_read=1, columns=[column])
        summator.append(df_temp.value_counts())
    summator = pd.concat(summator, axis=1).fillna(0.0).astype(int)
    return summator.sum(axis=1)
