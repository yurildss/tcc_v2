#!/usr/bin/env python3
import os
from pathlib import Path

import pandas as pd
import wfdb


def download_mitbih(target_folder="data/raw/mitbih"):
    os.makedirs(target_folder, exist_ok=True)
    if any(f.endswith(".dat") for f in os.listdir(target_folder)):
        print(f"Dados já presentes em {target_folder}. Pulando download.")
        return target_folder

    database = "mitdb"
    wfdb.dl_database(database, dl_dir=target_folder)
    print("Download concluído!")
    return target_folder


def annotate_records(data_folder="data/raw/mitbih", output_folder="data/raw/mit-bih-arrhythmia-database-1.0.0"):
    os.makedirs(output_folder, exist_ok=True)
    records = [os.path.splitext(f)[0] for f in os.listdir(data_folder) if f.endswith(".dat")]
    records = sorted(set(records))

    for rec_name in records:
        print(f"Processando registro {rec_name}...")
        record = wfdb.rdrecord(os.path.join(data_folder, rec_name))
        signal = record.p_signal
        annotation = wfdb.rdann(os.path.join(data_folder, rec_name), 'atr')

        df = pd.DataFrame(signal, columns=[f"channel_{i}" for i in range(signal.shape[1])])
        df["sample #"] = range(len(df))
        df["type"] = "NC"

        for s, t in zip(annotation.sample, annotation.symbol):
            if s < len(df):
                df.loc[s, "type"] = t

        output_csv = os.path.join(output_folder, f"{rec_name}_labeled.csv")
        df.to_csv(output_csv, index=False)
        print(f"{rec_name}_labeled.csv criado!")

    return output_folder


def concat_records(csv_folder="data/raw/mit-bih-arrhythmia-database-1.0.0", output_csv="data/raw/mitbih_all_records.csv"):
    csv_files = sorted([os.path.join(csv_folder, f) for f in os.listdir(csv_folder) if f.endswith(".csv")])
    print(f"Concatenando {len(csv_files)} arquivos CSV...")

    df_list = []
    for file in csv_files:
        temp_df = pd.read_csv(file)
        record_name = os.path.splitext(os.path.basename(file))[0]
        temp_df["record"] = record_name
        df_list.append(temp_df)

    df_all = pd.concat(df_list, ignore_index=True)
    df_all.to_csv(output_csv, index=False)
    print(f"Arquivo único gerado: {output_csv}")
    return output_csv


def renumerate_samples(csv_path="data/raw/mitbih_all_records.csv", output_csv="data/raw/mitbih_all_records_renumerada.csv"):
    df = pd.read_csv(csv_path)
    df["sample #"] = range(len(df))
    df.to_csv(output_csv, index=False)
    print(f"Arquivo renumerado salvo em: {output_csv}")
    return output_csv


def filter_relevant(csv_path="data/raw/mitbih_all_records.csv", output_csv="data/raw/batimentos_filtrados.csv"):
    df = pd.read_csv(csv_path)
    tipos_relevantes = ["N", "L", "R", "V", "A", "F", "S", "Q", "/", "f", "NC"]
    df_filtrado = df[df["type"].isin(tipos_relevantes)]
    df_filtrado.to_csv(output_csv, index=False)
    print(f"Arquivo filtrado gerado: {output_csv}")
    return output_csv


def main():
    download_folder = download_mitbih()
    annotate_folder = annotate_records(download_folder)
    concat_csv = concat_records(annotate_folder)
    renumerate_samples(concat_csv)
    filter_relevant(concat_csv)


if __name__ == "__main__":
    main()
