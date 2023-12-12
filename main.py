import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st


if __name__ == '__main__':
    df = pd.read_csv('datasets/D_merge.csv').drop('Unnamed: 0', axis=1)
    st.title('EDA для данных о клиентах банка')
    st.write('Данные о клиентах хранятся в следующей таблице:')
    st.write(df.head())

    font_ticks = {'family': 'serif', 'color': 'black', 'size': 16}
    font_title = {'family': 'serif', 'color': 'black', 'size': 22}
    font_labels = {'family': 'serif', 'color': 'black', 'size': 20}

    # распределения признаков
    st.subheader('Распределения числовых признаков')

    fig, ax = plt.subplots(9, 1, figsize=(12, 56), dpi=180)
    real_columns = ['AGE', 'CHILD_TOTAL', 'DEPENDANTS', 'OWN_AUTO', 'WORK_TIME', 'PERSONAL_INCOME', 'CREDIT', 'TERM', 'FST_PAYMENT']
    for i in range(0, 9):
        ax[i].hist(df[real_columns[i]], color='indianred', bins=15)
        ax[i].set_xlabel(real_columns[i], fontdict=font_labels)
        ax[i].set_ylabel('COUNT', fontdict=font_labels)

        ax[i].set_xticklabels(ax[i].get_xticklabels(), fontdict=font_ticks)
        ax[i].set_yticklabels(ax[i].get_yticklabels(), fontdict=font_ticks)

        ax[i].grid(color='gray', linestyle='--', linewidth=0.25)
    st.pyplot(fig)


    st.subheader('Распределения категориальных признаков')

    fig, ax = plt.subplots(11, 1, figsize=(16, 80), dpi=200)
    cat_columns1 = ['TARGET', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
                   'FL_PRESENCE_FL', 'GEN_TITLE', 'JOB_DIR', 'FAMILY_INCOME', 'CLOSED_FL']
    for i in range(0, 11):
        ax[i].barh(df[cat_columns1[i]].unique(), df[cat_columns1[i]].value_counts().values, color='indianred')

        ax[i].set_title(cat_columns1[i], fontdict=font_labels)
        ax[i].set_xlabel('COUNT', fontdict=font_labels)

        ax[i].set_xticklabels(ax[i].get_xticklabels(), fontdict=font_ticks)
        ax[i].set_yticklabels(ax[i].get_yticklabels(), fontdict=font_ticks)

        ax[i].grid(color='gray', linestyle='--', linewidth=0.25)
    st.pyplot(fig)


    fig, ax = plt.subplots(4, 1, figsize=(18, 48), dpi=200)
    cat_columns2 = ['REG_ADDRESS_PROVINCE', 'FACT_ADDRESS_PROVINCE', 'POSTAL_ADDRESS_PROVINCE', 'GEN_INDUSTRY']
    for i in range(0, 4):
        ax[i].barh(df[cat_columns2[i]].unique()[:10], df[cat_columns2[i]].value_counts().values[:10], color='indianred')

        ax[i].set_title(cat_columns2[i], fontdict=font_labels)
        ax[i].set_xlabel('COUNT', fontdict=font_labels)

        ax[i].set_xticklabels(ax[i].get_xticklabels(), fontdict=font_ticks)
        ax[i].set_yticklabels(ax[i].get_yticklabels(), fontdict=font_ticks)

        ax[i].grid(color='gray', linestyle='--', linewidth=0.25)
    st.pyplot(fig)

    # матрица корреляций
    st.header('Матрица корреляций числовых признаков')

    fig, ax = plt.subplots(1, 1, figsize=(16, 16), dpi=200)
    plot = sns.heatmap(df[real_columns].corr(), cmap="YlGnBu", annot=True)
    st.pyplot(plot.get_figure())

    # зависимость целевой переменной от признаков
    st.header('Зависимость целевой переменной от некоторых категориальных признаков')
    fig, ax = plt.subplots(4, 1, figsize=(18, 64), dpi=180)

    target_columns = ['GENDER', 'EDUCATION', 'MARITAL_STATUS', 'FAMILY_INCOME']
    for i in range(4):
        x = np.array((df.groupby(target_columns[i])['TARGET'].count() - df.groupby(target_columns[i])['TARGET'].sum()).keys())
        y_0 = (df.groupby(target_columns[i])['TARGET'].count() - df.groupby(target_columns[i])['TARGET'].sum()).values
        y_1 = (df.groupby(target_columns[i])['TARGET'].sum()).values

        X_axis = np.arange(len(x))

        ax[i].bar(X_axis - 0.2, y_0, 0.4, label='Нет отклика')
        ax[i].bar(X_axis + 0.2, y_1, 0.4, label='Есть отклик')

        ax[i].set_xticks(X_axis, x, rotation='vertical', fontdict=font_ticks)
        ax[i].set_yticklabels(ax[i].get_yticklabels(), fontdict=font_ticks)
        ax[i].set_xlabel(target_columns[i], fontdict=font_labels)
        ax[i].set_ylabel("COUNT", fontdict=font_labels)
        ax[i].legend(fontsize="16")

        plt.tight_layout()

    st.pyplot(fig)


    # числовые характеристики распределения числовых столбцов
    st.header('Числовые характеристики распределения числовых столбцов')
    st.write(df.describe())