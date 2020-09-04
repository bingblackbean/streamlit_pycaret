"""
in terminal run: streamlit run main.py
in another terminal run: mlflow ui
"""

import streamlit as st
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import os
import pycaret.classification as pc_cl
import pycaret.regression as pc_rg
import mlflow




def get_model_training_logs(n_lines = 10):
    file = open('logs.log', 'r')
    lines = file.read().splitlines()
    file.close()
    return lines[-n_lines:]


ML_TASK_LIST = ['回归', '分类']
RG_MODEL_LIST = ['lr', 'svm', 'rf', 'xgboost', 'lightgbm']
CL_MODEL_LIST = ['lr', 'dt', 'svm', 'rf', 'xgboost', 'lightgbm']


def list_files(directory, extension):
    # list certain extension files in the folder
    return [f for f in os.listdir(directory) if f.endswith('.' + extension)]


def concat_file_path(file_folder, file_selected):
    # handle the folder path with '/' or 'without './'
    # and concat folder path and file path
    if str(file_folder)[-1] != '/':
        file_selected_path = file_folder + '/' + file_selected
    else:
        file_selected_path = file_folder + file_selected
    return file_selected_path

@st.cache(suppress_st_warning=True)
def load_csv(file_selected_path, nrows):
    # load certain rows
    try:
        if nrows == -1:
            df = pd.read_csv(file_selected_path)
        else:
            df = pd.read_csv(file_selected_path, nrows=nrows)
    except Exception as ex:
        df = pd.DataFrame([])
        st.exception(ex)
    return df


def app_main():
    st.title("自动化机器学习平台")
    if st.sidebar.checkbox('定义数据源'):
        file_folder = st.sidebar.text_input('文件夹', value="data")
        data_file_list = list_files(file_folder, 'csv')
        if len(data_file_list) ==0:
            st.warning(f'当路径无可用数据集')
        else:
            file_selected = st.sidebar.selectbox(
                '选择文件', data_file_list)
            file_selected_path = concat_file_path(file_folder, file_selected)
            nrows = st.sidebar.number_input('行数', value=-1)
            n_rows_str = '全部' if nrows == -1 else str(nrows)
            st.info(f'已选择文件：{file_selected_path}，读取行数为{n_rows_str}')
    else:
        file_selected_path = None
        nrows = 100
        st.warning(f'当前选择文件为空，请选择。')
    if st.sidebar.checkbox('探索性分析'):
        if file_selected_path is not None:
            if st.sidebar.button('一键生成报告'):
                df = load_csv(file_selected_path, nrows)
                pr = ProfileReport(df, explorative=True)
                st_profile_report(pr)
        else:
            st.info(f'没有选择文件，无法进行分析。')

    if st.sidebar.checkbox('快速建模'):
        if file_selected_path is not None:
            task = st.sidebar.selectbox('选择任务', ML_TASK_LIST)
            if task == '回归':
                model = st.sidebar.selectbox('选取模型', RG_MODEL_LIST)
            elif task == '分类':
                model = st.sidebar.selectbox('选取模型', RG_MODEL_LIST)
            df = load_csv(file_selected_path, nrows)
            try:
                cols = df.columns.to_list()
                target_col = st.sidebar.selectbox('选取预测对象', cols)
            except BaseException:
                st.sidebar.warning(f'数据格式无法正确读取')
                target_col = None

            if target_col is not None and st.sidebar.button('训练模型'):
                if task == '回归':
                    st.success(f'数据预处理。。。')
                    pc_rg.setup(
                        df,
                        target=target_col,
                        log_experiment=True,
                        experiment_name='ml_',
                        log_plots=True,
                        silent=True,
                        verbose=False,
                        profile=True)
                    st.success(f'数据预处理完毕。')
                    st.success(f'训练模型。。。')
                    pc_rg.create_model(model, verbose=False)
                    st.success(f'模型训练完毕。。。')
                    #pc_rg.finalize_model(model)
                    st.success(f'模型已经创建')
                elif task == '分类':
                    st.success(f'数据预处理。。。')
                    pc_cl.setup(
                        df,
                        target=target_col,
                        fix_imbalance=True,
                        log_experiment=True,
                        experiment_name='ml_',
                        log_plots=True,
                        silent=True,
                        verbose=False,
                        profile=True)
                    st.success(f'数据预处理完毕。')
                    st.success(f'训练模型。。。')
                    pc_cl.create_model(model, verbose=False)
                    st.success(f'模型训练完毕。。。')
                    #pc_cl.finalize_model(model)
                    st.success(f'模型已经创建')
    if st.sidebar.checkbox('查看系统日志'):
        n_lines =st.sidebar.slider(label='行数',min_value=3,max_value=50)
        if st.sidebar.button("查看"):
            logs = get_model_training_logs(n_lines=n_lines)
            st.text('系统日志')
            st.write(logs)
    try:
        all_runs = mlflow.search_runs(experiment_ids=0)
    except:
        all_runs = []
    if len(all_runs) != 0:
        if st.sidebar.checkbox('预览模型'):
            ml_logs = 'http://kubernetes.docker.internal:5000/  -->开启mlflow，命令行输入:mlflow ui'
            st.markdown(ml_logs)
            st.dataframe(all_runs)
        if st.sidebar.checkbox('选择模型'):
            selected_run_id = st.sidebar.selectbox('从已保存模型中选择',all_runs[all_runs['tags.Source']=='create_model']['run_id'].tolist())
            selected_run_info = all_runs[(
                all_runs['run_id'] == selected_run_id)].iloc[0, :]
            st.code(selected_run_info)
            if st.sidebar.button('预测数据'):
                model_uri = f'runs:/' + selected_run_id + '/model/'
                model_loaded = mlflow.sklearn.load_model(model_uri)
                df = pd.read_csv(file_selected_path, nrows=nrows)
                #st.success(f'模型预测中。。。   ')
                pred = model_loaded.predict(df)
                pred_df = pd.DataFrame(pred, columns=['预测值'])
                st.dataframe(pred_df)
                pred_df.plot()
                st.pyplot()
    else:
        st.sidebar.warning('没有找到训练好的模型')
if __name__ == '__main__':
    app_main()

