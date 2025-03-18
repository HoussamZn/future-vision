import streamlit as st
import pandas as pd
import numpy as np

from models.componants import title_with_bt
from models.data import detect_columns_type
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from models.data import detect_continuous_columns
import plotly_express as px
from imblearn.over_sampling import SMOTE



def preprocessing():
    st.title("Pre-Processing Dashbord")
    st.markdown(
        """
        <style>
        .stButton > button,.stDownloadButton > button {
            float: right;
        }
        
        .stDownloadButton > button {
        margin-top: 28px; /* Adjust the top margin as needed */
        float : right;
        }
        

        </style>
        """,
        unsafe_allow_html=True,
    )


    #inti session variables
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'preprocessing_progress' not in st.session_state:
        st.session_state.preprocessing_progress = 0.0
    if "preprocessing_step" not in st.session_state:
        st.session_state.preprocessing_step = "First step : Handle Missing Values"
    if "on_update" not in st.session_state:
        st.session_state.on_update = False
    if 'selected_column_to_balance' not in st.session_state:
        st.session_state.selected_column_to_balance = None

    if st.session_state.on_update == True :
        st.session_state.on_update = False
        st.toast('Data updated...',icon='✔')

    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = None
    if "onhot_encoder" not in st.session_state:
        st.session_state.onhot_encoder = None
    if "pre_column" not in st.session_state:
        st.session_state.pre_column = None
    if "encoded" not in st.session_state:
        st.session_state.encoded = False


    if st.session_state.data is not None:
        my_bar = st.progress(st.session_state.preprocessing_progress,"Progress")
        if st.session_state.preprocessing_progress == 0.0:
            columns_with_nulls = st.session_state.data.columns[st.session_state.data.isnull().any()].tolist()
            next_step_bt = title_with_bt(st.session_state.preprocessing_step,"Next step")
            if len(columns_with_nulls)==0:
                st.info('No missing values...')
            else :
                missing_values_grid = st.columns([3,3,2])
                with missing_values_grid[0]:
                    selected_column = st.selectbox("Select column",columns_with_nulls)
                continuous_columns,categorical_columns=detect_columns_type(st.session_state.data,threshold=10)
                with missing_values_grid[1]:
                    if selected_column in categorical_columns:
                        methods = ["most_frequent","constant"]
                    else : 
                        methods = ["most_frequent","constant","mean",'median']
                    selected_method = st.selectbox("Select Methods",methods)
                with missing_values_grid[2]:
                    try :
                        st.session_state.data[selected_column]=st.session_state.data[selected_column].astype(pd.BooleanDtype())
                    except:
                        pass
                    column_type = st.session_state.data[selected_column]
                    if pd.api.types.is_integer_dtype(column_type):
                        constant =st.number_input("Constant to replace missing values",value=0,disabled=selected_method!="constant")
                    elif pd.api.types.is_float_dtype(column_type):
                        constant =st.number_input("Constant to replace missing values",value=0.0,disabled=selected_method!="constant")
                    elif pd.api.types.is_bool_dtype(column_type) :
                        st.caption("Constant to replace missing values")
                        constant =st.toggle(f"{selected_column}",value=False,disabled=selected_method!="constant")
                    elif pd.api.types.is_string_dtype(column_type) or pd.api.types.is_object_dtype(column_type):
                        constant =st.text_input("Constant to replace missing values",value="Value",disabled=selected_method!="constant")

                handle_missing_bt = st.button("Apply",use_container_width=True)
                if handle_missing_bt:
                    data_with_types = st.session_state.data.dtypes
                    imputer = SimpleImputer(strategy=selected_method,fill_value=constant if selected_method=="constant" else None)
                    st.session_state.data[selected_column] = imputer.fit_transform(st.session_state.data[[selected_column]]).ravel()
                    st.session_state.data[selected_column] = st.session_state.data[selected_column].astype(data_with_types[selected_column])
                    st.session_state.on_update = True
                    st.rerun()

            if next_step_bt:
                st.session_state.preprocessing_progress+=0.33
                st.session_state.preprocessing_step="Second step : Encode categorical variables"
                st.rerun()
                
        if st.session_state.preprocessing_progress == 0.33:
            data_grid = st.columns([5, 1,1])
            with data_grid[0]:
                st.subheader(st.session_state.preprocessing_step)
            with data_grid[1]:
                previous_step_bt = st.button("Previous step")
            with data_grid[2]:
                next_step_bt = st.button("Next step")

            
            non_numeric_columns = st.session_state.data.select_dtypes(exclude=['number']).columns.tolist()
            if len(non_numeric_columns) == 0:
                st.info('All columns are encoded , or has only number values...')
                if st.session_state.pre_column is None: st.session_state.pre_column = dict(st.session_state.data.dtypes)
            elif st.session_state.encoded==False:
                st.session_state.pre_column = dict(st.session_state.data.dtypes)
                to_encode_columns = [x for x in non_numeric_columns if len(st.session_state.data[x].unique())>2]
                non_numeric_columns = list(set(non_numeric_columns)-set(to_encode_columns))

                show_list = [x for x in st.session_state.data.columns.tolist() if x in non_numeric_columns or x in to_encode_columns]
                to_show_df = pd.DataFrame([["One-Hot" if x in to_encode_columns else "Label" for x in show_list]],columns=show_list)
                st.divider()
                to_encode_bt = title_with_bt("Columns that require encoding",'Apply encoding')
                st.dataframe(to_show_df,hide_index=True,use_container_width=True)
                selected_encode = st.multiselect("Columns to encoded",options=show_list,default=show_list,help="Usually you should not encode the target column")




                if to_encode_bt:

                    label_encoder_list = {}
                    for column in non_numeric_columns:
                        if column in selected_encode:
                            label_encoder = LabelEncoder()
                            st.session_state.data[column] = label_encoder.fit_transform(st.session_state.data[column])
                            label_encoder_list[column] = label_encoder
                            st.session_state.label_encoder = label_encoder_list

                    to_encode_columns = list(set(to_encode_columns) - (set(show_list)-set(selected_encode)))
                    encoder = OneHotEncoder(sparse_output=False)
                    encoded = encoder.fit_transform(st.session_state.data[to_encode_columns])
                    st.session_state.onhot_encoder = [encoder,to_encode_columns]
                    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(to_encode_columns))
                    st.session_state.data = pd.concat([st.session_state.data.drop(to_encode_columns, axis=1), encoded_df], axis=1)
                    st.session_state.on_update = True
                    st.session_state.encoded=True
                    st.rerun()
            else :
                st.success("Data endoded successfully...")
                if st.session_state.pre_column is None: st.session_state.pre_column = dict(st.session_state.data.dtypes)

            if previous_step_bt:
                st.session_state.preprocessing_progress-=0.33
                st.session_state.preprocessing_step="First step : Handle Missing Values"
                st.rerun()
            if next_step_bt:
                st.session_state.preprocessing_progress+=0.33
                st.session_state.preprocessing_step="Last step : Balance the Dataset"
                st.rerun()

        if st.session_state.preprocessing_progress == 0.66:
            data_grid = st.columns([5, 1,1])
            with data_grid[0]:
                st.subheader(st.session_state.preprocessing_step)
            with data_grid[1]:
                previous_step_bt = st.button("Previous step")
            with data_grid[2]:
                next_step_bt = st.button("Next step")

            continuous_columns,categorical_columns = detect_columns_type(st.session_state.data)
            if len(categorical_columns) > 0:
                selected_column = st.selectbox("Select column to balance",options=categorical_columns,index=categorical_columns.index(st.session_state.selected_column_to_balance) if st.session_state.selected_column_to_balance is not None else 0)
                balance_grid = st.columns(2)
                
                with balance_grid[0]:
                    category_counts = st.session_state.data[selected_column].value_counts().reset_index()
                    category_counts.columns = [selected_column, 'Count']
                    fig2 = px.bar(category_counts, x=selected_column, y='Count', 
                        title=f'Bar chart of distribution of {selected_column} (Count)', 
                        color=selected_column,  # Optional: color bars by category
                        color_discrete_sequence=px.colors.qualitative.Set2)  # Optional: color palette
                    fig2.update_layout(xaxis_title=selected_column, yaxis_title='Count',showlegend=False)
                    fig2.update_traces(marker_showscale=False)
                    st.plotly_chart(fig2)

                with balance_grid[1]:
                    fig = px.pie(st.session_state.data, names=selected_column, title=f'Bar chart of distribution of {selected_column} (Count)', 
                        color=selected_column, 
                        color_discrete_sequence=px.colors.qualitative.Set2)
                    fig.update_traces(textinfo='percent+label')
                    st.plotly_chart(fig)

                balance_bt = st.button("Balance Data",use_container_width=True)
                if balance_bt:
                    st.session_state.selected_column_to_balance = selected_column
                    smote = SMOTE(random_state=42)
                    X, Y = smote.fit_resample(st.session_state.data.drop(selected_column,axis=1), st.session_state.data[selected_column])
                    X[selected_column] = Y
                    st.session_state.data = X
                    st.rerun()
            else :
                st.info('No categorical column to balance...')

            if next_step_bt:
                st.session_state.preprocessing_progress=1.0
                st.session_state.preprocessing_step="Preprocessing is done !"
                st.rerun()
            if previous_step_bt:
                st.session_state.preprocessing_progress-=0.33
                st.session_state.preprocessing_step="Second step : Encode categorical variables"
                st.rerun() 

        if st.session_state.preprocessing_progress == 1:
            data_grid = st.columns([5, 1])
            with data_grid[0]:
                st.subheader(st.session_state.preprocessing_step)
            with data_grid[1]:
                previous_step_bt = st.button("Previous step")
            #with data_grid[2]:
                #next_step_bt = st.button("Models section")
            st.success('Data preprocessed succefuly',icon='✔')

            if previous_step_bt:
                st.session_state.preprocessing_progress=0.66
                st.session_state.preprocessing_step="Last step : Balance the Dataset"
                st.rerun()
            #if next_step_bt:
                #st.switch_page('pages/Models.py')
                #pass


        st.divider()
        st.subheader("Data Preview")
        st.dataframe(st.session_state.data,use_container_width=True)

    else :
        st.info('Import or create a data to preprocces...')
        if False :
            to_data_grid = st.columns([4,1])
            with to_data_grid[1]:
                to_data = st.button("Data Dashboard")

            if to_data :
                #st.switch_page('pages/Data.py')
                pass
