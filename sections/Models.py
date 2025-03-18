import pickle
import streamlit as st
import pandas as pd
import numpy as np

from streamlit_option_menu import option_menu
from models.componants import title_with_bt
from models.machinelearning import get_prediction_type
from models.data import detect_columns_type
from models.machinelearning import get_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import plotly_express as px
from sklearn.metrics import mean_absolute_percentage_error,r2_score,mean_squared_error,root_mean_squared_error
from sklearn.preprocessing import StandardScaler

from models.file import Model
def models():
    st.title("Models Dashboard")
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
    st.session_state.on_create_model_obj_trained = False
    #inti session variables
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'init_mod' not in st.session_state:
        st.session_state.init_mod = 0

    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'columns' not in st.session_state:
        st.session_state.columns = None
    if 'target' not in st.session_state:
        st.session_state.target = None
    if 'on_create_model' not in st.session_state:
        st.session_state.on_create_model = None
    if 'on_create_model_obj' not in st.session_state:
        st.session_state.on_create_model_obj = None
    if "on_create_model_obj_trained" not in st.session_state:
        st.session_state.on_create_model_obj_trained = False
    if "model_class" not in st.session_state:
        st.session_state.model_class = None
    if "scaler" not in st.session_state:
        st.session_state.scaler = None
        
    if "label_encoder" not in st.session_state:
        st.session_state.label_encoder = None
    if "onhot_encoder" not in st.session_state:
        st.session_state.onhot_encoder = None
    

    if 'on_create_model_tp' not in st.session_state:
        st.session_state.on_create_model_tp = None


    def load_model():
        try :
            st.session_state.model_class = pickle.load(st.session_state.model_path)
        except:
            print('errrrrrroooooooooor')
            st.session_state.model_class = None

    def switch_init(option):
        if st.session_state[option] == 'Import model':
            st.session_state.init_mod = 0
        else :
            st.session_state.init_mod = 1



    with st.sidebar:
        options = ["Initialize Model", "Make a Prediction", "Export Model"]
        sections = st.pills("Models dahsboard sections", options, selection_mode="multi",default=options)
        st.divider()

    if len(sections)==0:
        st.info('Select a section in the sidebar to display....')

    if 'Initialize Model' in sections:
        st.header('Initialize Model',divider="blue")

        selected = option_menu(None,["Import model", 'Use new model'],default_index=st.session_state.init_mod,
            icons=['cloud-arrow-down-fill', 'file-earmark-plus-fill'], menu_icon="cast",orientation="horizontal",on_change=switch_init,key='init_mod_str',
            styles={"nav-link-selected": {"background-color": "#0068c9"},
                    "nav-item":{"margin":"0 5px"},
                    "nav-link":{"--hover-color":"#add8ff"}
                    })




        if st.session_state.init_mod == 0:
            file = st.file_uploader("Import a model file ", type=["pkl"], key='model_path',on_change=load_model,accept_multiple_files=False)

        if st.session_state.init_mod == 1:
            st.divider()

            st.subheader('Machine learning models')
            

            models_list = ["Logistic Regression","Decision Tree",'Naive Bayes',"Support Vector Machine","K-means",'K-nearest neighbor',"Random Forest","Neural Network"]
            model_grid = st.columns(4)
            with model_grid[0]:
                lr_bt = st.button(models_list[0],use_container_width=True)
                dt_bt = st.button(models_list[1],use_container_width=True)
            with model_grid[1]:
                nb_bt = st.button(models_list[2],use_container_width=True)
                svm_bt = st.button(models_list[3],use_container_width=True)
            with model_grid[2]:
                km_bt = st.button(models_list[4],use_container_width=True)
                knn_bt = st.button(models_list[5],use_container_width=True)
            with model_grid[3]:
                rf_bt = st.button(models_list[6],use_container_width=True)
                nn_bt = st.button(models_list[7],use_container_width=True)

            if lr_bt : 
                st.session_state.on_create_model = 0
                st.session_state.on_create_model_obj_trained = False
            if dt_bt : 
                st.session_state.on_create_model = 1
                st.session_state.on_create_model_obj_trained = False
            if nb_bt : 
                st.session_state.on_create_model = 2
                st.session_state.on_create_model_obj_trained = False
            if svm_bt : 
                st.session_state.on_create_model = 3
                st.session_state.on_create_model_obj_trained = False
            if km_bt : 
                st.session_state.on_create_model = 4
                st.session_state.on_create_model_obj_trained = False
            if knn_bt : 
                st.session_state.on_create_model = 5
                st.session_state.on_create_model_obj_trained = False
            if rf_bt : 
                st.session_state.on_create_model = 6
                st.session_state.on_create_model_obj_trained = False
            if nn_bt : 
                st.session_state.on_create_model = 7
                st.session_state.on_create_model_obj_trained = False
            
            

            #model parametres ----------------------------------------
            params = {}
            dafault_params = st.toggle("Default parametres",value=True)

            if not dafault_params and st.session_state.on_create_model in [0,1,3,4,5,6]:
                st.subheader("Model Parametres")
                st.info('Hint : leave the parametres to default value if you are not familiar with machine learning')
                if st.session_state.on_create_model == 0:
                    opt= st.columns([8,1,8,1,8])
                    with opt[0]:
                        penalty = st.selectbox("Penalty",options=["l1","l2",'elasticnet','None'],index=1,help="The type of regularization to apply to prevent overfitting")
                        if penalty == 'None' : penalty=None
                    with opt[2]:
                        max_iter = st.slider("Max iteration",min_value=50,max_value=1000,value=100,step=50,help="Maximum number of iterations for the solver to converge")
                    with opt[4]:
                        C =st.slider("C",min_value=0.1,max_value=2.0,value=1.0,step=0.1,help='Inverse of the regularization strength. Smaller values specify stronger regularization')
                    params = {"penalty":penalty,"max_iter":max_iter,'C':C}
                elif st.session_state.on_create_model == 1:
                    opt= st.columns([8,1,8,1,8])
                    with opt[0]:
                        criterion = st.selectbox("Criterion",options=["gini","entropy",'log_loss','squared_error','friedman_mse','absolute_error','poisson'],index=0,help="The function to measure the quality of a split")
                        if criterion in ["gini","entropy",'log_loss'] : st.session_state.on_create_model_tp=1
                        else : st.session_state.on_create_model_tp=2
                    with opt[2]:
                        max_depth = st.slider("Max depth",min_value=0,max_value=50,value=0,step=1,help="The maximum depth of the tree, Limits the number of splits to control overfitting, leave it to zero for no maximum depth")
                        if max_depth == 0 : max_depth=None
                    with opt[4]:
                        min_samples_split = st.slider("Minimum number of samples",min_value=2,max_value=50,value=2,step=1,help="The minimum number of samples required to split an internal node")
                    params = {"criterion":criterion,"max_depth":max_depth,'min_samples_split':min_samples_split}
                elif st.session_state.on_create_model == 3:
                    opt= st.columns([8,1,8,1,8])
                    with opt[0]:
                        kernel = st.selectbox("Kernel",options=["linear","poly",'rbf','sigmoid'],index=2,help="Specifies the kernel type to be used in the algorithm")
                    with opt[2]:
                        C =st.slider("C",min_value=0.1,max_value=2.0,value=1.0,step=0.1,help='Regularization parameter. Controls the trade-off between achieving a low error on the training data and minimizing the model complexity. Smaller values specify stronger')
                    with opt[4]:
                        max_iter = st.slider("Max iteration",min_value=0,max_value=1000,value=0,step=50,help="Maximum number of iterations to run the solver, keep it in 0 for no limit")
                        if max_iter == 0 : max_iter = -1
                    params = {"kernel":kernel,"C":C,'max_iter':max_iter}
                elif st.session_state.on_create_model == 4:
                    opt= st.columns([8,1,8])
                    with opt[0]:
                        n_clusters =st.slider("Clusters Number (K)",min_value=2,max_value=10,value=3,step=1,help='the number of clusters to form, i.e., the number of centroids to initialize')
                    with opt[2]:
                        max_iter = st.slider("Max iteration",min_value=100,max_value=1000,value=300,step=50,help="Maximum number of iterations the algorithm will run for a single initialization")
                    params = {"n_clusters":n_clusters,"max_iter":max_iter}
                elif st.session_state.on_create_model == 5:
                    opt= st.columns([8,1,8])
                    with opt[0]:
                        n_neighbors =st.slider("Number of neighbors)",min_value=1,max_value=20,value=5,step=1,help='The number of neighbors to use for predictions')
                    with opt[2]:
                        metric = st.selectbox("The distance metric",options=["euclidean","manhattan",'minkowski','chebyshev'],index=2,help="The distance metric used for finding nearest neighbors")
                    params = {"n_neighbors":n_neighbors,"metric":metric}
                elif st.session_state.on_create_model == 6:
                    opt_2= st.columns([8,1,8])
                    with opt_2[0]:
                        n_estimators = st.slider("Number of trees",min_value=50,max_value=500,value=100,step=50,help="The number of trees in the forest")
                    with opt_2[2]:
                        max_depth = st.slider("Max depth",min_value=0,max_value=50,value=0,step=1,help="The maximum depth of the tree, Limits the number of splits to control overfitting, leave it to zero for no maximum depth")
                        if max_depth == 0:max_depth=None
                    opt= st.columns([8,2,8,8])
                    with opt[0]:
                        criterion = st.selectbox("Criterion",options=["gini","entropy",'log_loss','squared_error','friedman_mse','absolute_error','poisson'],index=0,help="The function to measure the quality of a split")
                        if criterion in ["gini","entropy",'log_loss'] : st.session_state.on_create_model_tp=1
                        else : st.session_state.on_create_model_tp=2
                    with opt[2]:
                        st.write("Bootsrap")
                        bootstrap = st.toggle("Use bootsrap sampling",value=True,help="Whether bootstrap sampling is used when building treess")
                    with opt[3]:
                        min_samples_split = st.slider("Minimum number of samples",min_value=2,max_value=50,value=2,step=1,help="The minimum number of samples required to split an internal node")
                    params = {"n_estimators":n_estimators,"max_depth":max_depth,'criterion':criterion,'bootstrap':bootstrap,"min_samples_split":min_samples_split}


            st.divider()
            #to_data = title_with_bt('Training Data',"Data Dashboard")
            st.subheader('Training Data')
            #if to_data :
                #st.switch_page('pages/Data.py')
                #pass
            if st.session_state.data is not None and st.session_state.on_create_model is not None:
                st.dataframe(st.session_state.data,use_container_width=True)

                train_grid = st.columns([16,1,8])
                with train_grid[0]:
                    st.subheader("Target Columns")
                    target_grid = st.columns(2)
                    with target_grid[0]:
                        types = get_prediction_type(st.session_state.on_create_model,None if dafault_params else st.session_state.on_create_model_tp)
                        prediction_type = st.selectbox("Prediction type",types,disabled=len(types)<2)
                    continuous_columns,categorical_columns = detect_columns_type(st.session_state.data)
                    if prediction_type == 'Classification':
                        columns_to_select = list(categorical_columns)
                    elif prediction_type == 'Regression':
                        columns_to_select = list(continuous_columns)
                    elif prediction_type == 'Clustering':
                        columns_to_select = None
                    
                    with target_grid[1]:
                        target_column = st.selectbox("Target column",columns_to_select)
                    
                with train_grid[2]:
                    st.subheader("Test data size")
                    test_size = st.slider('Test data size',0,100,step=5,label_visibility='hidden',value=20,disabled=prediction_type == 'Clustering')

                if len(types) == 1 and types[0]=='Clustering':
                    st.info(f'Selected model can only make {types[0]} and it doesnt need to specify a target column...')
                elif len(types) == 1:
                    st.info(f'Selected model can only make {types[0]}...')
                
                if st.session_state.on_create_model_obj_trained == False:
                    st.session_state.on_create_model_obj = get_model(st.session_state.on_create_model,prediction_type,**params)

                st.divider()

                strat_train_bt = title_with_bt('Model Training',"Start training")
                display_grid = st.columns(2)
                with display_grid[0]:
                    st.metric("Selected Model",models_list[st.session_state.on_create_model], border=True)
                with display_grid[1]:
                    st.metric("Prediction type",prediction_type, border=True)
                if prediction_type != 'Clustering' :
                    with display_grid[0]:
                        st.metric("Target column",target_column, border=True)
                    with display_grid[1]:
                        st.metric("Test data size",f"{test_size}%", border=True)

                
                pre_processed = len(st.session_state.data.columns[st.session_state.data.isnull().any()].tolist()) == 0 and (len(st.session_state.data.select_dtypes(exclude=['number']).columns.tolist()) == 0 or (len(st.session_state.data.select_dtypes(exclude=['number']).columns.tolist()) == 1 and target_column in st.session_state.data.select_dtypes(exclude=['number']).columns.tolist()))
                one_value = len(st.session_state.data[target_column].unique()) > 1 if target_column is not None else True
                if strat_train_bt and (target_column is not None or prediction_type == 'Clustering') and one_value :
                    if pre_processed:
                        if prediction_type != 'Clustering':
                            x_train,x_test,y_train,y_test = train_test_split(st.session_state.data.drop(target_column,axis=1),st.session_state.data[target_column],test_size=test_size * 0.01,random_state=42)
                        with st.spinner('Training model...'):
                            if prediction_type != 'Clustering':
                                scaler = StandardScaler()
                                x_train = scaler.fit_transform(x_train)
                                st.session_state.scaler = scaler
                                x_test = scaler.transform(x_test)
                                st.session_state.on_create_model_obj.fit(x_train,y_train)
                            else :
                                scaler = StandardScaler()
                                x_train = st.session_state.data.copy()
                                x_train = scaler.fit_transform(x_train)
                                st.session_state.scaler = scaler
                                st.session_state.on_create_model_obj.fit(x_train)

                            st.session_state.model = st.session_state.on_create_model_obj
                            st.session_state.columns = st.session_state.data.columns.tolist()
                            if prediction_type != 'Clustering':
                                st.session_state.columns.remove(target_column)
                                st.session_state.target = target_column
                            else :
                                st.session_state.target = "Clusters"
                            column_types = dict(st.session_state.data.dtypes)
                            pre_column = dict(st.session_state.pre_column) if ('pre_column' in st.session_state and st.session_state.pre_column is not None) else dict(st.session_state.data.dtypes)
                            if prediction_type != 'Clustering':
                                del column_types[target_column]
                                if target_column in pre_column:
                                    del pre_column[target_column]
                            st.session_state.model_class = Model(st.session_state.model,pre_column,st.session_state.target,st.session_state.scaler,st.session_state.label_encoder,st.session_state.onhot_encoder)
                            st.session_state.on_create_model_obj_trained = True
                        st.success('Model trained ',icon='✔')
                    else :
                        st.warning("Data is not preprocessed , go to 'Preprocessing' section before training model", icon="⚠️")
                elif strat_train_bt and target_column is None :
                    st.warning('No target column selected', icon="⚠️")
                elif strat_train_bt and len(st.session_state.data[target_column].unique()) <= 1:
                    st.warning('Target column has only 1 unique value ', icon="⚠️")


                st.divider()

                st.subheader('Model Testing')

                if st.session_state.on_create_model_obj_trained == True:
                    st.markdown("#### Stats :")
                    if prediction_type == 'Classification':
                        y_pred = st.session_state.on_create_model_obj.predict(x_test)
                        accuracy = accuracy_score(y_test.values,y_pred)*100
                        precision = precision_score(y_test.values,y_pred,average='macro')*100
                        recall = recall_score(y_test.values,y_pred,average='macro')*100
                        F1score = f1_score(y_test.values,y_pred,average='macro')*100
                        c_matrix = confusion_matrix(y_test.values, y_pred)

                        test_con_grid = st.columns(2)
                        test_grid = st.columns(2)
                        with test_con_grid[0]:
                            with test_grid[0]:
                                st.metric("Accuracy",f'{accuracy:.0f}%', border=True)
                            with test_grid[1]:
                                st.metric("Precision",f'{precision:.0f}%', border=True)
                        with test_con_grid[1]:
                            with test_grid[0]:
                                st.metric("Recall",f'{recall:.0f}%', border=True)
                            with test_grid[1]:
                                st.metric("F1 Score",f'{F1score:.0f}%', border=True)
                    elif prediction_type == 'Regression' :
                        y_pred = st.session_state.on_create_model_obj.predict(x_test)
                        mse = mean_squared_error(y_test,y_pred)
                        mape = mean_absolute_percentage_error(y_test,y_pred)
                        rmse = root_mean_squared_error(y_test,y_pred)
                        r2 = r2_score(y_test,y_pred)

                        test_grid_reg = st.columns(2)
                        with test_grid_reg[0]:
                            st.metric("Mean Squared Error",f'{mse:.2f}', border=True)
                        with test_grid_reg[1]:
                            st.metric("Mean Absolute Percentage Error",f'{mape:.2f}', border=True)
                        with test_grid_reg[0]:
                            st.metric("Root Mean Squared Error",f'{rmse:.2f}', border=True)
                        with test_grid_reg[1]:
                            st.metric("R-squared",f'{r2:.2f}', border=True)
                    elif prediction_type == 'Clustering':
                        inertia = st.session_state.model_class.model.inertia_
                        from sklearn.metrics import silhouette_score
                        silhouette = silhouette_score(st.session_state.data, st.session_state.model_class.model.labels_)

                        test_grid_reg = st.columns(2)
                        with test_grid_reg[0]:
                            st.metric("Inertia (Within-Cluster Sum of Squares)",f'{inertia:.2f}', border=True)
                        with test_grid_reg[1]:
                            st.metric("Silhouette Score",f'{silhouette:.2f}', border=True)

                    if prediction_type == 'Classification':
                        graphs_grid = st.columns([5,1,5])
                        with graphs_grid[0]:
                            st.markdown("#### Confusion Matrix :")
                            C_matrix_fig = px.imshow(
                            c_matrix,
                            text_auto=True,  # Automatically add text annotations
                            color_continuous_scale="Blues",
                            labels=dict(x="Predicted", y="Actual", color="Count"),
                            x=[f"Predicted {x}" for x in st.session_state.data[target_column].unique().tolist()],  # Customize x-axis labels
                            y=[f"Actual {x}" for x in st.session_state.data[target_column].unique().tolist()]         # Customize y-axis labels
                            )
                            C_matrix_fig.update_layout(
                            xaxis_title="Predicted Labels",
                            yaxis_title="Actual Labels"
                            )
                            st.plotly_chart(C_matrix_fig)

                        with graphs_grid[2]:
                            st.markdown("#### Metrics Histogram :")
                            metrics = {
                                "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                                "Value (%)": [accuracy, precision, recall, F1score]
                            }
                            metrics_fig = px.bar(
                                metrics,
                                x="Metric",
                                y="Value (%)",
                                text="Value (%)",
                                color="Metric",
                                labels={"Value (%)": "Percentage (%)"},
                            )
                            metrics_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                            metrics_fig.update_layout(yaxis_range=[0, 100]) 
                            st.plotly_chart(metrics_fig)
                    elif prediction_type=='Regression' :
                        show_regression_df = pd.DataFrame(y_pred,columns=["Predicted Values"])
                        show_regression_df["Real Values"] = y_test.values
                        show_regression_df["Difference"] = np.abs(y_pred-y_test.values)
                        st.dataframe(show_regression_df,use_container_width=True)

                else :
                    st.info('Train model to see it testing result...')


            elif st.session_state.data is None :
                st.info('Import or create a dataset to train model...')
            elif st.session_state.on_create_model is None :
                st.info('Select a model to train...')
            
    if 'Make a Prediction' in sections:
        st.header('Make a Prediction',divider="blue")
        if st.session_state.model_class is not None:
            st.subheader("Prediction Data")
            pred_data = pd.DataFrame(columns=st.session_state.model_class.columns.keys()).astype(st.session_state.model_class.columns)

            label = []
            if st.session_state.model_class.label_encoder is not None:
                for column,encoder in st.session_state.model_class.label_encoder.items():
                    if pred_data.dtypes[column] == 'bool' :
                        add = pd.DataFrame({column:[False]})
                    else :
                        add = pd.DataFrame({column:[encoder.classes_.tolist()[0]]}).astype(pd.CategoricalDtype(encoder.classes_))
                    label.append(add)


            if st.session_state.model_class.onehot_encoder is not None :
                onhot_dict = dict(zip(st.session_state.model_class.onehot_encoder[1], st.session_state.model_class.onehot_encoder[0].categories_))
                for column,values in onhot_dict.items():
                    add = pd.DataFrame({column:[values[0]]}).astype(pd.CategoricalDtype(values))
                    label.append(add)

            if st.session_state.model_class.label_encoder is not None or st.session_state.model_class.onehot_encoder is not None:
                label = pd.concat(label,axis=1)
                pred_data[label.columns] = label
            pred_data_edit = st.data_editor(pred_data,num_rows="dynamic",use_container_width=True)
            
            predict_bt = title_with_bt("Results","Predict")
            if predict_bt and not pred_data_edit.isnull().values.any():
                pred_data = pred_data_edit.copy()
            
                if st.session_state.model_class.label_encoder is not None:
                    for column,encoder in st.session_state.model_class.label_encoder.items():
                        pred_data[column] = encoder.transform(pred_data[column])
                    
                if st.session_state.model_class.onehot_encoder is not None:
                    onhot_encoded = st.session_state.model_class.onehot_encoder[0].transform(pred_data[st.session_state.model_class.onehot_encoder[1]])
                    encoded_df = pd.DataFrame(onhot_encoded, columns=st.session_state.model_class.onehot_encoder[0].get_feature_names_out(st.session_state.model_class.onehot_encoder[1]))
                    pred_data = pd.concat([pred_data.drop(st.session_state.model_class.onehot_encoder[1], axis=1), encoded_df], axis=1)

                pred_data_scled = st.session_state.model_class.scaler.transform(pred_data)
                y = st.session_state.model_class.model.predict(pred_data_scled)
                pred_data_edit[st.session_state.model_class.target]=y
                st.dataframe(pred_data_edit,use_container_width=True)
            elif predict_bt:
                st.warning('You cant predict on a dataframe that contains None values, Please insert the missing values',icon="⚠️")
            
        else :
            st.info('Import or train a new model to make predictions...')


    if 'Export Model' in sections:
        st.header('Export Model',divider="blue")
        if st.session_state.model_class is not None:
            export_pickle = pickle.dumps(st.session_state.model_class)
            export_grid = st.columns([3,1])
            with export_grid[0]:
                export_name = st.text_input("Model name",placeholder='Model',value='Model')
            with export_grid[1]:
                export_bt = st.download_button(
                label="🡻 Export",
                data=export_pickle,
                file_name=f"{export_name}.pkl",
                use_container_width=True
                )
        else :
            st.info('Import or train a new model to export...')
