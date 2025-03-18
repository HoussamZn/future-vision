import io
import streamlit as st
import pandas as pd
import numpy as np


from models.file import readfile
from models.file import exportfile
from models.data import detect_columns_type
from models.file import change_header

from streamlit_option_menu import option_menu


def data():
    st.title("Data Dashboard")
    st.markdown(
        """
        <style>
        .stButton > button,.stDownloadButton > button {
            float: right;
        }
        

        .custom-download-button {
        margin-top: 29px; /* Adjust the top margin as needed */
        float : right;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

    #inti session variables
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'on_edit' not in st.session_state:
        st.session_state.on_edit = False
    if "on_update" not in st.session_state:
        st.session_state.on_update = False
    if 'init_mod_data' not in st.session_state:
        st.session_state.init_mod_data = 0


    if st.session_state.on_update == True :
        st.session_state.on_update = False
        st.toast('Data updated...',icon='✔')


    #sidebar
    with st.sidebar:
        options = ["Initialize Data", "Modify Data", "Data Preview", "Export Data"]
        sections = st.pills("Data sections", options, selection_mode="multi",default=options)
        st.divider()

    #on change functions
    def load_data():
        try :
            readed_df = readfile(st.session_state.data_path)
            st.session_state.data = change_header(readed_df)
            if st.session_state.data is None : st.error("Imported data doesnt have all columns names")
        except:
            st.session_state.data = None
        reset_preprocessing()

    def switch_init(option):
        if st.session_state[option] == 'Import data':
            st.session_state.init_mod_data = 0
        else :
            st.session_state.init_mod_data = 1

    def reset_preprocessing():
        st.session_state.preprocessing_progress = 0.0
        st.session_state.preprocessing_step = "First step : Handle Missing Values"

    #----------------------------------page start-----------------------------------

    if len(sections)==0:
        st.info('Select a section in the sidebar to display....')

    #Initialiiiize--------------------------------------------------------------------------------------------------------
    if "Initialize Data" in sections:
        st.header('Initialize Data',divider="blue",)
        selected = option_menu(None,["Import data", 'Create data'],default_index=st.session_state.init_mod_data,
            icons=['cloud-arrow-down-fill', 'file-earmark-plus-fill'], menu_icon="cast",orientation="horizontal",on_change=switch_init,key='init_mod_data_str',
            styles={"nav-link-selected": {"background-color": "#0068c9"},
                    "nav-item":{"margin":"0 5px"},
                    "nav-link":{"--hover-color":"#add8ff"}
                    })

        if st.session_state.init_mod_data == 0:
            #st.header('Import data',divider="blue")
            file = st.file_uploader("Import a data file ", type=["csv", "xls" , "xlsx" , "xlsm" ,'json'], key='data_path',on_change=load_data,accept_multiple_files=False)

        if st.session_state.init_mod_data == 1:
            load_data_columns_nb = None
            #st.header('Create data',divider="blue")

            st.divider()

            st.subheader('Number of columns')
            load_data_columns_nb_wg = st.number_input("Insert the number of columns :",value=5,min_value=1,label_visibility="collapsed")
            load_data_columns_nb = load_data_columns_nb_wg

            st.divider()

            st.subheader('Columns labels')
            max_columns = 6
            nb_in_line = int(np.ceil(load_data_columns_nb / max_columns))

            columns_names = list(range(load_data_columns_nb))
            for l,_ in enumerate(range(nb_in_line)):
                columns_name_grid = st.columns(max_columns)
                for i,x in enumerate(columns_name_grid) :
                    if (l*max_columns)+i < load_data_columns_nb:
                        with x:
                            columns_names[(l*max_columns)+i] = st.text_input(f"Column {(l*max_columns)+i+1}",value=f"Column {(l*max_columns)+i+1}")

            st.divider()

            st.subheader('Columns data types')
            columns_type = list(range(load_data_columns_nb))
            for l,_ in enumerate(range(nb_in_line)):
                columns_type_grid = st.columns(max_columns)
                for i,x in enumerate(columns_type_grid) :
                    if (l*max_columns)+i < load_data_columns_nb:
                        with x:
                            columns_type[(l*max_columns)+i] = st.selectbox(f"Column {(l*max_columns)+i+1}",options=["float",'int',"string","bool"])

            st.divider()

            data_grid = st.columns([3, 1])
            with data_grid[0]:
                st.subheader('Insert data')
            with data_grid[1]:
                submit = st.button("Create data")
            if submit :
                st.success('Data created',icon='✔')


            
            column_types_dict = dict(zip(columns_names, columns_type))
            editable_data = pd.DataFrame([[0] * load_data_columns_nb],columns=columns_names).astype(column_types_dict)
            editable_data_wg = st.data_editor(editable_data,num_rows="dynamic",use_container_width=True)
            if submit :
                st.session_state.data = editable_data_wg
                reset_preprocessing()
        


    #moddiifffyyyyyyy--------------------------------------------------------------------------------------------------------
    if "Modify Data" in sections:

        st.header('Modify Data',divider="blue")
        if st.session_state.data is None :
            st.info('Import or create a dataset to modify...')

        else :
            if st.session_state.on_edit == False:
                edit_bt = st.button("✎ Modify")
                if edit_bt : 
                    st.session_state.on_edit = True
                    st.rerun()
                st.dataframe(st.session_state.data,use_container_width=True)
                

            if st.session_state.on_edit == True:
                cancel_grid = st.columns([7,1])
                with cancel_grid[0] :
                    if st.button("✔ Save"):
                        st.session_state.on_update = True
                        #st.session_state.data = st.session_state.on_edit_data_wg
                        a = st.session_state.on_edit_data_wg.to_csv(index=False)
                        b = pd.read_csv(io.StringIO(a), sep=",")
                        st.session_state.data = b
                        reset_preprocessing()
                        st.session_state.on_edit = False
                        st.rerun()
                with cancel_grid[1] :
                    if st.button("✖ Cancel"):
                        st.session_state.on_edit = False
                        st.rerun()

                st.session_state.on_edit_data_wg=st.data_editor(st.session_state.data,num_rows="dynamic",use_container_width=True,key='on_edit_data')

            st.divider()

            add_grid = st.columns([3, 1])
            with add_grid[0]:
                st.subheader('Add column')
            with add_grid[1]:
                add_bt = st.button("➕ Add",disabled=st.session_state.on_edit == False)
            add_input_grid = st.columns(2)
            with add_input_grid[0]:
                add_column_name = st.text_input("Column name",value="Column",disabled=st.session_state.on_edit == False)
            if add_column_name in st.session_state.data.columns.tolist() and st.session_state.on_edit == True:
                st.warning('Column name already exist in data , it will be replaced by empty column with the selected data type if you didnt change the name', icon="⚠️")
            with add_input_grid[1]:
                add_column_type = st.selectbox("Column data type",options=["float",'int',"string","bool"],disabled=st.session_state.on_edit == False)
            if add_bt:
                add_column_type
                st.session_state.data[add_column_name] = pd.Series([0] * st.session_state.data.shape[0],dtype=add_column_type)
                reset_preprocessing()
                st.session_state.on_update = True
                st.rerun()

            st.divider()

            columns_grid = st.columns([30,1,30])
            with columns_grid[0]:
                delete_grid = st.columns([3, 1])
                with delete_grid[0]:
                    st.subheader('Delete columns')
                with delete_grid[1]:
                    delete_bt = st.button("🗑 Delete",disabled=st.session_state.on_edit == False)

                delete_columns = st.multiselect(
                    "Select columns to delete",
                    st.session_state.data.columns.to_list(),disabled=st.session_state.on_edit == False,)
                    
            if delete_bt and len(delete_columns)>0 :
                st.session_state.data.drop(delete_columns,axis='columns',inplace=True)
                st.session_state.on_update = True
                st.rerun()
            elif delete_bt:
                st.warning('No column selected', icon="⚠️")

            with columns_grid[2]:
                keep_grid = st.columns([3, 1])
                with keep_grid[0]:
                    st.subheader('Only Keep columns')
                with keep_grid[1]:
                    keep_bt = st.button("✔ Apply",disabled=st.session_state.on_edit == False)

                keep_columns = st.multiselect(
                    "Select columns to keep",
                    st.session_state.data.columns.to_list(),disabled=st.session_state.on_edit == False)
                
            if keep_bt and len(keep_columns)>0 :
                st.session_state.data = st.session_state.data[keep_columns]
                reset_preprocessing()
                st.session_state.on_update = True
                st.rerun()
            elif keep_bt:
                st.warning('No column selected', icon="⚠️")

            st.divider()

    #Preeeeevieeeew--------------------------------------------------------------------------------------------------------
    if "Data Preview" in sections:
        st.header('Data Preview',divider="blue")
        if st.session_state.data is not None and len(st.session_state.data.columns.to_list())>0:
            st.subheader("Data Summary")
            st.dataframe(st.session_state.data.describe(),use_container_width=True)


            a, b = st.columns(2)
            c, d = st.columns(2)

            a.metric("Number of rows",st.session_state.data.shape[0], border=True)
            b.metric("Number of columns",st.session_state.data.shape[1], border=True)

            c.metric("Number of duplicated rows",st.session_state.data.duplicated().sum(), border=True)
            d.metric("Number of rows that countain empty cells",st.session_state.data.isna().any(axis=1).sum(), border=True)

            st.divider()

            filter_grid = st.columns([3, 1])
            with filter_grid[0]:
                st.subheader("Filter Data")
            with filter_grid[1]:
                st.markdown(
                    """
                    <style>
                    .stButton > button,.stDownloadButton > button {
                        float: right;
                    }
                    

                    .custom-download-button {
                    margin-top: 29px; /* Adjust the top margin as needed */
                    float : right;
                    }

                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                filter_bt = st.button("✔ Apply filter on original data")

            feature_select_grid = st.columns(2)
            columns = st.session_state.data.columns.tolist()
            with feature_select_grid[0]:
                selected_column = st.selectbox("Select column to filter by", columns)
            unique_values = st.session_state.data[selected_column].unique()
            with feature_select_grid[1]:
                selected_value = st.selectbox("Select value", unique_values)

            filtered_df = st.session_state.data[st.session_state.data[selected_column] == selected_value]
            st.dataframe(filtered_df,use_container_width=True)

            if filter_bt:
                st.session_state.data = filtered_df
                reset_preprocessing()
                st.rerun()

            st.divider()

            count_bar_chart_grid = st.columns([3, 1])
            with count_bar_chart_grid[0]:
                st.subheader("Count bar chart")
            with count_bar_chart_grid[1]:
                count_bar_bt = st.button("✔ Generate bar chart")
                
            continuous_columns,_ = detect_columns_type(st.session_state.data)
            columns = list(set(st.session_state.data.columns.tolist())-set(continuous_columns))
            count_graph_column = st.selectbox("Select column to visualize",columns)

            if count_bar_bt and count_graph_column is not None:
                plot_data = st.session_state.data[count_graph_column].value_counts().reset_index()
                st.bar_chart(plot_data,x=count_graph_column,y='count')
            elif count_bar_bt:
                st.warning('No categorical column in your data', icon="⚠️")

            st.divider()

            line_char_grid = st.columns([3, 1])
            with line_char_grid[0]:
                st.subheader("Scatter chart")
            with line_char_grid[1]:
                st.markdown(
                    """
                    <style>
                    .stButton > button,.stDownloadButton > button {
                        float: right;
                    }
                    

                    .custom-download-button {
                    margin-top: 29px; /* Adjust the top margin as needed */
                    float : right;
                    }

                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                line_bt = st.button("✔ Generate scatter chart")
            columns = st.session_state.data.columns.to_list()
            axis_grid= st.columns(2)
            with axis_grid[0]:
                x_columns = st.selectbox("Select X-axis column",columns+['ID'])
            with axis_grid[1]:
                Y_columns = st.selectbox("Select Y-axis column",columns)

            if line_bt:
                if x_columns!='ID' :st.scatter_chart(st.session_state.data,x=x_columns,y=Y_columns)
                else :st.scatter_chart(st.session_state.data,y=Y_columns)
            st.divider()

        else :
            st.info('Import or create a dataset to preview...')

        


    #expooooooort--------------------------------------------------------------------------------------------------------
    if "Export Data" in sections:

        st.header('Export Data',divider="blue")
        if st.session_state.data is None :
            st.info('Import or create a dataset to export...')
        export_grid = st.columns([4, 4,1])
        with export_grid[0]:
            option = st.selectbox(
            "Export data as :",
            ("csv", "xlsx" , "xlsm",'json',"xml","xls"),
            disabled=st.session_state.data is None)

        with export_grid[1]:
            export_name = st.text_input("File name", "Data",disabled=st.session_state.data is None)

        try :export_par = exportfile(st.session_state.data,option)
        except : export_par=""
        with export_grid[2]:
            st.markdown('<div class="custom-download-button">', unsafe_allow_html=True)
            export_bt = st.download_button(
            label="🡻 Export",
            data=export_par,
            file_name=f"{export_name}.{option}",
            disabled=st.session_state.data is None)
            st.markdown('</div>', unsafe_allow_html=True)
        if export_bt:
            st.success('Data exported ',icon='✔')
        st.divider()

