""" 
    File Name:          MoReL/data_prep.py
    Author:             Xiaotian Duan (xduan7)
    Email:              xduan7@uchicago.edu
    Date:               5/1/19
    Python Version:     3.5.4
    File Description:   

"""

# def dataframe_reform(file_path):
#     # df = df.apply(pd.to_numeric) ?
#
#     if file_path.endswith('.csv') \
#             or os.path.exists(file_path + '.csv'):
#         return
#
#     print(f'Processing {file_path} ...')
#     try:
#         # When column information is given
#         df = pd.read_csv(file_path,
#                          sep='\t',
#                          header=0,
#                          index_col=0,
#                          na_values='na')
#     except IndexError:
#         df = pd.read_csv(file_path,
#                          sep='\t',
#                          header=None,
#                          index_col=0,
#                          skiprows=1,
#                          na_values='na')
#         df.index.names = ['NAME']
#         df.columns = list(range(len(df.columns)))
#
#     df.to_csv(path_or_buf=file_path + '.csv',
#               float_format='%g',
#               na_rep='nan')

# dir = c.DATA_DIR + '/drug/'
# for file in os.listdir(dir):
#      file_name = os.fsdecode(file)
#      file_path = os.path.join(dir, file_name)
#      dataframe_reform(file_path)





# # Testing segment for cell data loading
# data_path = os.path.join(c.DATA_DIR, 'cell/')
# for data_type in CellDataType:
#     for subset_type in CellSubsetType:
#         for scaling_method in CellScalingMethod:
#             try:
#                 df = load_cell_data(data_path, data_type,
#                                     subset_type, scaling_method)
#                 print(df.head())
#             except FileExistsError as e:
#                 print(e)










