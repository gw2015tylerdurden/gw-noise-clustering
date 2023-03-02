import base64
from io import BytesIO
import numpy as np
import dash
import h5py
import pandas as pd
import plotly.express as px
import umap
from dash import dcc, html
from dash.dependencies import Input, Output
from PIL import Image
from sklearn.cluster import SpectralClustering

random_state = 256

def numpyTob64FmtPng(array):
    im_pil = Image.fromarray(array)
    buff = BytesIO()
    im_pil.save(buff, format="png")
    im_b64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    return im_b64

def getGravitySpyDatasetIndex(hoverData):
    label = hoverData['points'][0]['curveNumber']
    num = hoverData['points'][0]['pointNumber']
    return sum(each_label_data_num[:label]) + num

# datasetをpdDataFrameに格納
dataset_path = './data/trainingset.h5'
label_indies = pd.read_csv('./data/gravity_spy_labels.csv', index_col=0, header=None)
iic_pred_labels = pd.read_csv('./data/sc_pred_labels.csv', index_col=0, header=None)
z_autoencoder = pd.read_csv('./data/z-autoencoder-outputs.csv', index_col=0)


z_umap = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.5, random_state=random_state).fit_transform(z_autoencoder)

col_name = ['umap-component-1', 'umap-component-2', 'umap-component-3']
# 3d scatterの引数に対応したpdDataFrameに格納
df = pd.DataFrame(z_umap, index=z_autoencoder.index, columns=col_name)

# 各クラスのデータ数を取得, マウスホバー時に該当するデータを探査する際に使用
# [328, 232, 58, 1869, 66, 454, 279, 830, 573, 657, 453, 181, 88, 27, 453, 285, 459, 354, 116, 472, 44, 305]
each_label_data_num = z_autoencoder.index.value_counts(sort=False).values.tolist() # sortすると降順になってしまう

sc = SpectralClustering(n_clusters=26, random_state=random_state, gamma=2.0, assign_labels="kmeans", affinity="rbf").fit(z_umap)

# default表示
fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2], color=df.index)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id="image", figure=fig),
        html.Div([
            dcc.Checklist(
                id='color-toggle',
                options=[{'label': 'Spectral Clustering Labels', 'value': 'index'}],
                value=[],
                inputStyle={"margin-right": "5px", "margin-left": "5px"}
            )
        ], style={'display': 'inline-block'})
    ]),
    html.Div(id="output"),
])

# Define the new callback triggered by the button
@app.callback(
    Output('image', 'figure'),  # Output is the figure of the dcc.Graph component
    Input('color-toggle', 'value'),  # Input is the number of times the button is clicked
)
def switch_color(value):
    if 'index' in value:
        # If the toggle button is checked, switch the color parameter to sc.labels_
        new_fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2], color=sc.labels_)

    else:
        # If the toggle button is unchecked, switch the color parameter back to df.index
        new_fig = px.scatter_3d(df, x=col_name[0], y=col_name[1], z=col_name[2], color=df.index)


    new_fig.update_traces(marker_size=1, mode='markers', marker=dict(showscale=False))
    new_fig.update_layout(
        height=900,
        font=dict(size=18),
        legend=dict(font=dict(size=20))
    )

    return new_fig


# マウスホバーしたデータを画像表示する関数をcallbackとして登録
@app.callback(Output('output', 'children'), [Input('image', 'hoverData')])
def displayImage(hoverData):
    if hoverData:
        idx = getGravitySpyDatasetIndex(hoverData)
        images = [h5_handle[all_image_paths[idx]][i] for i in range(4)] # [0, 1, 2, 3] -> 0.5, 1.0, 2.0, 4.0 sec
        spacer = np.zeros((images[0].shape[0], 10)).astype(np.uint8)  # 10ピクセル分の隙間を作成する, uint8はpng変換で必要
        target_image = np.concatenate([images[0], spacer, images[1], spacer, images[2], spacer, images[3]], axis=1)
        im_b64 = numpyTob64FmtPng(target_image)
        value = 'data:image/png;base64,{}'.format(im_b64)
        return html.Img(src=value, height='200px')
    return None

# h5の全てのファイルパスを取得
all_image_paths = []
with h5py.File(dataset_path) as f:
    f.visit(lambda key : all_image_paths.append(key) if isinstance(f[key], h5py.Dataset) else None)
h5_handle = h5py.File(dataset_path, 'r')

app.run_server(debug=True) # debug:エラー時などweb上で確認ができる
